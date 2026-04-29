# 📋 Presentación Técnica: Caficulbot - Sistema MLOps Offline para Caficultores

## Contenido Ejecutivo

**Caficulbot** es un asistente de inteligencia artificial multimodal diseñado específicamente para caficultores colombianos. Funciona **completamente offline** (sin necesidad de internet), proporciona respuestas en tiempo real sobre cultivo, plagas, enfermedades y gestión de fincas, y se ejecuta en hardware de bajo costo (laptops con GPU de 8GB).

---

## 1. El Problema que Resuelve

### Contexto
Los caficultores colombianos enfrentan desafíos técnicos diarios:
- **Diagnóstico de plagas**: ¿Es roya, antracnosis o algo más?
- **Información en tiempo real**: Necesitan respuestas instantáneas, a menudo sin conectividad
- **Idioma**: Requieren respuestas en español y contextalizadas al café colombiano
- **Accesibilidad**: Muchos no tienen conexión confiable a internet en la finca

**Solución tradicional (antes de Caficulbot)**:
- Llamadas a agronomistas (costoso)
- Búsquedas en internet (requiere conexión)
- Libros y manuales (no son dinámicos)

**Solución Caficulbot**:
- IA especializada offline
- Respuestas inmediatas 24/7
- Soporta voz, texto e imágenes
- Cero costos de conectividad

---

## 2. Arquitectura Técnica

### 2.1 Componentes Principales

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│                   USUARIO (Caficultor)                  │
│                                                          │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ Web Browser o Aplicación
                        ↓
         ┌──────────────────────────────────┐
         │   STREAMLIT WEB (puerto 8501)    │
         │   - Chat interface               │
         │   - Micrófono (voz)             │
         │   - Cámara (fotos)              │
         │   - Upload de archivos          │
         └──────────────┬───────────────────┘
                        │
                        │ HTTP POST /ask
                        │ (texto + imagen)
                        ↓
         ┌──────────────────────────────────┐
         │    MAIN API (puerto 8000)        │
         │    - FastAPI                     │
         │    - Carga modelo Gemma-3N       │
         │    - Orquestación (routing)      │
         │    - GPU/CPU detection           │
         └──────────────┬───────────────────┘
                        │
          ┌─────────────┼──────────────┬────────────────┐
          │             │              │                │
          ↓             ↓              ↓                ↓
      ┌────────┐   ┌────────┐   ┌────────┐       ┌────────┐
      │ Inv... │   │ Gastos │   │Cosecha │  ...  │Ingresos│
      │ 8001   │   │ 8002   │   │ 8003   │       │ 8004   │
      └────┬───┘   └────┬───┘   └────┬───┘       └────┬───┘
           │             │              │                │
           └─────────────┴──────────────┴────────────────┘
                         │
                         ↓
          ┌─────────────────────────────┐
          │   SQLite Databases          │
          │  - inventario.db            │
          │  - gastos.db                │
          │  - cosecha.db               │
          │  - ingresos.db              │
          └─────────────────────────────┘
```

### 2.2 Stack Tecnológico

| Componente | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **LLM** | Gemma-3N Fine-tuneado | 6B params | Procesamiento de lenguaje multimodal |
| **Framework Web API** | FastAPI | 0.104+ | Servicio de inferencia del modelo |
| **Framework Web UI** | Streamlit | 1.28+ | Interfaz de usuario |
| **Bases de datos** | SQLite + SQLAlchemy | 3.10.1+ | Persistencia de datos |
| **Transcripción de audio** | Whisper (faster-whisper) | small | Convertir voz a texto |
| **GPU Inference** | PyTorch + CUDA | 2.0+ / 12.1+ | Aceleración de inferencia |
| **Orquestación** | Docker Compose | 3.8+ | Gestión de servicios |

---

## 3. Flujo de Ejecución Detallado

### 3.1 Escenario: Usuario pregunta voz + imagen

**Entrada**: Caficultor toma foto de hoja enferma, graba pregunta "¿Qué tiene esta hoja?"

**Paso 1: Streamlit captura inputs**
```python
# app/web.py (línea ~150)
audio_bytes = recorder_output["bytes"]  # Audio del usuario
image_bytes = uploaded_file.read()       # Imagen cargada
```

**Paso 2: Transcribir audio a texto**
```python
# app/web.py (línea ~170)
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu")  # CPU para no competir con GPU del modelo
segments, _ = model.transcribe(audio_bytes, language="es")
question = " ".join([s.text for s in segments])
# Result: "¿Qué enfermedad tiene esta hoja?"
```

**Paso 3: Enviar a Main API**
```python
# app/web.py (línea ~200)
response = requests.post(
    "http://api:8000/ask",
    data={"question": question},
    files={"image": image_bytes}
)
```

⚠️ **Nota importante**: Usa `http://api:8000`, NO `localhost:8000`
- En Docker Compose, los servicios se llaman por **nombre de servicio**
- "api" es el nombre definido en `docker-compose.yml`
- Docker Compose proporciona DNS interno para resolver ese nombre

**Paso 4: API recibe y procesa**
```python
# app/api.py (línea ~200)
@app.post("/ask")
async def ask(question: str, image: UploadFile = None):
    pil_image = None
    if image:
        pil_image = Image.open(io.BytesIO(await image.read()))
    
    # Armar messages para el modelo
    if image:
        system_prompt = SYSTEM_PROMPT_IMAGE
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
    else:
        system_prompt = SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
```

**Paso 5: Invocar modelo (Gemma-3N)**
```python
# app/api.py (línea ~250)
from transformers import pipeline

# El modelo ya está cargado en GPU desde startup
output = pipe(
    text=messages,
    images=pil_image,           # PIL Image, no bytes
    max_new_tokens=200,
    temperature=0.7
)

raw_response = output[0]["generated_text"]
```

**El modelo analiza**:
1. La imagen (features visuales de la hoja)
2. El texto de la pregunta (contexto lingüístico)
3. Los datos de entrenamiento (1000+ docs de CENICAFE sobre café)

**Respuesta posible 1 (conocimiento puro)**:
```
"La hoja presenta síntomas de Roya (Hemileia vastatrix). 
Los síntomas incluyen pústulas amarillas en el envés.
Recomendación: Aplicar fungicidas cúpricos o azufrados."
```

**Respuesta posible 2 (detecta que necesita datos en BD)**:
```json
{"tool": "inventario_consulta", "argumentos": "producto=fungicida cúprico"}
```

**Paso 6: Parsear salida (si es JSON)**
```python
# app/api.py (línea ~280)
tool_call = parse_tool_call(raw_response)
if tool_call:
    # Llamar al microservicio apropiadoif tool_call["tool"] == "inventario_consulta":
        response = requests.get(
            "http://inventario:8001/consulta",
            params=parse_args(tool_call["argumentos"])
        )
        inventory_data = response.json()
        # Generar respuesta final con los datos
```

**Paso 7: Devolver respuesta**
```python
# app/api.py (línea ~300)
return JSONResponse({
    "answer": "Tienes 5 kg de fungicida cúprico en bodega...",
    "has_image": True,
    "model": "Gemma-3N",
    "processing_time": 2.3
})
```

**Paso 8: Streamlit muestra resultado**
```python
# app/web.py (línea ~220)
st.write(response.json()["answer"])
```

---

## 4. Modelos y Fine-tuning

### 4.1 Gemma-3N: El Modelo Base

**Características**:
- **Origen**: Google
- **Tamaño**: 6 mil millones de parámetros
- **Peso en disco**: ~3.2 GB (comprimido a int4)
- **Capacidades**: Texto + imagen + audio
- **Velocidad de inferencia**: ~0.5-2 segundos por pregunta (GPU), ~5-10 seg (CPU)

**Comparación con otros modelos**:
```
GPT-4 (OpenAI):
  - Parámetros: 1.7 billones (estimado)
  - Costo: $0.003 por 1K tokens
  - Requiere: Internet, API key
  - Latencia: ~2-5 segundos

Llama 3 70B:
  - Parámetros: 70 mil millones
  - Costo: Hardware GPU A100 $3/hora
  - Requiere: GPU potente (A100)
  - Latencia: ~1-2 segundos

Gemma-3N (nuestro):
  - Parámetros: 6 mil millones
  - Costo: Una sola vez (no recurrente)
  - Requiere: GPU 8GB (RTX 4060) o Mac M-series
  - Latencia: ~1 segundo (GPU), ~5 seg (CPU)

DistilBERT (baseline):
  - Parámetros: 66 millones
  - Velocidad: Muy rápida
  - Limitación: Solo clasificación/embeddings, no generación
```

### 4.2 Fine-tuning: Especialización

**¿Por qué fine-tuneamos?**

El modelo base Gemma-3N es "médico general". Al entrenar con datos de café, se convierte en "cardiólogo del café".

**Datasets utilizados**:

1. **CENICAFE Documents** (1,000+ documentos)
   - Guías técnicas sobre cultivo
   - Monografías sobre plagas
   - Recomendaciones sobre fertilización
   - Todo de fuentes oficiales colombianas

2. **Image Dataset** (2,616 imágenes etiquetadas)
   - Hojas sanas vs enfermas
   - Tipos de plagas
   - Síntomas de enfermedades
   - Anotadas por expertos

3. **Function Calling Pairs** (2,700 ejemplos)
   - Preguntas que requieren BD
   - Preguntas que requieren solo conocimiento
   - Ejemplos de JSON output correcto

**Proceso de fine-tuning** (ver FINETUNING_GUIDE.md para detalles):
```
Gemma-3N base
    │
    ├→ Lora Adapters (parámetros adicionales, 0.1%)
    │
    ├→ Entrenar con datasets
    │  - Learning rate: 1e-4
    │  - Batch size: 4
    │  - Epochs: 3
    │
    └→ Gemma-3N Fine-tuned (especializado en café)
```

**Resultado**:
```python
# Accuracy en task de classification (prueba manual)
Base model:  "Roya" - clasificación correcta: 60%
Fine-tuned:  "Roya" - clasificación correcta: 92%

# Respuestas en español
Base model:  Cambia entre español e inglés
Fine-tuned:  Siempre en español, con terminología local
```

---

## 5. Microservicios y Orquestación

### 5.1 ¿Por qué microservicios?

**Alternativa 1: Monolito (todo en un programa)**
```python
# app.py - MONOLITO
@app.post("/ask")
def ask(question, image):
    # Procesar pregunta
    # Cargar modelo
    # Consultar inventario
    # Consultar gastos
    # Consultar cosecha
    # Consultar ingresos
    # Guardar historial
    # Retornar respuesta
```

**Problemas del monolito**:
- Si falla "guardar historial" → cae todo
- Si muchos usuarios consultan BD → lentitud total
- No puedes escalar solo la BD sin escalar la IA
- Actualizar un módulo requiere reiniciar TODO

**Alternativa 2: Microservicios (nuestro enfoque)**
```
┌────────────────────────────┐
│  Main API (modelo)         │
│  - Falla → solo IA muere   │
│  - Escalable por separado  │
└────────────────────────────┘
       │      │       │       │
       ↓      ↓       ↓       ↓
      BD1    BD2     BD3     BD4
     (solo lectura/escritura)
```

**Ventajas**:
- Aislamiento: Si BD2 falla, BD1/3/4 siguen
- Escalabilidad: Replica solo lo que crece
- Deployment: Actualiza servicios independientes
- Equipo: Varios equipos trabajan en paralelo

### 5.2 Los 5 Microservicios

#### **1. Main API (puerto 8000)**
```python
# app/api.py
# Responsabilidades:
# - Cargar y ejecutar Gemma-3N
# - Recibir preguntas + imágenes
# - Parsear JSON del modelo
# - Orquestar llamadas a otros servicios
# - Retornar respuestas

@app.post("/ask")
async def ask(question: str, image: UploadFile = None):
    ...
```

**Dependencias**: Python, transformers, torch, fastapi
**Memoria**: ~5GB (modelo + overhead)
**GPU**: Requiere GPU para performance

#### **2. Inventario API (puerto 8001)**
```python
# app/databases/inventario/main.py
# Responsabilidades:
# - CRUD de productos
# - Consultas de stock

@app.get("/consulta")
def consulta(producto: str):
    # SELECT * FROM inventario WHERE producto LIKE producto
    return {...}

@app.post("/crear")
def crear(producto: str, cantidad: int):
    # INSERT INTO inventario
    ...
```

**Base de datos**: SQLite `inventario.db`
**Schema**:
```python
class Producto(Base):
    __tablename__ = "inventario"
    id: int (PK)
    producto: str (fertilizante, fungicida, etc.)
    cantidad: float (kg)
    fecha_última_actualización: datetime
```

#### **3. Gastos API (puerto 8002)**
```python
# app/databases/gastos/main.py
# Responsabilidades:
# - CRUD de gastos mensuales
# - Reportes de gastos

@app.get("/mes")
def por_mes(año: int, mes: int):
    # SELECT SUM(monto) FROM gastos WHERE año=año AND mes=mes
    return {...}
```

**Schema**:
```python
class Gasto(Base):
    __tablename__ = "gastos"
    id: int
    año: int
    mes: int (1-12)
    categoria: str (semillas, agua, mano obra, etc.)
    monto: float (COP)
```

#### **4. Cosecha API (puerto 8003)**
```python
# app/databases/cosecha/main.py
# Responsabilidades:
# - Registro de cosechas
# - Kilos obtenidos, fechas, etc.
```

#### **5. Ingresos API (puerto 8004)**
```python
# app/databases/ingresos/main.py
# Responsabilidades:
# - Registro de ventas
# - Precios, cantidades
```

### 5.3 Comunicación Entre Servicios

**En Docker Compose**, los servicios se encuentran entre sí por **nombre DNS interno**:

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - INVENTORY_API_BASE_URL=http://inventario:8001
      - EXPENSES_API_BASE_URL=http://gastos:8002
```

**Internamente en api.py**:
```python
import os
import requests

INVENTORY_URL = os.getenv("INVENTORY_API_BASE_URL")

def consultar_inventario(producto: str):
    response = requests.get(
        f"{INVENTORY_URL}/consulta",
        params={"producto": producto}
    )
    return response.json()
```

**¿Por qué "http://inventario:8001" y no "http://localhost:8001"?**
- `localhost` dentro de un contenedor = el mismo contenedor
- En Docker Compose, hay un DNS interno que resuelve "inventario" → IP del contenedor inventario
- Es magia de Docker Compose (user-defined bridge network)

---

## 6. Cómo Ejecutar el Sistema

### 6.1 Opción A: Ejecución Local con run-local.sh

**¿Qué es un archivo `.sh`?**

`.sh` significa "Shell script" - es un archivo de texto que contiene comandos de terminal ejecutados en secuencia, similar a un batch file de Windows pero para Linux/Mac.

**¿Por qué `.sh` en vez de solo Python?**

Porque necesitas:
1. Crear virtual environment
2. Instalar dependencias
3. Detectar GPU/CPU
4. Iniciar 6 servicios en paralelo (pero secuencialmente con waits)
5. Limpiar al salir

Toda esa lógica es difícil en puro Python.

**Cómo funciona `run-local.sh`**:

```bash
#!/bin/bash  ← Indica que es un script de shell

# Variables de color
RED='\033[0;31m'
GREEN='\033[0;32m'

# Función cleanup que se ejecuta al salir (Ctrl+C)
cleanup() {
    pkill -P $$
    for port in 8000 8001 8002 8003 8004 8501; do
        lsof -ti:$port | xargs kill -9
    done
    exit 0
}

# Si presionas Ctrl+C, ejecuta cleanup
trap cleanup EXIT INT TERM

# Crear virtual environment si no existe
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activar virtual environment
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Detectar plataforma (macOS Intel, Apple Silicon, Linux)
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    pip install torch torchvision torchaudio
elif command -v nvidia-smi &> /dev/null; then
    # Linux con GPU NVIDIA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    # CPU puro
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Iniciar servicios uno por uno
cd databases/inventario
python -m uvicorn main:app --port 8001 > ../../logs/inventario.log 2>&1 &
sleep 2  ← Espera para que el servicio se estabilice

cd ../gastos
python -m uvicorn main:app --port 8002 > ../../logs/gastos.log 2>&1 &
sleep 2

# ... repetir para cosecha, ingresos ...

# Iniciar API principal (requiere que las BDs estén listas)
python -m uvicorn api:app --port 8000 > logs/api.log 2>&1 &
sleep 15  ← Espera más porque el modelo tarda en cargar

# Iniciar Streamlit (requiere que API esté lista)
streamlit run web.py --port 8501 > logs/streamlit.log 2>&1 &

# Esperar infinitamente (mientras no presiones Ctrl+C)
while true; do sleep 1; done
```

**Paso a paso para ejecutar**:

```bash
# 1. Descargar modelo (una sola vez)
python download.py
# Esto descarga ~3.2 GB a ./models/

# 2. Hacerlo ejecutable
chmod +x app/run-local.sh

# 3. Ejecutar
cd app
./run-local.sh

# Espera unos 30 segundos mientras se cargan los servicios
# Verás:
# ✓ Servicio Inventario está activo en puerto 8001
# ✓ Servicio Gastos está activo en puerto 8002
# ... etc

# 4. Abrir en navegador
open http://localhost:8501
# O en Linux: xdg-open http://localhost:8501

# 5. Para detener, presiona Ctrl+C en la terminal
```

**Troubleshooting de run-local.sh**:

```bash
# Error: "command not found: ./run-local.sh"
→ Olvidaste chmod +x o no estás en el directorio app/

# Error: "port 8000 already in use"
→ El script lo limpia, pero si falla:
lsof -ti:8000 | xargs kill -9

# Error: "GPU not detected"
→ Checa que pytorch esté instalado:
python -c "import torch; print(torch.cuda.is_available())"

# Error: "Model not found"
→ Ejecuta primero: python download.py
```

### 6.2 Opción B: Ejecución con Docker Compose

**¿Qué es Docker Compose?**

Docker Compose es un archivo YAML que describe múltiples contenedores Docker y cómo conectarlos. En vez de ejecutar 6 comandos `docker run`, ejecutas 1 comando `docker-compose up`.

**Ventaja sobre run-local.sh**:
- Ambiente aislado (no contamina tu sistema)
- Reproducible (funciona igual en cualquier máquina)
- Fácil de escalar
- Ideal para producción

**Paso a paso**:

```bash
# 1. Asegurar que Docker está instalado
docker --version
# Output: Docker version 24.0.x

# 2. Asegurar que docker-compose está instalado
docker-compose --version
# Output: Docker Compose version 2.x

# 3. Descargar modelo
python download.py
# Descarga ~3.2 GB a ./models/

# 4. Ir al directorio app
cd app

# 5. Revisar el docker-compose.yml
cat docker-compose.yml
```

**Estructura de docker-compose.yml**:

```yaml
version: '3.8'

services:
  # Base de datos PostgreSQL (opcional, ahora SQLite)
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: caficulbot
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - caficulbot-network

  # Servicio 1: Inventario
  inventario:
    build:
      context: .
      dockerfile: Dockerfile
    command: python -m uvicorn databases.inventario.main:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    volumes:
      - ./app:/app/app
      - ./app/databases:/app/databases
    networks:
      - caficulbot-network
    depends_on:
      - postgres

  # Servicio 2: Gastos
  gastos:
    build: .
    command: python -m uvicorn databases.gastos.main:app --host 0.0.0.0 --port 8002
    ports:
      - "8002:8002"
    volumes:
      - ./app:/app/app
      - ./app/databases:/app/databases
    networks:
      - caficulbot-network
    depends_on:
      - postgres

  # ... repetir para cosecha, ingresos ...

  # API Principal
  api:
    build: .
    command: python -m uvicorn api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
      - ./models:/app/models  ← Donde cargamos el modelo
    environment:
      INVENTORY_API_BASE_URL: http://inventario:8001
      EXPENSES_API_BASE_URL: http://gastos:8002
      PRODUCTION_API_BASE_URL: http://cosecha:8003
      INCOME_API_BASE_URL: http://ingresos:8004
    networks:
      - caficulbot-network
    depends_on:
      - inventario
      - gastos
      - cosecha
      - ingresos
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia  ← Asignar GPU
              count: 1
              capabilities: [gpu]

  # Interfaz Web
  web:
    build: .
    command: streamlit run web.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    volumes:
      - ./app:/app/app
    networks:
      - caficulbot-network
    depends_on:
      - api

networks:
  caficulbot-network:
    driver: bridge

volumes:
  postgres_data:
```

**Líneas clave explicadas**:

| Línea | Significa |
|-------|-----------|
| `build: .` | Usar Dockerfile en este directorio |
| `command: ...` | Qué comando ejecutar dentro del contenedor |
| `ports: - "8001:8001"` | Puerto host:puerto contenedor |
| `volumes: - ./app:/app/app` | Montar carpeta host en contenedor |
| `environment: INVENTORY_API_BASE_URL: http://inventario:8001` | Variable de entorno (nota: usa nombre de servicio "inventario", no localhost) |
| `depends_on: - inventario` | No iniciar este servicio hasta que "inventario" esté listo |
| `networks: - caficulbot-network` | Conectar a la red para que se vean entre sí |
| `deploy.resources.devices` | Reservar GPU NVIDIA |

**Ejecutar con Docker Compose**:

```bash
# Opción 1: Iniciar en foreground (ves logs en vivo)
docker-compose up

# Output esperado:
# inventario_1  | INFO:     Uvicorn running on http://0.0.0.0:8001
# gastos_1      | INFO:     Uvicorn running on http://0.0.0.0:8002
# api_1         | Loading model... (tarda ~15 seg)
# web_1         | Streamlit app running on http://0.0.0.0:8501

# Opción 2: Iniciar en background (detached mode)
docker-compose up -d

# Opción 3: Ver logs después de iniciar
docker-compose logs -f api
docker-compose logs -f web
```

**Detener servicios**:

```bash
# Opción 1: Si está en foreground
Ctrl+C

# Opción 2: Si está en background
docker-compose down

# Opción 3: Bajar y eliminar volúmenes
docker-compose down -v
```

**Ver estado**:

```bash
docker-compose ps
# Output:
# NAME                COMMAND                  SERVICE      STATUS
# app-inventario-1    python -m uvicorn ...    inventario   Up 2 minutes
# app-gastos-1        python -m uvicorn ...    gastos       Up 2 minutes
# app-api-1           python -m uvicorn ...    api          Up 1 minute 30 seconds
# app-web-1           streamlit run web.py     web          Up 1 minute
```

**Troubleshooting Docker Compose**:

```bash
# Error: "Cannot connect to Docker daemon"
→ Inicia Docker:
open /Applications/Docker.app  # macOS
# o en Linux:
sudo systemctl start docker

# Error: "port 8001 already in use"
→ Otro contenedor o proceso está usando ese puerto:
docker-compose down  # Detiene todos los contenedores
docker ps -a        # Lista contenedores
docker rm container_name  # Elimina si no está en uso

# Error: "No space left on device"
→ Docker Compose acumula imágenes/volúmenes:
docker system prune -a  # Limpia TODO (¡cuidado!)

# Ver logs en detalle
docker-compose logs -f --tail=100 api
```

---

## 7. Casos de Uso

### 7.1 Consulta Técnica (Solo Conocimiento)

**Usuario**: "¿Cuál es el ciclo de vida de la Broca del Café?"

**Flujo**:
1. Pregunta → Streamlit
2. Streamlit → API (sin imagen)
3. API → Gemma-3N
4. Gemma-3N (tiene entrenamiento de 1000+ docs CENICAFE) → respuesta
5. Respuesta → Usuario

**Tiempo**: ~1 segundo (GPU)
**Nota**: Sin necesidad de BD

**Respuesta esperada**:
```
El ciclo de vida de la Broca del Café (Hypothenemus hampei):

1. Adulto penetra el fruto: ~3 días
2. Deposita huevos en la almendra: ~5-10 huevos
3. Incubación de huevos: 8-10 días
4. Larva (4 estadíos): 15-20 días
5. Pupa: 8-12 días
6. Adulto emergente: vive ~7-8 meses

Ciclo completo: 35-50 días en condiciones óptimas (25-28°C).

Manejo:
- Recolección de fruto caído
- Poda sanitaria
- Trampas con cafeína
```

### 7.2 Consulta Operativa (Conocimiento + BD)

**Usuario**: "¿Cuánto fertilizante nitrogenado me queda?" + imagen de etiqueta

**Flujo**:
1. Pregunta + imagen → Streamlit
2. Streamlit → API
3. API → Gemma-3N
4. Gemma-3N reconoce tipo de fertilizante en imagen → **detecta que necesita BD**
5. Gemma-3N → JSON: `{"tool": "inventario_consulta", "argumentos": "producto=fertilizante nitrogenado"}`
6. API parsea JSON → llama a inventario:8001
7. Inventario → BD → respuesta: "tenemos 50 kg"
8. API → genera respuesta en lenguaje natural
9. Respuesta → Usuario

**Tiempo**: ~2 segundos
**Nota**: Requiere BD

**Respuesta esperada**:
```
Según la imagen, es fertilizante nitrogenado (urea 46-0-0).

En tu inventario tienes:
- 50 kg de urea (adquirida hace 2 semanas)
- Costo: $150,000 COP

Recomendación: Para una hectárea en floración, necesitas ~180 kg.
Te faltan 130 kg.
```

### 7.3 Diagnóstico Visual

**Usuario**: Carga foto de hoja con manchas marrones + "¿Qué es esto?"

**Flujo**:
1. Foto + pregunta → Streamlit
2. Streamlit → API (multimodal)
3. API → Gemma-3N (con imagen como input)
4. Gemma-3N analiza imagen + entrenamiento con 2,616 imágenes etiquetadas
5. Respuesta (diagnóstico + recomendaciones)

**Tiempo**: ~2 segundos
**Nota**: Análisis visual, sin BD

**Respuesta esperada**:
```
Diagnóstico: Antracnosis (Colletotrichum spp.)

Evidencia visual:
- Manchas circulares concéntricas
- Centro marrón oscuro
- Bordes más claros
- Típico de ambiente húmedo

Control:
1. Remover hojas afectadas
2. Mejorar ventilación
3. Fungicida: Cobre o Mancozeb
4. Evitar mojado foliar después del atardecer
```

---

## 8. Rendimiento y Limitaciones

### 8.1 Benchmarks

**Hardware**: RTX 4060 8GB VRAM

| Escenario | Latencia | Memoria GPU | Notas |
|-----------|----------|-------------|-------|
| Pregunta texto | 0.8 seg | 6.2 GB | Incluye tokenización |
| Pregunta + imagen | 1.3 seg | 7.1 GB | Análisis visual |
| Pregunta + imagen + BD | 1.8 seg | 7.1 GB | +0.5 seg por latencia de red |
| 10 preguntas concurrentes | ~1.8 seg avg | OOM | No soporta concurrencia (diseño) |

### 8.2 Limitaciones Conocidas

1. **No soporta concurrencia**: Si dos usuarios hacen preguntas al mismo tiempo, la segunda espera
   - Solución: Usar GPU con más memoria o implementar cola de procesamiento

2. **Requiere GPU para performance**: En CPU es ~10x más lento
   - Solución: Usar CPU es viable pero requiere paciencia

3. **Modelo offline**: No se actualiza con nuevos datos sin reentrenamiento
   - Solución: Fine-tuning periódico con nuevos datos

4. **Requiere 3.2 GB en disco**: No viable en dispositivos muy antiguos
   - Solución: Usar modelo más pequeño (Gemma-2B si es necesario)

---

## 9. Seguridad

### 9.1 Consideraciones

**Datos sensibles**:
- Registros de inventario (qué productos hay)
- Registros de gastos (cuánto gasta el agricultor)
- Imágenes de finca

**Enfoque de seguridad**:
- **Offline-first**: Los datos nunca salen de la finca
- **Autenticación**: Streamlit puede tener usuario/contraseña (opcional)
- **Base de datos**: SQLite (no es multiusuario, pero es simple)
- **HTTPS**: Usar en producción si se expone a red

**Mejoras futuras**:
- Agregar autenticación (Flask-Login o similar)
- Encriptar BD SQLite
- Usar PostgreSQL en lugar de SQLite para multi-usuario
- Audit logs de quién accedió qué

---

## 10. Próximos Pasos

### 10.1 Mejoras Corto Plazo
- Agregar más imágenes al dataset (más precisión en diagnósticos)
- Implementar caché de respuestas (reutilizar si pregunta es repetida)
- Agregar persistencia de chat (historias de conversación)

### 10.2 Mejoras Mediano Plazo
- Autenticación y multi-usuario
- Sincronización con nube (backup, actualizaciones)
- App móvil (Flutter o React Native)

### 10.3 Escala a Producción
- SageMaker para servir el modelo (escalable, monitoreado)
- RDS (PostgreSQL) en lugar de SQLite
- ElastiCache para sesiones de usuario
- CloudFront para CDN (UI)

---

## Conclusión

Caficulbot es un ejemplo práctico de **MLOps moderno**: 
- Arquitectura de microservicios
- Fine-tuning especializado
- Ejecución offline
- Stack open source

Es escalable desde un laptop del agricultor a un servidor en la nube, sin cambiar el código fundamental.

