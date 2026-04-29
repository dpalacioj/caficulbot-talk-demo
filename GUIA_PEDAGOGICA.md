# ☕ Guía Pedagógica: Del "Hola Docker" a un Sistema MLOps Real

Bienvenido/a. Si ya hiciste la práctica de la **app de gatitos** (`04-Deployment/deploy/intro-dockers`), aquí damos el siguiente salto: entender cómo se ve un sistema MLOps **de verdad** en producción, usando Caficulbot como caso real.

> 💡 **Importante**: Esta guía NO busca que ejecutes todo el sistema (requiere GPU y 3GB de modelo). El objetivo es **entender la arquitectura, los conceptos y cómo se conecta todo**. Al final verás cómo esto se traduce a herramientas reales como SageMaker y Databricks.

---

## 🎯 Objetivos de Aprendizaje

Al terminar esta guía podrás:

1. ✅ Entender qué es un **microservicio** y por qué existen
2. ✅ Entender qué es **Docker Compose** y cuándo usarlo (no solo Docker)
3. ✅ Leer un `docker-compose.yml` línea por línea
4. ✅ Entender cómo **FastAPI** se usa para servir modelos de ML
5. ✅ Entender qué es **fine-tuning** y por qué se hace
6. ✅ Entender cómo diferentes frameworks se conectan en un proyecto real
7. ✅ Saber cómo esto se ve en **producción real** (SageMaker, Databricks)

---

## 📐 De la App de Gatitos al Caficulbot: El Salto

Antes de meternos al código, compara ambos proyectos:

| Aspecto | App de Gatitos 🐱 | Caficulbot ☕ |
|---------|-------------------|----------------|
| **Propósito** | Enseñar Docker básico | Sistema MLOps real |
| **Servicios** | 1 (FastAPI) | 6 (API + 4 BDs + UI) |
| **Modelo ML** | Ninguno | Gemma-3N fine-tuneado (6B parámetros) |
| **Orquestación** | `docker run` | `docker-compose up` |
| **Base de datos** | No tiene | PostgreSQL + SQLAlchemy |
| **Frontend** | HTML estático | Streamlit interactivo |
| **Tamaño imagen** | ~150 MB | ~5-8 GB (con CUDA + modelo) |
| **Hardware** | Cualquier PC | Idealmente GPU |
| **Tipo de despliegue** | Demo | Producción offline |

**La analogía**: Los gatitos son como aprender a andar en bicicleta en un parque. El caficulbot es como manejar en una autopista con semáforos, intercambios y muchos carros coordinados.

---

# PARTE 1: Los Conceptos Clave

## 🧩 ¿Qué es un Microservicio? (Analogía: El Restaurante)

Imagina dos formas de organizar un restaurante:

### **Opción A: Monolito (todo junto)**
Un solo chef hace TODO: recibe al cliente, toma la orden, cocina, calcula la cuenta, limpia. Si el chef se enferma, el restaurante cierra.

### **Opción B: Microservicios (cada uno lo suyo)**
- **Mesero** → toma órdenes
- **Cocinero de entradas** → solo prepara entradas
- **Cocinero de platos fuertes** → solo platos fuertes
- **Cajero** → solo cobra

Si el cocinero de entradas se enferma, **el resto sigue funcionando**. Además, si hay muchos clientes pidiendo entradas, contratas **dos cocineros de entradas** (escalas solo esa parte).

**Eso es un microservicio**: cada servicio hace UNA cosa bien, y se comunican entre sí vía HTTP.

### En Caficulbot

```
┌────────────────────┐
│  Streamlit (UI)    │ ← "El mesero"
└─────────┬──────────┘
          │ HTTP
          ↓
┌────────────────────┐
│   Main API         │ ← "El chef principal" (usa el modelo de IA)
│   (Gemma-3N)       │
└─────────┬──────────┘
          │ HTTP
    ┌─────┼─────┬─────┬─────┐
    ↓     ↓     ↓     ↓
┌──────┐┌──────┐┌──────┐┌──────┐
│Invent││Gastos││Cosech││Ingres│  ← "Especialistas por tema"
│8001  ││8002  ││8003  ││8004  │
└──────┘└──────┘└──────┘└──────┘
```

Cada cajita es **un contenedor Docker independiente**. Si el servicio de "Gastos" falla, los demás siguen funcionando.

---

## 🐳 ¿Qué es Docker Compose? (Analogía: El Director de Orquesta)

**Docker** → te permite correr **UN contenedor** con `docker run`.

**Docker Compose** → te permite correr **MUCHOS contenedores coordinados** con un solo comando.

### Analogía musical

- Un músico tocando solo = `docker run` (un instrumento)
- Una orquesta sinfónica con 40 músicos = `docker-compose up`

El **director de orquesta** (Docker Compose) se encarga de:
- 🎯 Decir quién empieza a tocar primero (dependencias)
- 🎯 Coordinar que todos toquen la misma pieza (red compartida)
- 🎯 Saber cuándo alguien debe parar (shutdown)
- 🎯 Dar las partituras a cada músico (variables de entorno)

### ¿Cuándo usar cada uno?

| Situación | Usar |
|-----------|------|
| Una sola app (como gatitos) | `docker run` |
| App + base de datos | `docker-compose` |
| App + BD + cache + cola + UI | `docker-compose` (obligatorio) |
| Muchas réplicas, alta disponibilidad, autoscaling | **Kubernetes** (siguiente nivel) |

---

## ⚡ ¿Qué es FastAPI? (y por qué aquí)

FastAPI es un framework de Python que convierte funciones en **endpoints HTTP**.

### Ejemplo mínimo

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/saludar")
def saludar(nombre: str):
    return {"mensaje": f"Hola {nombre}"}
```

Si ejecutas `uvicorn main:app --port 8000`, puedes visitar:
```
http://localhost:8000/saludar?nombre=Juan
→ {"mensaje": "Hola Juan"}
```

### Por qué FastAPI para servir modelos de ML

1. **Rápido** (asíncrono, uno de los frameworks Python más veloces)
2. **Validación automática** de inputs (tipos, rangos)
3. **Documentación automática** (`/docs` te da un Swagger UI gratis)
4. **Fácil de dockerizar** (como ya viste con gatitos)

**En Caficulbot, CADA uno de los 5 microservicios es un FastAPI**. Sí, cinco FastAPIs corriendo al mismo tiempo.

---

# PARTE 2: El Modelo de IA (Gemma-3N)

## 🧠 ¿Qué es Gemma-3N?

**Gemma-3N** es un modelo de lenguaje multimodal (texto + imagen + audio) creado por Google. Tiene **6 mil millones de parámetros** (pesa ~3GB comprimido).

**Comparación de tamaños**:

| Modelo | Parámetros | Hardware mínimo |
|--------|-----------|-----------------|
| GPT-4 | ~1.7 billones (estimado) | Data center |
| Llama 3 70B | 70 mil millones | GPU A100 |
| **Gemma-3N** | **6 mil millones** | **GPU 8GB o Mac M-series** |
| DistilBERT | 66 millones | Laptop |

La gracia de Gemma-3N es que es **lo suficientemente pequeño para correr en un laptop de un caficultor**, pero lo suficientemente potente para ser útil.

## 🎨 ¿Qué es Fine-Tuning?

**Analogía**: Un médico general vs. un cardiólogo.

- **Médico general** = modelo base (sabe de todo un poco)
- **Cardiólogo** = modelo fine-tuneado (especialista en un tema)

El fine-tuning toma un modelo base (Gemma-3N) y lo **especializa** en un dominio con datos específicos. En este caso:

- 1,000+ documentos técnicos de **CENICAFE** (centro nacional del café)
- 2,616 imágenes etiquetadas de **plagas y enfermedades del café**
- 2,700 ejemplos de **function calling** (cuándo consultar la BD)

**Resultado**: Un modelo que sabe de café colombiano mucho mejor que el modelo base.

## 🔧 ¿Qué es "Function Calling"?

Imagina que le preguntas al modelo:

**Pregunta 1**: "¿Cómo se controla la roya?"
→ El modelo **sabe la respuesta** (fue entrenado con docs de CENICAFE).

**Pregunta 2**: "¿Cuánto fertilizante tenemos en bodega?"
→ El modelo **NO puede saberlo** (es dato en tiempo real, está en la BD).

**Solución**: Entrenar al modelo para que detecte la diferencia y **devuelva un JSON** cuando necesita consultar la BD:

```json
{"tool": "inventario_consulta", "argumentos": "producto=fertilizante"}
```

El código de `api.py` detecta ese JSON, llama al microservicio de Inventario (puerto 8001), obtiene el dato, y lo devuelve en lenguaje natural.

**Eso es function calling**: el modelo "llama" funciones externas cuando las necesita.

---

# PARTE 3: Arquitectura Completa

## 🏗️ Diagrama Detallado

```
                    USUARIO
                       │
                       ↓ http://localhost:8501
           ┌───────────────────────┐
           │   Streamlit (web.py)  │  ← Frontend (chat)
           │   Contenedor: web     │
           └───────────┬───────────┘
                       │ HTTP POST /ask
                       ↓ http://api:8000
           ┌───────────────────────┐
           │  FastAPI (api.py)     │  ← Carga el modelo Gemma-3N
           │  Contenedor: api      │     en memoria al arrancar
           │  + GPU reservada      │
           └───────────┬───────────┘
                       │
                       │ Si detecta function calling:
                       ↓
       ┌───────────────┼───────────────┬────────────────┐
       ↓               ↓               ↓                ↓
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ inventario │  │  gastos    │  │  cosecha   │  │  ingresos  │
│ :8001      │  │  :8002     │  │  :8003     │  │  :8004     │
│ FastAPI    │  │  FastAPI   │  │  FastAPI   │  │  FastAPI   │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │                │
      └───────────────┴───────┬───────┴────────────────┘
                              ↓
                    ┌──────────────────┐
                    │   PostgreSQL     │  ← Base de datos compartida
                    │   :5432          │
                    └──────────────────┘
```

## 🧱 Las 3 Capas

| Capa | Tecnología | Función |
|------|-----------|---------|
| **Presentación** | Streamlit | Chat web, subir imágenes, grabar voz |
| **Lógica** | FastAPI + Gemma-3N | Procesar preguntas, llamar al modelo, orquestar |
| **Datos** | PostgreSQL + SQLAlchemy | Guardar inventario, gastos, cosechas, ingresos |

Esto es el patrón **clásico de 3 capas** que verás en el 90% de las apps web y MLOps.

---

# PARTE 4: Cómo se Conectan los Frameworks

Esta es la parte "aha!" — dónde cada framework entra en juego.

## 🔗 Flujo Completo de una Pregunta con Voz + Imagen

Imagina que un caficultor:
1. Toma una foto de una hoja enferma
2. Graba con su voz: "¿Qué enfermedad tiene?"
3. Presiona enviar

Esto es lo que pasa **paso a paso**:

### **Paso 1: Streamlit (web.py)**
```python
# El usuario sube foto y graba audio en la UI
audio_bytes = recorder_output["bytes"]
image_bytes = uploaded_file.read()
```

### **Paso 2: Whisper transcribe el audio (local, en CPU)**
```python
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu")
segments, _ = model.transcribe(audio_file, language="es")
question = " ".join([s.text for s in segments])
# question = "¿Qué enfermedad tiene?"
```

**Nota**: Whisper corre **en el mismo contenedor que Streamlit**, no es otro microservicio.

### **Paso 3: Streamlit envía a la Main API**
```python
response = requests.post(
    "http://api:8000/ask",
    data={"question": question},
    files={"image": image_bytes}
)
```

⚠️ **Fíjate**: usa `http://api:8000`, NO `http://localhost:8000`. En Docker Compose, **los servicios se llaman entre sí por su nombre de servicio** (definido en `docker-compose.yml`).

### **Paso 4: FastAPI recibe, invoca el modelo Gemma-3N**
```python
pil_image = Image.open(io.BytesIO(image_bytes))
messages = [
    {"role": "system", "content": SYSTEM_PROMPT_IMAGE},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
]
output = pipe(text=messages, images=pil_image, max_new_tokens=200)
```

Aquí el modelo (que ya está cargado en GPU) analiza la imagen y genera texto.

### **Paso 5: FastAPI devuelve la respuesta**
```python
return JSONResponse({
    "answer": "La hoja presenta signos de roya. Se recomienda...",
    "has_image": True
})
```

### **Paso 6: Streamlit muestra la respuesta**
```python
st.write(response.json()["answer"])
```

### Diagrama del flujo

```
[Usuario]
   │ voz + imagen
   ↓
[Streamlit] ──(Whisper)──→ texto
   │
   │ POST /ask (texto + imagen)
   ↓
[Main API - FastAPI]
   │
   │ pipe(text=messages, images=pil_image)
   ↓
[Gemma-3N en GPU]
   │
   │ genera respuesta
   ↓
[Streamlit]
   │
   ↓
[Usuario ve respuesta]
```

---

# PARTE 5: Leyendo el `docker-compose.yml` Línea por Línea

Abre el archivo `app/docker-compose.yml` y léelo con esta guía.

## Anatomía general

```yaml
version: '3.8'                 # ← Versión de la sintaxis de compose

services:                      # ← Aquí listas todos tus contenedores
  postgres: ...
  inventario: ...
  gastos: ...
  cosecha: ...
  ingresos: ...
  api: ...
  web: ...

networks:                      # ← Red virtual para que se vean entre sí
  caficulbot-network:
    driver: bridge

volumes:                       # ← Almacenamiento persistente
  postgres_data:
```

## Un servicio por dentro (ejemplo: `api`)

```yaml
api:
  build: .                                    # Construye usando el Dockerfile local
  command: python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
  volumes:
    - ./app:/app/app                          # Código en vivo (desarrollo)
    - ./models:/app/models                    # Modelo de ML
  ports:
    - "8000:8000"                             # Puerto host:contenedor
  environment:
    - INVENTORY_API_BASE_URL=http://inventario:8001
    - EXPENSES_API_BASE_URL=http://gastos:8002
  depends_on:                                 # NO arranques antes que estos
    - inventario
    - gastos
    - cosecha
    - ingresos
  networks:
    - caficulbot-network
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia                    # Reservar GPU
            count: 1
            capabilities: [gpu]
```

### Conceptos clave por línea

| Directiva | Qué hace | Analogía |
|-----------|----------|----------|
| `build: .` | Construye imagen desde Dockerfile local | "Arma la pieza aquí" |
| `command` | Comando que corre dentro del contenedor | "Qué hace el músico" |
| `volumes` | Monta carpeta del host en el contenedor | "Cajón compartido" |
| `ports` | Mapea puerto host→contenedor | "Puerta de entrada" |
| `environment` | Variables de configuración | "Instrucciones en papel" |
| `depends_on` | Orden de arranque | "Prerrequisitos" |
| `networks` | Red compartida | "Estar en la misma sala" |

## 🔑 El Detalle Crítico: Nombres de Servicios como DNS

En el `docker-compose.yml`, dentro de la red `caficulbot-network`, cada servicio es accesible **por su nombre**:

```yaml
environment:
  - INVENTORY_API_BASE_URL=http://inventario:8001   # ← "inventario" es el nombre del servicio
  - DATABASE_URL=postgresql://...@postgres:5432/... # ← "postgres" es el nombre del servicio
```

**Esto NO es magia**: Docker Compose crea un DNS interno donde cada nombre de servicio se resuelve a la IP de su contenedor. Por eso `http://localhost:8001` **NO funciona** dentro del contenedor (ese `localhost` sería el contenedor mismo, no el host).

---

# PARTE 6: ¿Cómo se Ve Esto en Producción Real?

Esta sección responde a la pregunta que siempre sale en clase: **"¿y esto cómo se hace de verdad en una empresa?"**

## 🏢 Tres Escenarios Reales

### **Escenario A: Caficultor Rural (como está diseñado)**
- **Hardware**: Laptop con GPU o Mac M-series
- **Conectividad**: OFFLINE (no hay internet en la finca)
- **Deploy**: Docker Compose en el laptop
- **Actualización**: Técnico visita con USB con nueva versión
- **Monitoreo**: Logs locales, sin alertas externas

Este es el caso del **caficulbot real**. Está optimizado para offline.

### **Escenario B: Empresa Cafetera Mediana (ej: Federación de Cafeteros)**
- **Hardware**: Servidor en la oficina central + laptops clientes
- **Conectividad**: Online
- **Deploy**: Docker Compose en servidor, Streamlit web accesible por VPN
- **Actualización**: CI/CD con GitHub Actions
- **Monitoreo**: Prometheus + Grafana

### **Escenario C: Empresa Grande en la Nube (AWS/Azure/GCP)**
Aquí Docker Compose **ya no alcanza**. Entran:

| Componente del Caficulbot | Equivalente en AWS | Equivalente en Azure | Equivalente en Databricks |
|---------------------------|--------------------|-----------------------|---------------------------|
| Modelo Gemma-3N | **SageMaker Endpoint** | Azure ML Endpoint | **Mosaic AI Serving** |
| Main API (FastAPI) | ECS / Lambda | Container Apps | Serving Job |
| PostgreSQL | RDS | Azure SQL | Delta Lake |
| Streamlit UI | Amplify / ECS | Static Web Apps | Databricks Apps |
| Docker Compose | **ECS Task Definition** o **EKS (Kubernetes)** | AKS | Databricks Workflows |
| Almacenamiento modelo | S3 | Blob Storage | Unity Catalog Volumes |

### Ejemplo: Si migraras Caficulbot a SageMaker

```
┌─────────────────────────┐
│   CloudFront (CDN)      │ ← Distribuye UI al mundo
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│   ECS (FastAPI API)     │ ← Tu código, pero en la nube
└───────────┬─────────────┘
            │
    ┌───────┼───────┐
    ↓       ↓       ↓
┌───────┐ ┌────┐ ┌──────────┐
│ Sage- │ │RDS │ │ Inferen- │
│ Maker │ │ PG │ │ tia (GPU)│
│ Endp. │ └────┘ └──────────┘
└───────┘
    │
    │ Modelo Gemma-3N
    ↓
┌───────┐
│  S3   │
└───────┘
```

**Ventaja**: Autoscaling automático, alta disponibilidad, monitoreo integrado.
**Desventaja**: Costo. Un endpoint GPU de SageMaker cuesta ~$1-3 USD/hora.

## 🔔 ¿Y las Alertas? (Pregunta frecuente)

En producción real agregas:

| Necesidad | Herramienta típica |
|-----------|--------------------|
| ¿El modelo está respondiendo? | Health checks + CloudWatch / Datadog |
| ¿Respuestas del modelo son buenas? | **Evaluación LLM** (MLflow, LangSmith) |
| ¿El modelo hace drift? | **Model monitoring** (Evidently, Fiddler) |
| ¿Alguien está intentando romper el modelo? | **Guardrails** (Llama Guard, Azure Content Safety) |
| Alerta a Slack cuando algo falla | PagerDuty / Opsgenie / Slack webhook |

Estas son las piezas que Caficulbot NO tiene (porque es offline), pero que en AWS/Databricks se integran.

## 📊 Producción en Databricks (Alternativa Popular)

Si tu empresa usa Databricks (común en Colombia en bancos y retail):

1. **Modelo**: se guarda en **Unity Catalog** (registro de modelos)
2. **Serving**: **Mosaic AI Model Serving** expone un endpoint REST
3. **Pipeline de datos**: **Delta Live Tables** procesa datos del cultivo
4. **UI**: **Databricks Apps** (equivalente a Streamlit pero gestionado)
5. **Monitoreo**: **Lakehouse Monitoring** detecta drift automáticamente
6. **Fine-tuning**: **Mosaic AI Training** (ex MosaicML)

Ventaja: todo integrado, con governance y linaje de datos.

---

# PARTE 7: 🎓 Ejercicios Prácticos

> No necesitas ejecutar todo el sistema. Estos ejercicios son de **lectura + análisis**.

## Ejercicio 1: Cazar el Flujo

Abre `app/api.py` y responde:

1. ¿En qué línea se carga el modelo? → Pista: busca `@app.on_event("startup")`
2. ¿En qué línea se detecta si hay GPU (CUDA/MPS/CPU)?
3. ¿Qué endpoint procesa las preguntas? → Busca `@app.post(...)`
4. ¿Qué función parsea el JSON cuando el modelo quiere llamar una herramienta?

## Ejercicio 2: Mapear Servicios a Puertos

Sin mirar el README, usando solo `docker-compose.yml`, llena esta tabla:

| Servicio | Puerto | Para qué sirve |
|----------|--------|----------------|
| postgres | ? | ? |
| inventario | ? | ? |
| gastos | ? | ? |
| cosecha | ? | ? |
| ingresos | ? | ? |
| api | ? | ? |
| web | ? | ? |

## Ejercicio 3: Dibujar el Flujo de una Pregunta

Dibuja en papel (o con un diagrama online) lo que pasa cuando el usuario pregunta:

**"¿Cuánto fertilizante tenemos?"**

Pistas:
- ¿Por qué servicios pasa?
- ¿Cuándo entra el modelo? ¿Cuándo entra la BD?
- ¿Qué JSON devuelve el modelo?

## Ejercicio 4: Comparar con los Gatitos

Haz una tabla de dos columnas que compare qué hay en la app de gatitos vs. Caficulbot:
- ¿Qué conceptos son los mismos?
- ¿Qué conceptos son nuevos en Caficulbot?

## Ejercicio 5: Diseño en la Nube

Si tu jefe te pidiera "lleven Caficulbot a AWS para 10,000 caficultores":

1. ¿Qué servicios de AWS usarías para cada componente?
2. ¿Qué NO funcionaría igual (ej: offline)?
3. ¿Cuánto costaría aproximadamente al mes? (Pista: investiga precio de SageMaker GPU endpoint)

---

# PARTE 8: 🐛 Dudas Frecuentes de Estudiantes

### "¿Necesito saber todo este código para entender MLOps?"
**No.** Necesitas entender **la arquitectura** (qué se comunica con qué). El código es secundario. Un ingeniero MLOps diseña sistemas como este, no necesariamente los programa desde cero.

### "¿Por qué no ponen todo en un solo contenedor?"
Funcionaría... para una demo. En producción:
- No puedes escalar una parte (ej: solo el modelo) sin escalar todo
- Si falla algo, cae todo
- Varios equipos no pueden trabajar en paralelo
- Actualizar una parte requiere reiniciar todo

### "¿Esto es lo mismo que una API REST?"
Sí, **cada microservicio ES una API REST**. La diferencia es que en vez de tener UNA API REST grande, tienes VARIAS APIs REST pequeñas que se hablan entre sí.

### "¿Docker Compose sirve para producción?"
**Para producción pequeña, sí.** Para producción grande (miles de usuarios), se usa **Kubernetes** (k8s). Kubernetes es "Docker Compose con esteroides": autoscaling, auto-healing, rolling updates, etc.

### "¿Por qué Streamlit y no React?"
**Streamlit** se programa en Python puro (fácil para data scientists). **React** es JavaScript (requiere otro skillset). Para prototipos y demos internas de ML, Streamlit es el estándar.

### "¿El modelo se entrena cada vez que arranca el contenedor?"
**No.** Se **carga** (el archivo ya entrenado se lee a memoria/GPU). Entrenar el modelo se hizo UNA vez en los notebooks de `fine_tuning/`. Eso tarda horas y requiere GPU potente.

---

# PARTE 9: ✅ Checklist de Aprendizaje

Márcalo mentalmente:

- [ ] Entiendo qué es un microservicio y por qué existen
- [ ] Entiendo la diferencia entre `docker run` y `docker-compose up`
- [ ] Puedo leer un `docker-compose.yml` básico
- [ ] Sé qué es `depends_on` y para qué sirve
- [ ] Entiendo por qué los servicios se llaman por nombre (no `localhost`)
- [ ] Sé qué es FastAPI y por qué se usa para servir modelos
- [ ] Entiendo qué es fine-tuning y por qué se hace
- [ ] Entiendo qué es function calling
- [ ] Puedo dibujar el flujo de una pregunta en el sistema
- [ ] Sé cómo esto se traduciría a SageMaker/Databricks
- [ ] Sé qué piezas faltan para "producción real" (alertas, monitoreo, drift)

---

# PARTE 10: 📚 Recursos para Profundizar

## Docker y Compose
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes (siguiente nivel)](https://kubernetes.io/docs/tutorials/)

## FastAPI
- [FastAPI oficial](https://fastapi.tiangolo.com/)
- [Serving ML models with FastAPI (tutorial)](https://fastapi.tiangolo.com/advanced/)

## MLOps en la nube
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [Databricks MLOps](https://www.databricks.com/product/machine-learning)
- [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/)

## LLMs y Fine-tuning
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Unsloth (la librería usada para fine-tune Gemma)](https://github.com/unslothai/unsloth)
- [Gemma-3N model card](https://huggingface.co/google/gemma-3n-E2B-it)

## Monitoreo de modelos
- [Evidently AI](https://www.evidentlyai.com/)
- [MLflow](https://mlflow.org/)

---

# 🎉 Cierre

Has hecho el salto de "correr un contenedor" a "entender un sistema MLOps real". Lo importante no es que memorices cada línea de código, sino que entiendas:

1. ✅ **Por qué existen los microservicios** (escalabilidad, aislamiento)
2. ✅ **Qué resuelve Docker Compose** (coordinar múltiples servicios)
3. ✅ **Cómo FastAPI conecta un modelo con el mundo**
4. ✅ **Qué es fine-tuning y cuándo tiene sentido**
5. ✅ **Cómo se ve esto en la nube real** (SageMaker, Databricks)

> 💡 **Recuerda**: Docker, FastAPI y Gemma-3N son **piezas de LEGO**. Puedes combinarlas de mil formas. Caficulbot es una forma; SageMaker Endpoint es otra; Databricks Serving es otra. Lo que aprendiste aquí te sirve para TODAS.

---

**Próximos pasos sugeridos**:
1. Volver a `04-Deployment/deploy/intro-dockers` y notar qué tan simple se ve ahora
2. Intentar un ejercicio de medio nivel: agregar una BD Redis a los gatitos con Docker Compose
3. Leer sobre Kubernetes como "el siguiente paso" cuando el sistema crece
