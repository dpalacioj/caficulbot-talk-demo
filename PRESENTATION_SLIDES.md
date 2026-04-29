# 🎤 Diapositivas de Presentación: Caficulbot

## Slide 1: Portada

```
╔════════════════════════════════════════════════════════╗
║                                                        ║
║   ☕ CAFICULBOT                                       ║
║   Asistente de IA Multimodal Offline                 ║
║   para Caficultores Colombianos                       ║
║                                                        ║
║   Presentación Técnica                                ║
║   Arquitectura, Fine-tuning y MLOps                  ║
║                                                        ║
║   Abril 2026                                          ║
║                                                        ║
╚════════════════════════════════════════════════════════╝

Presentador: [Tu nombre]
Email: david.palacio@unosquare.com
Repositorio: github.com/caficulbot
```

---

## Slide 2: ¿El Problema?

```
┌─────────────────────────────────────────────────────────┐
│  DESAFÍOS DE CAFICULTORES COLOMBIANOS               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ❌ Diagnosticar plagas y enfermedades                 │
│     "¿Qué tiene esta hoja?"                           │
│                                                         │
│  ❌ Información en tiempo real sin internet            │
│     Muchas fincas = OFFLINE                           │
│                                                         │
│  ❌ Idioma y contexto local                            │
│     Respuestas en español, de café colombiano         │
│                                                         │
│  ❌ Costo de asesoramiento                             │
│     Agronomistas son caros                            │
│                                                         │
│  ❌ Acceso a libros y manuales                         │
│     No son dinámicos, se desactualizan               │
│                                                         │
└─────────────────────────────────────────────────────────┘

Necesidad: IA especializada, offline, en español
```

---

## Slide 3: La Solución

```
┌─────────────────────────────────────────────────────────┐
│  CAFICULBOT: ASISTENTE IA OFFLINE                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ Diagnostica plagas y enfermedades                  │
│     Análisis de imágenes de hojas                     │
│                                                         │
│  ✅ Funciona sin internet                              │
│     Todo corre en la máquina local (laptop)           │
│                                                         │
│  ✅ Especializado en café colombiano                   │
│     Fine-tuneado con datos de CENICAFE               │
│                                                         │
│  ✅ Multimodal                                         │
│     Texto, voz (grabación), imágenes                 │
│                                                         │
│  ✅ Bajo costo de hardware                             │
│     Corre en GPU de 8GB (RTX 4060 o M-series)        │
│                                                         │
│  ✅ Disponible 24/7                                    │
│     Sin latencia de internet, sin dependencias        │
│                                                         │
└─────────────────────────────────────────────────────────┘

Horario: Cualquiera. Costo: Una sola vez (no recurrente)
```

---

## Slide 4: Arquitectura General

```
┌─────────────────────────────────────────────────────────┐
│                    USUARIO FINAL                        │
├─────────────────────────────────────────────────────────┤
│                         │                               │
│              ┌──────────▼──────────┐                    │
│              │  STREAMLIT WEB UI   │                   │
│              │  Chat + Voz + Foto  │                   │
│              │  (puerto 8501)      │                   │
│              └──────────┬──────────┘                    │
│                         │ HTTP POST                      │
│              ┌──────────▼──────────┐                    │
│              │  MAIN API (FastAPI) │                   │
│              │  Gemma-3N Fine-tune │                   │
│              │  (puerto 8000)      │                   │
│              └──────────┬──────────┘                    │
│                         │                               │
│          ┌──────────────┼──────────────┐               │
│          ▼              ▼              ▼               │
│     ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│     │ Inventa-│   │ Gastos  │   │Cosecha/ │  ...    │
│     │ rio API │   │ API     │   │Ingresos │          │
│     │8001     │   │8002     │   │8003/4   │          │
│     └─────────┘   └─────────┘   └─────────┘          │
│          │              │              │               │
│          └──────────────┼──────────────┘               │
│                         │                               │
│              ┌──────────▼──────────┐                    │
│              │  SQLite Databases   │                   │
│              │  (Persistencia)     │                   │
│              └─────────────────────┘                    │
│                                                         │
└─────────────────────────────────────────────────────────┘

Patrón: Microservicios + Modelo IA Offline
```

---

## Slide 5: Stack Tecnológico

```
┌─────────────────────────────────────────────────────────┐
│          TECNOLOGÍAS UTILIZADAS (Stack)             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  🧠 Modelo de IA                                        │
│    • Gemma-3N (6 mil millones parámetros)            │
│    • Fine-tuneado con datos de CENICAFE             │
│    • Multimodal (texto + imagen + audio)            │
│                                                         │
│  🔧 Backend                                            │
│    • FastAPI (framework HTTP)                        │
│    • Python 3.10+                                    │
│    • SQLAlchemy (ORM)                                │
│    • SQLite (persistencia)                           │
│                                                         │
│  🎨 Frontend                                           │
│    • Streamlit (web UI)                              │
│    • Audio recorder widget                           │
│    • Camera widget                                   │
│                                                         │
│  🎤 Audio                                              │
│    • Whisper (OpenAI, faster-whisper)               │
│    • Transcripción local en CPU                      │
│    • Español (es) con precisión alta                │
│                                                         │
│  🚀 Aceleración                                        │
│    • GPU: NVIDIA (CUDA 12.1+) o Apple Silicon (MPS) │
│    • PyTorch con optimizaciones                      │
│                                                         │
│  🐳 Orquestación                                       │
│    • Docker Compose (dev + producción pequeña)      │
│                                                         │
└─────────────────────────────────────────────────────────┘

Todas tecnologías open-source, sin dependencias propietarias
```

---

## Slide 6: El Modelo: Gemma-3N

```
┌─────────────────────────────────────────────────────────┐
│  GEMMA-3N: MODELO BASE                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Origen: Google AI                                     │
│  Parámetros: 6 mil millones                           │
│  Peso: 3.2 GB (comprimido int4)                       │
│  Velocidad: 0.5-2 seg por pregunta (GPU)             │
│  Multimodal: Texto + Imagen + Audio                 │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │         COMPARACIÓN CON OTROS MODELOS          │ │
│  ├─────────────────────────────────────────────────┤ │
│  │                                                 │ │
│  │  Modelo        │ Parámetros│ Hardware  │ Costo│ │
│  │  ─────────────────────────────────────────────  │ │
│  │  GPT-4         │ 1.7T      │ Cloud     │ API$ │ │
│  │  Llama 70B     │ 70B       │ A100 GPU  │ High │ │
│  │  Gemma-3N      │ 6B        │ 8GB GPU   │ Once │ │
│  │  DistilBERT    │ 66M       │ CPU       │ Rápido
│  │                                                 │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  Ventaja: Relación performance/costo/accesibilidad  │
│           es única para ML offline                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 7: Fine-tuning: Especialización

```
┌─────────────────────────────────────────────────────────┐
│  FINE-TUNING: DE GENERAL A ESPECIALISTA            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Analogía: Médico General vs Cardiólogo              │
│                                                         │
│                                                         │
│  ANTES (Base Model):                                   │
│  "¿Qué es la Roya?" → Respuesta genérica            │
│                                                         │
│         ↓ [Fine-tuning con café]                      │
│                                                         │
│  DESPUÉS (Especializado):                             │
│  "¿Qué es la Roya?" → Respuesta detallada            │
│                       sobre Roya del Café Colombiano  │
│                       con tratamientos locales        │
│                                                         │
│  ─────────────────────────────────────────────────── │
│  Datos de entrenamiento:                              │
│  ─────────────────────────────────────────────────── │
│                                                         │
│  1. CENICAFE Documents (1,000+ técnicos)              │
│     • Guías de cultivo                                │
│     • Monografías de plagas                          │
│     • Recomendaciones agrícolas                      │
│                                                         │
│  2. Image Dataset (2,616 imágenes etiquetadas)       │
│     • Hojas sanas vs enfermas                        │
│     • Tipos de plagas                                │
│     • Síntomas de enfermedades                       │
│                                                         │
│  3. Function Calling Pairs (2,700 ejemplos)          │
│     • Enseñar al modelo cuándo llamar BD            │
│     • Cuándo usar solo conocimiento                 │
│                                                         │
└─────────────────────────────────────────────────────────┘

Resultado: Accuracy en clasificación sube de 60% a 92%
```

---

## Slide 8: Microservicios

```
┌─────────────────────────────────────────────────────────┐
│  ¿POR QUÉ MICROSERVICIOS?                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ❌ MONOLITO (Todo junto)                              │
│     if falla BD → cae todo                            │
│     si crece → escala TODO                            │
│     actualizar 1 módulo → reinicia todo               │
│                                                         │
│  ✅ MICROSERVICIOS (Separado)                          │
│     if falla BD → API sigue viva                      │
│     si crece inventario → escala solo eso            │
│     actualizar BD → no afecta API                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  LOS 5 MICROSERVICIOS                          │ │
│  ├─────────────────────────────────────────────────┤ │
│  │                                                 │ │
│  │  1. Main API (8000) - Modelo IA               │ │
│  │  2. Inventario (8001) - Productos            │ │
│  │  3. Gastos (8002) - Costos mensuales         │ │
│  │  4. Cosecha (8003) - Producción              │ │
│  │  5. Ingresos (8004) - Ventas                 │ │
│  │  6. Streamlit (8501) - UI                    │ │
│  │                                                 │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  Cada uno es una API REST FastAPI independiente      │
│  Se comunican por HTTP (local o remoto)              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 9: Flujo de una Pregunta (Ejemplo Real)

```
┌─────────────────────────────────────────────────────────┐
│  FLUJO: USUARIO PREGUNTA CON VOZ + IMAGEN           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  PASO 1: Usuario graba voz + toma foto                │
│  "¿Qué enfermedad tiene esta hoja?"                  │
│                    │                                   │
│  PASO 2: Streamlit captura                             │
│         audio_bytes + image_bytes                     │
│                    │                                   │
│  PASO 3: Whisper transcribe (en CPU)                   │
│         audio → "¿Qué enfermedad tiene esta hoja?"  │
│                    │                                   │
│  PASO 4: POST a Main API (8000)                        │
│         /ask?question=...&image=...                    │
│                    │                                   │
│  PASO 5: API carga imagen a memoria                    │
│         PIL.Image.open(image_bytes)                    │
│                    │                                   │
│  PASO 6: API invoca modelo (Gemma-3N en GPU)          │
│         input: imagen + texto + system_prompt        │
│         output: respuesta en español (2-3 seg)       │
│                    │                                   │
│  PASO 7: Parsear salida (¿JSON o texto?)              │
│         if JSON → función calling (BD)                │
│         else → respuesta directa                      │
│                    │                                   │
│  PASO 8: Retornar JSON a Streamlit                     │
│         {"answer": "La hoja tiene roya..."}           │
│                    │                                   │
│  PASO 9: Streamlit muestra respuesta al usuario       │
│                                                         │
│  ⏱️  TOTAL: ~2 segundos (GPU)                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 10: Cómo Ejecutar - run-local.sh

```
┌─────────────────────────────────────────────────────────┐
│  EJECUTAR CON run-local.sh (Opción 1)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ¿Qué es un archivo .sh?                              │
│  → Shell Script (comando de terminal ejecutados)      │
│  → Automatiza tareas repetitivas                       │
│  → Como un batch de Windows pero para Linux/Mac       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 1: Descargar modelo (una sola vez)       │ │
│  │  $ python download.py                          │ │
│  │  → Descarga 3.2 GB a ./models/                 │ │
│  │  → Toma ~5-10 minutos                          │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 2: Hacer script ejecutable               │ │
│  │  $ cd app                                       │ │
│  │  $ chmod +x run-local.sh                       │ │
│  │  (chmod = change mode de permisos)             │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 3: Ejecutar script                       │ │
│  │  $ ./run-local.sh                              │ │
│  │  → Crea virtual environment                    │ │
│  │  → Instala dependencias                        │ │
│  │  → Detecta GPU (NVIDIA/Apple Silicon/CPU)     │ │
│  │  → Instala PyTorch apropiadamente             │ │
│  │  → Inicia 6 servicios secuencialmente         │ │
│  │  → Espera a que carguen                        │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 4: Abrir en navegador                    │ │
│  │  $ open http://localhost:8501                  │ │
│  │  (macOS) o xdg-open (Linux)                    │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 5: Para detener                          │ │
│  │  $ Ctrl+C en la terminal                       │ │
│  │  → El script limpia todo automáticamente       │ │
│  │  → Mata procesos en puertos 8000-8501         │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 11: Qué Hace run-local.sh (Internamente)

```
┌─────────────────────────────────────────────────────────┐
│  ¿QUÉ HACE EL SCRIPT INTERNAMENTE?                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣  CREAR VIRTUAL ENVIRONMENT                       │
│      $ python3 -m venv venv                            │
│      → Entorno aislado para dependencias             │
│      → No contamina sistema                          │
│                                                         │
│  2️⃣  ACTIVAR VIRTUAL ENVIRONMENT                     │
│      $ source venv/bin/activate                       │
│      → Cambia Python a usar dependencias locales      │
│                                                         │
│  3️⃣  DETECTAR PLATAFORMA Y GPU                       │
│      if uname == Darwin && uname -m == arm64:         │
│          # Apple Silicon (M1/M2/M3)                   │
│          pip install torch + MPS support              │
│      elif nvidia-smi exists:                          │
│          # NVIDIA GPU en Linux                        │
│          pip install torch + CUDA 12.6 support        │
│      else:                                            │
│          # CPU solamente                              │
│          pip install torch CPU version               │
│                                                         │
│  4️⃣  INSTALAR DEPENDENCIAS                            │
│      $ pip install -r requirements.txt                │
│      → fastapi, uvicorn, torch, streamlit, etc.      │
│                                                         │
│  5️⃣  INICIAR SERVICIOS SECUENCIALMENTE                │
│      cd databases/inventario                          │
│      python -m uvicorn main:app --port 8001 &        │
│      sleep 2  # Esperar a que se estabilice          │
│                                                         │
│      cd ../gastos                                     │
│      python -m uvicorn main:app --port 8002 &        │
│      sleep 2                                          │
│      ... (repetir para cosecha, ingresos, API)       │
│                                                         │
│      streamlit run web.py --port 8501 &              │
│      sleep 15  # Esperar a modelo cargue             │
│                                                         │
│  6️⃣  MOSTRAR STATUS Y URLS                            │
│      lsof -ti:8000 | xargs ... # Verificar puertos  │
│      echo "API en http://localhost:8000"             │
│      echo "UI en http://localhost:8501"              │
│                                                         │
│  7️⃣  ESPERAR INFINITAMENTE (CTRL+C)                  │
│      while true; do sleep 1; done                     │
│      → Script se queda corriendo                      │
│      → Ctrl+C ejecuta cleanup() automáticamente      │
│                                                         │
└─────────────────────────────────────────────────────────┘

Sin el script: tendrías que correr estos 6 comandos manualmente
Con el script: 1 comando hace todo + limpia al salir
```

---

## Slide 12: Cómo Ejecutar - Docker Compose

```
┌─────────────────────────────────────────────────────────┐
│  EJECUTAR CON DOCKER COMPOSE (Opción 2)            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ¿Qué es Docker Compose?                              │
│  → Orquestador de contenedores Docker                 │
│  → Archivo YAML describe qué contenedores y cómo      │
│  → Reemplaza 6 comandos "docker run" con 1 comando   │
│                                                         │
│  Ventaja: Entorno aislado, reproducible, escalable   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 1: Verificar Docker instalado            │ │
│  │  $ docker --version                            │ │
│  │  $ docker-compose --version                    │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 2: Descargar modelo                      │ │
│  │  $ python download.py                          │ │
│  │  (igual que con run-local.sh)                  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 3: Ir a directorio app                   │ │
│  │  $ cd app                                       │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 4: Iniciar servicios                     │ │
│  │  $ docker-compose up                           │ │
│  │  (o docker-compose up -d para background)      │ │
│  │                                                 │ │
│  │  Output esperado:                               │ │
│  │  inventario_1 | running on 0.0.0.0:8001       │ │
│  │  gastos_1     | running on 0.0.0.0:8002       │ │
│  │  api_1        | running on 0.0.0.0:8000       │ │
│  │  web_1        | running on 0.0.0.0:8501       │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 5: Ver logs                              │ │
│  │  $ docker-compose logs -f api                  │ │
│  │  $ docker-compose logs -f web                  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 6: Abrir en navegador                    │ │
│  │  $ open http://localhost:8501                  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  PASO 7: Para detener                          │ │
│  │  $ Ctrl+C (si está en foreground)              │ │
│  │  $ docker-compose down (si está en background) │ │
│  │  → Detiene y elimina contenedores              │ │
│  └─────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 13: Diferencia run-local.sh vs Docker Compose

```
┌─────────────────────────────────────────────────────────┐
│  run-local.sh vs DOCKER COMPOSE                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│              │ run-local.sh  │ Docker Compose         │
│  ─────────────────────────────────────────────────    │
│  Instala en  │ Tu PC/Mac      │ En contenedores       │
│  Aislamiento │ NO (comparte   │ SÍ (contenedores      │
│              │  sistema)      │  aislados)           │
│  Configurar  │ Manual         │ Automático (YAML)     │
│  Reproducib. │ Depende OS     │ Igual en cualquier PC │
│  Ideal para  │ Desarrollo     │ Testing + Producción  │
│  Dificultad  │ Más fácil      │ Requiere Docker       │
│  Performance │ Nativa         │ ~5% overhead          │
│  ─────────────────────────────────────────────────    │
│                                                         │
│  RECOMENDACIÓN:                                        │
│                                                         │
│  👨‍💻 Desarrollo inicial:                                │
│     Usa run-local.sh (más simple)                     │
│                                                         │
│  🧪 Testing en equipo:                                │
│     Usa Docker Compose (todos iguales)                │
│                                                         │
│  🚀 Producción:                                        │
│     Docker Compose + Kubernetes si escala             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 14: docker-compose.yml (Explicado)

```
┌─────────────────────────────────────────────────────────┐
│  ESTRUCTURA DEL docker-compose.yml                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  version: '3.8'                                         │
│  ↑ Versión de sintaxis de Docker Compose             │
│                                                         │
│  services:                                             │
│  ↑ Lista todos los contenedores                       │
│                                                         │
│    inventario:                                         │
│    ↑ Nombre del servicio (DNS interno)                │
│                                                         │
│      build: .                                          │
│      ↑ Usar Dockerfile en este directorio             │
│                                                         │
│      command: python -m uvicorn ...                    │
│      ↑ Comando que ejecuta dentro del contenedor      │
│                                                         │
│      ports:                                            │
│        - "8001:8001"                                   │
│      ↑ Puerto host:puerto contenedor                  │
│      (accesible como localhost:8001 desde afuera)     │
│                                                         │
│      volumes:                                          │
│        - ./app:/app/app                                │
│      ↑ Montar carpeta host en contenedor              │
│      (cambios en host se ven en contenedor)           │
│                                                         │
│      environment:                                      │
│        - DATABASE_URL=...                              │
│      ↑ Variables de entorno (configuración)           │
│                                                         │
│      networks:                                         │
│        - caficulbot-network                            │
│      ↑ Red Docker para que se vean entre sí           │
│                                                         │
│      depends_on:                                       │
│        - postgres                                      │
│      ↑ NO INICIAR este servicio antes que postgres   │
│                                                         │
│      deploy:                                           │
│        resources:                                      │
│          reservations:                                 │
│            devices:                                    │
│              - driver: nvidia                          │
│              count: 1                                  │
│      ↑ Reservar 1 GPU NVIDIA para este servicio      │
│                                                         │
│  networks:                                             │
│    caficulbot-network:                                 │
│      driver: bridge                                    │
│  ↑ Red compartida entre contenedores                  │
│                                                         │
│  volumes:                                              │
│    postgres_data:                                      │
│  ↑ Almacenamiento persistente (no se borra)          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 15: Mensajes en Logs

```
┌─────────────────────────────────────────────────────────┐
│  ¿CÓMO VER LOS LOGS? (para saber qué pasa)         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CON run-local.sh:                                     │
│  ─────────────────                                     │
│  $ tail -f logs/api.log                                │
│  $ tail -f logs/streamlit.log                          │
│  $ tail -f logs/inventario.log                         │
│                                                         │
│  CON Docker Compose:                                   │
│  ────────────────────                                  │
│  $ docker-compose logs -f           # Todos los logs  │
│  $ docker-compose logs -f api        # Solo API       │
│  $ docker-compose logs -f web        # Solo UI        │
│  $ docker-compose logs -f --tail=50 api              │
│    # Últimas 50 líneas de API                        │
│                                                         │
│  MENSAJES ESPERADOS (todo bien):                       │
│  ───────────────────────────────────                   │
│                                                         │
│  inventario_1 | INFO:     Uvicorn running on          │
│                 http://0.0.0.0:8001                    │
│                                                         │
│  api_1        | Fetching safetensors repo manifest    │
│               | Loading model weights (takes 10-15s)  │
│               | INFO:     Uvicorn running on          │
│                 http://0.0.0.0:8000                    │
│                                                         │
│  web_1        | 2026-04-29 10:30:15.234               │
│               | Streamlit app running on              │
│                 http://0.0.0.0:8501                    │
│                                                         │
│  MENSAJES DE ERROR:                                    │
│  ─────────────────────                                 │
│                                                         │
│  ❌ "Address already in use"                           │
│  → Puerto ocupado por otro proceso                    │
│  Solución: docker-compose down && docker-compose up  │
│                                                         │
│  ❌ "CUDA out of memory"                               │
│  → Modelo no cabe en GPU                              │
│  Solución: Usar CPU (lento) o GPU más grande         │
│                                                         │
│  ❌ "Connection refused" (8001)                        │
│  → BD no está lista                                   │
│  Solución: Esperar más o revisar logs de inventario  │
│                                                         │
│  ❌ "Model not found"                                  │
│  → Olvidaste descargar modelo                         │
│  Solución: python download.py                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 16: Casos de Uso Reales

```
┌─────────────────────────────────────────────────────────┐
│  CASOS DE USO (EJEMPLOS REALES)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣  DIAGNÓSTICO VISUAL (Solo IA)                    │
│  ──────────────────────────────────────               │
│  Usuario: Sube foto de hoja marrón                     │
│           "¿Qué enfermedad tiene?"                     │
│                                                         │
│  Sistema: Analiza imagen + datos de entrenamiento    │
│           (2,616 imágenes de café)                    │
│                                                         │
│  Respuesta: "Antracnosis. Síntomas: manchas           │
│             concéntricas. Tratamiento: Fungicida     │
│             cúprico o Mancozeb"                       │
│                                                         │
│  Tiempo: ~2 segundos (GPU)                            │
│                                                         │
│  ─────────────────────────────────────────────────── │
│                                                         │
│  2️⃣  CONSULTA OPERATIVA (IA + BD)                    │
│  ───────────────────────────────────────             │
│  Usuario: "¿Cuánto fungicida tengo?"                  │
│                                                         │
│  Sistema: Modelo detecta que necesita BD             │
│           → JSON: {"tool": "inventario_consulta",   │
│                     "argumentos": "producto=fungicida"
│           → Consulta BD                              │
│           → Genera respuesta natural                 │
│                                                         │
│  Respuesta: "Tienes 5 kg de fungicida cúprico.       │
│             Precio: $50,000 COP. Para 1 hectárea     │
│             en floración necesitas 180 kg"           │
│                                                         │
│  Tiempo: ~2-3 segundos (GPU + latencia BD)           │
│                                                         │
│  ─────────────────────────────────────────────────── │
│                                                         │
│  3️⃣  PREGUNTA TÉCNICA (Solo IA)                      │
│  ────────────────────────────────                    │
│  Usuario: "¿Cuál es el ciclo de vida de la Broca?" │
│                                                         │
│  Sistema: Responde de datos de entrenamiento        │
│           (CENICAFE documentos)                       │
│                                                         │
│  Respuesta: "Ciclo: 35-50 días a 25-28°C...         │
│             Manejo: recolección de fruto caído..."   │
│                                                         │
│  Tiempo: ~1 segundo (GPU)                            │
│                                                         │
│  ─────────────────────────────────────────────────── │
│                                                         │
│  4️⃣  VARIAS PREGUNTAS EN CHAT                         │
│  ──────────────────────────────                      │
│  Usuario mantiene una conversación de 10 preguntas  │
│  Sistema mantiene contexto (en desarrollo)           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 17: Rendimiento y Limitaciones

```
┌─────────────────────────────────────────────────────────┐
│  RENDIMIENTO Y LIMITACIONES                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  RENDIMIENTO (Hardware: RTX 4060 8GB):               │
│  ─────────────────────────────────────               │
│                                                         │
│  Pregunta texto:        0.8 segundos                  │
│  Pregunta + imagen:     1.3 segundos                  │
│  Pregunta + imagen + BD: 1.8 segundos                │
│  En CPU (sin GPU):      5-10 segundos               │
│                                                         │
│  ─────────────────────────────────────────────────── │
│                                                         │
│  LIMITACIONES CONOCIDAS:                              │
│  ─────────────────────────────────────              │
│                                                         │
│  ❌ No soporta múltiples usuarios concurrentes       │
│     → Si 2 usuarios preguntan → segundo espera      │
│     → Solución: Cola de procesamiento o más GPU     │
│                                                         │
│  ❌ Requiere GPU para performance                    │
│     → CPU es viable pero lento                       │
│     → Solución: Usar CPU para baja demanda          │
│                                                         │
│  ❌ Modelo offline → no aprende automáticamente      │
│     → Requiere reentrenamiento para nuevos datos    │
│     → Cada 3-6 meses: reentrenar con nuevos docs   │
│                                                         │
│  ❌ Requiere 3.2 GB en disco                         │
│     → No cabe en dispositivos muy antiguos          │
│     → Solución: Usar modelo más pequeño (Gemma-2B) │
│                                                         │
│  ⚠️  Respuestas pueden alucinar                       │
│     → Modelo puede inventar información             │
│     → Siempre validar con experto                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 18: Roadmap y Futuro

```
┌─────────────────────────────────────────────────────────┐
│  ROADMAP Y FUTURO                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CORTO PLAZO (1-2 meses):                             │
│  ─────────────────────────                            │
│  ✅ Agregar más imágenes de enfermedades            │
│  ✅ Implementar caché de respuestas                  │
│  ✅ Guardar historial de chat                        │
│  ✅ Mejorar UX de Streamlit                          │
│                                                         │
│  MEDIANO PLAZO (3-6 meses):                          │
│  ────────────────────────────                        │
│  ✅ Autenticación multi-usuario                      │
│  ✅ Sincronización con nube (backup)                 │
│  ✅ App móvil (Flutter o React Native)               │
│  ✅ Reportes automáticos (semanal/mensual)          │
│                                                         │
│  LARGO PLAZO (6-12 meses):                           │
│  ────────────────────────────                        │
│  ✅ Migración a SageMaker (escala)                   │
│  ✅ Monitoreo de modelos (MLOps)                     │
│  ✅ Integración con IoT (sensores de finca)          │
│  ✅ Predicciones de plagas (seasonal)                │
│                                                         │
│  VISIÓN A FUTURO:                                     │
│  ──────────────────                                   │
│  10,000 caficultores usando Caficulbot               │
│  Reducción de pérdidas por plagas: -30%              │
│  Ingresos aumentados: +20%                            │
│  Carbono neutral: Datos offline = menos emisiones   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 19: Conclusiones

```
┌─────────────────────────────────────────────────────────┐
│  CONCLUSIONES                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ☕ Caficulbot es un ejemplo práctico de:            │
│                                                         │
│  1️⃣  Arquitectura de Microservicios                  │
│      → Separación de responsabilidades               │
│      → Escalabilidad independiente                   │
│      → Resilencia (fallo aislado)                    │
│                                                         │
│  2️⃣  Fine-tuning de LLMs                             │
│      → Especialización a dominio específico          │
│      → Data-driven approach                          │
│      → Mejora de accuracy significativa              │
│                                                         │
│  3️⃣  MLOps Offline                                   │
│      → Inference sin conectividad                    │
│      → Hardware accesible (GPU 8GB)                  │
│      → Stack 100% open-source                        │
│                                                         │
│  4️⃣  Orquestación con Docker Compose                │
│      → Reproducibilidad                               │
│      → Facilidad de deployment                       │
│      → Transición a Kubernetes cuando escale        │
│                                                         │
│  ✅ APRENDIZAJES CLAVE:                               │
│                                                         │
│  • No necesitas GPUs de $10,000+ para ML efectivo   │
│  • Modelos pequeños bien fine-tuneados > modelos   │
│    grandes genéricos                                 │
│  • Arquitectura escalable empieza con microservicios │
│  • Docker Compose es suficiente para muchos casos   │
│  • Offline-first es viable y deseable en muchos casos │
│                                                         │
│  📊 IMPACTO POTENCIAL:                                │
│                                                         │
│  Reducción de costo: Agronomistas $500K/año        │
│  Mejora de diagnóstico: 60% accuracy → 92%          │
│  Disponibilidad: 24/7 vs horas de negocio          │
│  Escalabilidad: 1 finca → 100,000 fincas           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 20: Preguntas y Recursos

```
┌─────────────────────────────────────────────────────────┐
│  PREGUNTAS Y RECURSOS                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📚 DOCUMENTACIÓN GENERADA:                            │
│                                                         │
│  • CLAUDE.md - Guía de desarrollo                      │
│  • GUIA_PEDAGOGICA.md - Tutorial educativo            │
│  • PRESENTACION_TECNICA.md - Referencia técnica      │
│  • PRESENTATION_SLIDES.md - Esta presentación         │
│  • FINETUNING_GUIDE.md - Guía de fine-tuning         │
│  • README.md - Inicio rápido                          │
│  • TROUBLESHOOTING.md - Solución de problemas        │
│                                                         │
│  🔗 ENLACES ÚTILES:                                    │
│                                                         │
│  • Gemma-3N: https://huggingface.co/google/gemma-3n │
│  • FastAPI: https://fastapi.tiangolo.com/            │
│  • Streamlit: https://streamlit.io/                   │
│  • Docker Compose: https://docs.docker.com/compose/  │
│  • HuggingFace: https://huggingface.co/              │
│  • CENICAFE: https://www.cenicafe.org/               │
│                                                         │
│  👤 CONTACTO:                                         │
│                                                         │
│  Email: david.palacio@unosquare.com                   │
│  GitHub: [Usuario/Repo]                               │
│  LinkedIn: [Perfil]                                    │
│                                                         │
│  ❓ PREGUNTAS:                                        │
│                                                         │
│  1. ¿Cómo escalar a múltiples usuarios?              │
│     → Kubernetes + Load balancing + varias GPUs      │
│                                                         │
│  2. ¿Cómo mantener datos sincronizados?              │
│     → Message queues + master-slave DB setup        │
│                                                         │
│  3. ¿Cuál es el costo real?                          │
│     → GPU: $500-1000 (una sola vez)                  │
│     → Mantenimiento: solo servidor local             │
│                                                         │
│  4. ¿Cuánto tarda fine-tuning?                       │
│     → 3-5 horas en A100 con dataset actual          │
│                                                         │
└─────────────────────────────────────────────────────────┘

¡GRACIAS!
```

---

**Notas de Presentación:**

- Duración total: ~30 minutos (1.5 min por slide)
- Pausas interactivas después de slides 5, 10, 15
- Demo en vivo si es posible (10 minutos ejecutando Caficulbot)
- Distribuir GUIA_PEDAGOGICA.md y PRESENTACION_TECNICA.md después

