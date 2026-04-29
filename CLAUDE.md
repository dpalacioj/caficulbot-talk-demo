# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Caficulbot** is an offline multimodal AI assistant for Colombian coffee farmers. Built on a fine-tuned Gemma-3N-E2B model, it provides expert knowledge on coffee cultivation, pests, diseases, and farm management via text, voice, and image inputs—all without requiring internet connectivity.

**Key Technologies:**
- Fine-tuned Gemma-3N (6B parameters) for coffee domain expertise
- FastAPI microservices architecture
- Streamlit web interface
- SQLite databases for farm management (inventory, expenses, harvest, income)
- Whisper (faster-whisper) for voice transcription
- CUDA/GPU inference (tested on RTX 4060 8GB VRAM)

## Development Commands

### Starting the Application

#### Option 1: Local Development with run-local.sh (Recommended for Learning)

**What is a .sh file?**
A `.sh` file is a **shell script** - a text file containing terminal commands executed in sequence. It's like a batch file in Windows, but for Linux/macOS. Instead of typing 6 commands manually, the script automates everything.

**Why use run-local.sh instead of running commands manually?**
1. **Automation**: Creates virtual environment, installs dependencies, starts 6 services
2. **GPU detection**: Automatically detects NVIDIA/Apple Silicon/CPU and installs appropriate PyTorch
3. **Service orchestration**: Starts services in correct order with proper wait times
4. **Cleanup**: Kills all processes properly when you press Ctrl+C
5. **Logging**: Saves output to files for debugging

**Step-by-step execution:**

```bash
# Step 1: Download the fine-tuned model (one time only)
python download.py
# This downloads ~3.2 GB to ./models/
# Takes 5-10 minutes depending on internet speed

# Step 2: Navigate to app directory
cd app

# Step 3: Make the script executable (needed on Linux/macOS)
chmod +x run-local.sh
# Note: Windows doesn't need this step

# Step 4: Run the script
./run-local.sh
# On Windows, use: .\run-local.sh

# Step 5: Wait for all services to start
# Expected output:
# ✓ Servicio Inventario está activo en puerto 8001
# ✓ Servicio Gastos está activo en puerto 8002
# ✓ Servicio Cosecha está activo en puerto 8003
# ✓ Servicio Ingresos está activo en puerto 8004
# ✓ API Principal está activo en puerto 8000
# ✓ Interfaz Web está activo en puerto 8501
# This takes 30-40 seconds total (GPU needs time to load model)

# Step 6: Open in browser
# macOS:
open http://localhost:8501
# Linux:
xdg-open http://localhost:8501
# Windows: Manually open http://localhost:8501 in your browser

# Step 7: To stop all services
# Press Ctrl+C in the terminal
# The script automatically:
# - Kills all running services
# - Cleans up ports 8000-8004 and 8501
# - Removes temporary files
```

**What the script does internally (for understanding):**

```bash
#!/bin/bash

# 1. Create virtual environment (isolated Python)
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 2. Activate virtual environment
# This makes Python use local packages, not system packages
source venv/bin/activate

# 3. Detect GPU and install appropriate PyTorch
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        # Apple Silicon (M1/M2/M3/M4)
        pip install torch torchvision torchaudio  # With MPS support
    else
        # Intel Mac
        pip install torch torchvision torchaudio
    fi
elif command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU on Linux
    pip install torch ... --index-url https://download.pytorch.org/whl/cu126
else
    # CPU only
    pip install torch ... --index-url https://download.pytorch.org/whl/cpu
fi

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Start services one by one with waits
cd databases/inventario
python -m uvicorn main:app --port 8001 &  # Start in background
sleep 2  # Wait for service to stabilize

cd ../gastos
python -m uvicorn main:app --port 8002 &
sleep 2

# ... repeat for cosecha, ingresos, api ...

# 6. Start UI
streamlit run web.py --port 8501 &
sleep 15  # Wait longer for model to load

# 7. Loop forever (until Ctrl+C)
while true; do sleep 1; done
```

**Services started:**
- Inventario API: `http://localhost:8001` (products/inventory)
- Gastos API: `http://localhost:8002` (expenses)
- Cosecha API: `http://localhost:8003` (harvest)
- Ingresos API: `http://localhost:8004` (income)
- Main API: `http://localhost:8000` (model inference)
- Streamlit Web UI: `http://localhost:8501` (user interface)

**Troubleshooting run-local.sh:**

```bash
# Error: "Permission denied" or "command not found"
→ Forgot chmod +x? Try: chmod +x app/run-local.sh

# Error: "port 8000 already in use"
→ Kill the old process:
lsof -ti:8000 | xargs kill -9
# Then try again

# Error: "CUDA not found" or "GPU not detected"
→ Check that you have CUDA 12.1+ installed:
nvidia-smi
# If not available, the script will use CPU (slower)

# Error: "Model not found"
→ You forgot step 1:
python download.py

# Error: "ModuleNotFoundError: torch"
→ Virtual environment not activated or dependencies not installed:
source app/venv/bin/activate
pip install -r app/requirements.txt

# Services started but can't access http://localhost:8501
→ Wait 30 seconds more for model to load
→ Check logs: tail -f app/logs/api.log
```

---

#### Option 2: Docker Compose (Recommended for Production/Testing)

**What is Docker Compose?**
Docker Compose is a tool that manages multiple Docker containers as one system. Instead of typing `docker run` 6 times with 20 parameters each, you write a YAML file once and use `docker-compose up`.

**Advantages over run-local.sh:**
- **Isolation**: Containers don't affect your system
- **Reproducibility**: Works identically on any machine (Windows/Mac/Linux)
- **Scalability**: Easy to add more instances
- **Production-ready**: Industry standard

**Step-by-step execution:**

```bash
# Step 1: Verify Docker is installed
docker --version
# Output should be: Docker version 24.0.x or higher

# Step 2: Download the model (same as run-local.sh)
python download.py

# Step 3: Navigate to app directory
cd app

# Step 4: Build Docker images (downloads dependencies)
docker-compose build
# Takes 5-10 minutes (downloads ~5GB of layers)

# Step 5: Start all services
docker-compose up
# OR in background:
docker-compose up -d

# Step 6: Wait for services to start
# Expected output:
# inventario_1  | INFO:     Uvicorn running on http://0.0.0.0:8001
# gastos_1      | INFO:     Uvicorn running on http://0.0.0.0:8002
# api_1         | Loading model weights (this takes 15-20 seconds)
# api_1         | INFO:     Uvicorn running on http://0.0.0.0:8000
# web_1         | Streamlit app running on http://0.0.0.0:8501

# Step 7: Open in browser
open http://localhost:8501

# Step 8: View logs in real-time
docker-compose logs -f       # All services
docker-compose logs -f api   # Just the API
docker-compose logs -f web   # Just the UI

# Step 9: Check service status
docker-compose ps
# Shows which containers are running

# Step 10: Stop services
# If running in foreground:
Ctrl+C
# If running in background:
docker-compose down
# To remove everything including volumes:
docker-compose down -v
```

**Docker Compose file explanation (docker-compose.yml):**

```yaml
version: '3.8'  # Syntax version

services:       # List all containers
  inventario:   # First service (database service)
    build: .    # Build using Dockerfile in this directory
    command: python -m uvicorn databases.inventario.main:app --port 8001
    ports:
      - "8001:8001"  # Map host port 8001 to container port 8001
    volumes:
      - ./app:/app/app  # Share folder between host and container
    networks:
      - caficulbot-network  # Connect to shared network

  api:          # Second service (main model API)
    build: .
    command: python -m uvicorn api:app --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models  # Share model folder
    environment:
      # How to reach other services (by name, not localhost)
      - INVENTORY_API_BASE_URL=http://inventario:8001
      - EXPENSES_API_BASE_URL=http://gastos:8002
    depends_on:
      - inventario  # Don't start api until inventario is ready
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia  # Use NVIDIA GPU
              count: 1
              capabilities: [gpu]

  web:          # Third service (UI)
    build: .
    command: streamlit run web.py --port 8501
    ports:
      - "8501:8501"
    depends_on:
      - api  # Don't start UI until API is ready

networks:
  caficulbot-network:  # Virtual network for services to communicate
    driver: bridge

volumes:        # Persistent storage
  postgres_data:  # Data survives container restart
```

**Key Docker Compose concepts:**

| Concept | Explanation |
|---------|-------------|
| `build` | Build image from Dockerfile |
| `command` | What to run inside the container |
| `ports` | Map host_port:container_port |
| `volumes` | Share folder between host and container |
| `environment` | Variables available inside container |
| `depends_on` | Start order (don't start before dependencies) |
| `networks` | Services communicate by name (service name = hostname) |
| `deploy.resources` | Reserve GPU/CPU resources |

**Important: Service naming in Docker Compose**

Inside Docker Compose, services can reach each other by **service name**, not localhost:

```python
# ✓ CORRECT (inside api container):
response = requests.get("http://inventario:8001/consulta")

# ✗ WRONG (would try to reach itself):
response = requests.get("http://localhost:8001/consulta")
```

This is because Docker Compose creates an internal DNS that resolves service names.

**Troubleshooting Docker Compose:**

```bash
# Error: "Cannot connect to Docker daemon"
→ Docker is not running
→ macOS: open /Applications/Docker.app
→ Linux: sudo systemctl start docker

# Error: "port 8501 already in use"
→ Another container is using the port
docker-compose down
# Or check what's using it:
lsof -ti:8501 | xargs kill -9

# Error: "GPU not available"
→ Install nvidia-docker:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# Or check NVIDIA Docker runtime is configured

# Error: "Image build failed"
→ Delete old images and rebuild:
docker system prune -a
docker-compose build --no-cache

# View detailed logs
docker-compose logs -f --tail=100 api
```

---

#### Comparison: run-local.sh vs Docker Compose

| Feature | run-local.sh | Docker Compose |
|---------|--------------|-----------------|
| Install location | Your PC/Mac | In containers |
| Affects system | Yes (modifies) | No (isolated) |
| Reproducibility | Depends on OS | Same everywhere |
| Setup time | 5 minutes | 15 minutes (first time) |
| Learning curve | Easier | Medium |
| Production use | Not ideal | Recommended |
| When to use | Learning/development | Team/production |

---

**Desktop GUI (not recommended - deprecated):**
```bash
python app/main.py
```

### Model Setup

**Download the fine-tuned model:**
```bash
python download.py
```
or
```bash
python app/download.py
```

The model downloads from HuggingFace to `./models/` directory. The application expects the model at this path.

### Development Dependencies

```bash
pip install -r app/requirements.txt
```

## Architecture

### Service Communication Pattern

The system uses a **microservices architecture** where the main API (`app/api.py`) orchestrates calls to four independent database services:

```
┌─────────────┐
│ Streamlit   │ ← User interface (web.py)
│  (port 8501)│
└──────┬──────┘
       │
       ↓
┌─────────────┐      ┌──────────────┐
│  Main API   │─────→│ Inventario   │ (port 8001)
│ (port 8000) │      │ SQLite DB    │
└──────┬──────┘      └──────────────┘
       │
       ├────────────→ ┌──────────────┐
       │              │   Gastos     │ (port 8002)
       │              │ SQLite DB    │
       │              └──────────────┘
       │
       ├────────────→ ┌──────────────┐
       │              │   Cosecha    │ (port 8003)
       │              │ SQLite DB    │
       │              └──────────────┘
       │
       └────────────→ ┌──────────────┐
                      │  Ingresos    │ (port 8004)
                      │ SQLite DB    │
                      └──────────────┘
```

**Main API** (`app/api.py`):
- Loads the fine-tuned Gemma-3N model on startup using transformers pipeline
- Handles multimodal requests (text + optional image)
- Implements function calling via JSON parsing in model outputs
- Routes tool calls to appropriate microservices via HTTP
- Two system prompts: `SYSTEM_PROMPT` for text and `SYSTEM_PROMPT_IMAGE` for vision tasks

**Database Services** (all follow the same pattern):
- Each in `app/databases/{service_name}/`:
  - `database.py`: SQLAlchemy models and session management
  - `main.py`: FastAPI endpoints for CRUD operations
  - `{service_name}.db`: SQLite database file
- Use identical structure with different schemas
- Accessed via HTTP from main API

**Streamlit Frontend** (`app/web.py`):
- Maintains chat history in session state
- Supports text input, voice recording (audio-recorder-streamlit), camera capture, and file upload
- Transcribes audio locally with Whisper (faster-whisper, `small` model on CPU)
- Sends multipart form data to main API

### Function Calling Mechanism

The model is fine-tuned to output JSON when administrative queries are detected:

```python
# Model outputs:
{"tool": "inventario_consulta", "argumentos": "producto=fertilizante"}
{"tool": "gastos_consulta", "argumentos": "mes=1,año=2024"}

# Parsed in app/api.py via parse_tool_call()
# Then routed to corresponding microservice
```

This is explicitly controlled in `SYSTEM_PROMPT` in `app/api.py:25-42` to minimize false positives.

### Database Schema Pattern

All four databases follow this pattern (example from Gastos):

```python
class Gasto(Base):
    __tablename__ = "gastos"
    id = Column(Integer, primary_key=True)
    año = Column(Integer, nullable=False)
    mes = Column(Integer, nullable=False)
    categoria = Column(String, nullable=True)
    monto = Column(Float, nullable=False)
```

Similar schemas exist for:
- **Inventario**: producto, cantidad
- **Cosecha**: (harvest data)
- **Ingresos**: (income data)

## Key Files

- `app/api.py` - Main FastAPI backend, model loading, function calling logic
- `app/web.py` - Streamlit UI with multimodal input handling
- `app/run-local.sh` - Production startup script with service orchestration
- `app/databases/{service}/main.py` - Individual FastAPI microservices
- `app/databases/{service}/database.py` - SQLAlchemy models and DB setup
- `download.py` / `app/download.py` - Model download from HuggingFace

## System Requirements

- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 4060 tested)
- **CUDA**: 12.1+ recommended for PyTorch compatibility

The application can run on CPU but will be significantly slower for inference.

## Fine-Tuning Context

The model is based on `gemma-3n-E2B` (6B parameters) and fine-tuned on:
1. 1,000+ CENICAFE technical documents (QA pairs)
2. 2,616 labeled coffee pest/disease images
3. 2,700 instruction-function calling pairs

**Resources:**
- Fine-tuned model: `sergioq2/gemma-3N-finetune-coffe_q4_off` (HuggingFace)
- Notebooks: `dataset/qa_generation.ipynb`, `dataset/function_calling.ipynb`, `fine_tuning/gemma3n_finetuning_coffeagent.ipynb`

## Working with the Code

### Adding a New Database Service

1. Create directory: `app/databases/{service_name}/`
2. Copy structure from existing service (e.g., `gastos/`)
3. Define schema in `database.py`
4. Implement endpoints in `main.py`
5. Add port allocation in `app/run-local.sh`
6. Update `app/api.py` to add function calling support if needed

### Modifying System Prompts

**Critical**: System prompts in `app/api.py` control when function calling is triggered:
- `SYSTEM_PROMPT` (line 25): Text-only queries
- `SYSTEM_PROMPT_IMAGE` (line 44): Vision queries

Be conservative with tool usage instructions to avoid false positives on domain knowledge questions.

### Response Extraction

Model outputs can be in various formats. The extraction chain in `app/api.py` handles this:
1. `extract_content_from_response()` - Handles str/list/dict response formats
2. `extract_response_content()` - Unwraps JSON if model outputs `{"respuesta": "..."}`
3. `parse_tool_call()` - Detects function calls in JSON format

### Voice Input Flow

1. User records audio in Streamlit (`audio-recorder-streamlit`)
2. Audio saved as temporary WAV file
3. `faster_whisper` transcribes with language="es", beam_size=5
4. Transcription sent to main API as text query

### Image Input Flow

1. User uploads image or captures from camera
2. Image stored in session state as bytes
3. Sent to `/ask` endpoint as multipart form data
4. PIL Image opened from bytes
5. Passed to pipeline with `SYSTEM_PROMPT_IMAGE` and role-based messages

## Common Issues

**Model not found error**: Run `python download.py` to download the model to `./models/`

**Port already in use**: The `run-local.sh` script kills processes on ports 8000-8004 and 8501 on startup

**GPU not detected**: Script automatically falls back to CPU PyTorch. Check CUDA installation if GPU should be available.

**Service startup timing**: Services have hardcoded sleep delays in `run-local.sh`. API requires ~15s to load model before Streamlit starts.
