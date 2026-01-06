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

**Local development (Linux):**
```bash
chmod +x app/run-local.sh
./app/run-local.sh
```

This script:
- Creates/activates a Python virtual environment
- Installs dependencies including PyTorch with CUDA support if GPU is available
- Starts all 6 services sequentially with proper wait times
- Creates logs in `app/logs/` directory
- Runs cleanup on exit (Ctrl+C)

**Services started:**
- Inventario API: `http://localhost:8001` (port 8001)
- Gastos API: `http://localhost:8002` (port 8002)
- Cosecha API: `http://localhost:8003` (port 8003)
- Ingresos API: `http://localhost:8004` (port 8004)
- Main API: `http://localhost:8000` (port 8000)
- Streamlit Web: `http://localhost:8501` (port 8501)

**Docker Compose (alternative):**
```bash
cd app
docker-compose up
```

**Desktop GUI:**
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
