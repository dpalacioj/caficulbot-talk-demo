# ☕ Caficulbot: Offline AI Assistant for Coffee Farmers

**Caficulbot** is a fully offline, multimodal AI assistant built to empower Colombian coffee farmers with expert knowledge on cultivation, pests, diseases, and farm management. It provides real-time answers to questions, processes images of plant diseases, and performs administrative tasks, all without needing an internet connection.

---

## 💡 Example Questions (in Spanish)

- ¿Cómo controlar la roya?
- ¿Cómo debe ser el secado en el café?
- (Upload an image of a plant) → ¿Qué enfermedad tiene?

---

## 📁 Project Structure

```plaintext
.
├── app
│   ├── api.py               # FastAPI backend that interfaces with the fine-tuned LLM
│   ├── web.py               # Streamlit frontend
│   ├── main.py              # Local GUI launcher using tkinter
│   ├── run-local.sh         # Launches all services and UIs
│   └── databases/
│       ├── gastos/
│       │   ├── gastos.db
│       │   ├── database.py
│       │   └── main.py      # FastAPI service (port 8002)
│       ├── ingresos/
│       ├── inventario/
│       └── cosecha/         # Each folder has the same structure
├── models/                  # Folder where the model is downloaded to
├── download.py              # Downloads the fine-tuned model from Hugging Face
├── requirements.txt
├── dataset/
│   ├── qa_generation.ipynb      # Generates QA dataset from CENICAFE docs
│   └── function_calling.ipynb   # Generates function-calling samples
├── fine_tuning/
│   └── gemma3n_finetuning_coffeagent.ipynb
'''
```

## 🔄 Modifications from Original

This fork includes the following enhancements:

### **Cross-Platform Support**
- Added macOS Apple Silicon (M-series) support with MPS acceleration
- Automatic device detection (MPS/CUDA/CPU) in `app/api.py`
- Modified `app/run-local.sh` for macOS compatibility
- Fixed model path resolution for different working directories

### **Documentation**
- `PRESENTACION_TECNICA.md` - Comprehensive technical documentation (Spanish, 2,122 lines)
- `PRESENTATION_SLIDES.md` - Presentation slides for technical talks (English, 17 slides)
- `FINETUNING_GUIDE.md` - Fine-tuning execution guide with platform recommendations
- `CLAUDE.md` - Development guide and architecture overview

### **Configuration**
- Environment variable loading in `download.py` with `.env` support
- Streamlit configuration files for non-interactive startup
- Enhanced `.gitignore` for development artifacts and compiled files

### **Performance Benchmarks (MacBook Pro M4 Max)**
- Text latency: 1.8s average
- Image latency: 3.2s average
- Throughput: 105 tokens/sec (text), 64 tokens/sec (image)
- VRAM usage: 6GB on MPS
- 12.8x faster than CPU with MPS acceleration

**Original project purpose**: Production deployment for Colombian coffee farmers
**This fork's purpose**: Educational demonstration and technical presentation

---

# Local Setup

## 🖥️ System Requirements

### **Original Requirements (Linux/NVIDIA)**
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with at least **8GB VRAM**
- ✅ Tested on: **NVIDIA RTX 4060 (8GB VRAM)**
- **CUDA**: Compatible with `torch` and `transformers` (CUDA 12.1+ suggested)

### **Modified Version (macOS Support)**
- **OS**: macOS (tested on macOS Sequoia 15.4)
- **Python**: 3.12+ (recommended for Apple Silicon)
- **Hardware**: Apple Silicon (M-series chips)
- ✅ **Tested on**: MacBook Pro M4 Max (36GB RAM, 32-core GPU)
- **Acceleration**: Metal Performance Shaders (MPS)


## 🔧 Installation

### **1. Clone the repository**

```bash
git clone https://github.com/dpalacioj/caficulbot-talk-demo.git
cd caficulbot-talk-demo
```

### **2.Make sure the script is executable**
```bash
chmod +x app/run-local.sh
```

### **3.Run the app**
```bash
./app/run-local.sh
```

This will start the following services:

api.py → http://localhost:8000

Inventory service → http://localhost:8001

Expenses service → http://localhost:8002

Harvest service → http://localhost:8003

Income service → http://localhost:8004

Frontend (Streamlit) → http://localhost:8501

**Launch the Desktop App**
To run the GUI-based version using tkinter:
```bash
python app/main.py
```


## 📘 Fine-Tuning Notebook Overview
To adapt the multimodal Gemma-3n-E2B model to the specific context of Colombian coffee farming, a dedicated fine-tuning notebook was developed. This notebook orchestrates the training pipeline across three stages:

### Model Initialization
Loads the 6B parameter gemma-3n-E2B (Instruct) model with support for text, image, and audio inputs, optimized for local deployment.

### Data Preparation
Integrates three custom datasets:

Over 1,000 technical documents from Cenicafé for question–answer fine-tuning.
2,616 labeled images of coffee pests and diseases for vision-layer adaptation.
2,700 instruction-function pairs to enable structured function calling.

### Training Execution
Applies parameter-efficient fine-tuning techniques to adapt the model without full retraining, enabling high accuracy and responsiveness in offline environments.


**Models and Datasets**
| Resource                 | Link                                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| Fine-tuned Model         | [gemma-3N-finetune-coffe\_q4\_off](https://huggingface.co/sergioq2/gemma-3N-finetune-coffe_q4_off)        |
| QA Dataset (CENICAFE)    | [sergioq2/coffe](https://huggingface.co/datasets/sergioq2/coffe)                                          |
| Function Calling Dataset | [sergioq2/functioncalling\_coffedata](https://huggingface.co/datasets/sergioq2/functioncalling_coffedata) |
| Image Dataset (Roboflow) | [Coffee Pests and Diseases](https://app.roboflow.com/detection-3nbwx/coffe-mw9n0/2/export)                |



**Optional: Dataset Generation & Fine-Tuning**
These notebooks are optional and used only to replicate dataset generation or model training:
dataset/qa_generation.ipynb → Builds QA pairs from CENICAFE documents
dataset/function_calling.ipynb → Builds function-calling samples
fine_tuning/gemma3n_finetuning_coffeagent.ipynb → Fine-tunes Gemma-3N using Unsloth
To use OpenAI APIs, create a .env file with your API key.


**Test the Application**
Once services are running:
Open the browser at:
http://localhost:8501

Try asking:
¿Cómo controlar la roya?
¿Cómo debe ser el secado en el café?
Upload an image and ask: ¿Qué enfermedad tiene?


## 📝 License & Contact

### **Original Author**
**Author**: Sergio Quintero

### **Fork Maintainer**
**Adapted by**: David Palacio
**Repository**: [dpalacioj/caficulbot-talk-demo](https://github.com/dpalacioj/caficulbot-talk-demo)
