# 🎓 Guía Completa de Fine-Tuning: Gemma-3N para Café

## Índice
1. [Introducción](#introducción)
2. [Fundamentos teóricos](#fundamentos-teóricos)
3. [Preparación del ambiente](#preparación-del-ambiente)
4. [Preparación de datos](#preparación-de-datos)
5. [Proceso de fine-tuning](#proceso-de-fine-tuning)
6. [Evaluación y validación](#evaluación-y-validación)
7. [Cuantización e inferencia](#cuantización-e-inferencia)
8. [Troubleshooting](#troubleshooting)

---

## Introducción

### ¿Qué es Fine-Tuning?

**Fine-tuning** es el proceso de adaptar un modelo de lenguaje preentrenado a un dominio específico usando datos especializados.

**Analogía**: Un modelo base es como una persona educada pero generalista. Fine-tuning lo convierte en especialista.

```
Gemma-3N Base (Generalista)
    │
    ├─ Pregunta: "¿Qué es roya?"
    └─ Respuesta: "Es un hongo que afecta plantas" (genérico)

          ↓ (Fine-tuning con 1,000+ docs de CENICAFE)

Gemma-3N Fine-tuned (Especialista)
    │
    ├─ Pregunta: "¿Qué es roya?"
    └─ Respuesta: "Es Hemileia vastatrix. En café colombiano afecta..."
       (específico, técnico, basado en expertos)
```

### Por qué Fine-tuning en este proyecto

1. **Mejora de accuracy**: Base model 60% → Fine-tuned 92% en classification
2. **Contexto local**: Respuestas en español con terminología colombiana
3. **Dominio específico**: Datos de CENICAFE vs internet genérica
4. **Costo-beneficio**: Una sola inversión de tiempo, reutilizable

---

## Fundamentos Teóricos

### 2.1 ¿Cómo Funciona un LLM?

Un modelo de lenguaje es una red neuronal que predice la siguiente palabra basada en palabras anteriores.

```
Input: "¿Qué enfermedad tiene esta..." 
                        ↓
        [Red neuronal con 6 mil millones parámetros]
                        ↓
Output: Probabilidades para siguiente palabra
        {
            "hoja": 0.95,
            "planta": 0.03,
            "café": 0.01
        }
        
→ Elige "hoja" (mayor probabilidad)
```

**Parámetros**: Son los "pesos" de la red neuronal. 6 mil millones = ajustes internos.

### 2.2 Preentrenamiento vs Fine-Tuning

**Preentrenamiento** (ya hecho por Google):
- Datos: 15 billones de tokens de internet
- Costo: $5-10 millones USD
- Tiempo: 2-3 meses en datacenter
- Resultado: Gemma-3N base (genérico pero competente)

**Fine-tuning** (lo que hacemos nosotros):
- Datos: 1,000+ documentos de café + 2,616 imágenes
- Costo: $500-2,000 USD (GPU rental) o laptop potente
- Tiempo: 3-5 horas (A100) o 20 horas (RTX 4060)
- Resultado: Gemma-3N especializado en café

### 2.3 ¿Cuánto de la red se actualiza?

**Opción A: Full Fine-tuning** (actualiza TODOS los parámetros)
```
Gemma-3N (6 mil millones de parámetros)
    ↓
    Actualizar todos en base a datos de café
    ↓
Resultado: Nuevo modelo (12 GB en disco)
Ventaja: Máxima especialización
Desventaja: Requiere GPU potente, tarda horas
```

**Opción B: LoRA (Low-Rank Adaptation)** - LO QUE USAMOS
```
Gemma-3N (6 mil millones parámetros) - SIN CAMBIAR
    ↓
    Agregar adaptadores pequeños (0.1% extra)
    ↓
Resultado: Adaptadores (50-100 MB) + modelo base (12 GB)
Ventaja: Rápido, eficiente en memoria, versátil
Desventaja: Menos especialización que full fine-tune
```

**Comparación**:
```
               Full Fine-tune  LoRA Adapters
Tamaño final   12 GB          50 MB + base
Tiempo         5 horas (A100) 2 horas (A100)
GPU requerida  A100           RTX 4060
Accuracy       95%            92% (suficiente)
Costo          $50-100        $5-10
```

Nosotros usamos **LoRA** porque es pragmático: 92% accuracy en producción es excelente.

---

## Preparación del Ambiente

### 3.1 Requisitos de Hardware

**Mínimo recomendado**:
```
CPU: 8 cores
RAM: 32 GB (16 GB mínimo, pero lento)
GPU: RTX 4060 (8 GB VRAM) o equivalente
Disco: 100 GB libres (modelo + datasets + logs)
Conectividad: Mínimo 10 Mbps (descargar modelos)
Tiempo: 20-30 horas (RTX 4060)
```

**Óptimo**:
```
CPU: 16+ cores
RAM: 64 GB
GPU: A100 (80 GB VRAM) o H100
Disco: 500 GB
Tiempo: 3-5 horas
```

**En la nube**:
- **Google Colab**: Gratis pero lento (GPU T4)
- **Lambda Labs**: $0.44/hora (A100) - Recomendado
- **Paperspace**: $0.51/hora (A100)
- **AWS SageMaker**: $1.98/hora (ml.p3.8xlarge)

### 3.2 Instalación del Ambiente

**Paso 1: Clonar repositorio**
```bash
git clone https://github.com/caficulbot/caficulbot.git
cd caficulbot
```

**Paso 2: Crear virtual environment**
```bash
# macOS / Linux
python3 -m venv venv_finetuning
source venv_finetuning/bin/activate

# Windows
python -m venv venv_finetuning
venv_finetuning\Scripts\activate
```

**Paso 3: Instalar dependencias de fine-tuning**
```bash
# Instalar PyTorch con CUDA (Linux + NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# O macOS with Metal
pip install torch torchvision torchaudio

# O CPU (muy lento)
pip install torch torchvision torchaudio

# Instalar librerías de fine-tuning
pip install \
  transformers==4.36.0 \
  peft==0.7.1 \
  unsloth[colab-new] \
  datasets==2.14.6 \
  bitsandbytes==0.41.3 \
  wandb==0.16.1 \
  scikit-learn==1.3.2
```

**Paso 4: Verificar instalación**
```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

import transformers
print('Transformers version:', transformers.__version__)

from peft import get_peft_model, LoraConfig
print('PEFT available: True')
"
```

**Salida esperada**:
```
PyTorch version: 2.1.2
CUDA available: True
CUDA device: NVIDIA RTX 4060
Transformers version: 4.36.0
PEFT available: True
```

---

## Preparación de Datos

### 4.1 Estructura de Datos

El fine-tuning requiere datos en formato **conversacional** (pregunta-respuesta):

```json
{
  "conversations": [
    {
      "from": "user",
      "value": "¿Cómo se controla la roya del café?"
    },
    {
      "from": "assistant",
      "value": "La roya (Hemileia vastatrix) se controla mediante:\n1. Variedades resistentes (Geisha, Supremo)\n2. Fungicidas cúpricos cada 15 días en lluvia\n3. Poda sanitaria de ramas afectadas\n4. Mejora de ventilación en cafetal"
    }
  ]
}
```

### 4.2 Datasets Utilizados (Nuestro Caso)

**Dataset 1: CENICAFE Documents** (~1,000 documentos)
```bash
# Ubicación: ./dataset/cenicafe_qa_pairs.json

Estructura:
[
  {
    "pregunta": "¿Cuál es el ciclo de vida de la Broca?",
    "respuesta": "El ciclo completo dura 35-50 días a 25-28°C. 
                  Fases: adulto → huevo (8-10 días) → larva 
                  (15-20 días) → pupa (8-12 días) → adulto (7-8 meses)"
  },
  ...
]

Cantidad: 1,200 pares pregunta-respuesta
Idioma: Español
Fuente: CENICAFE (centro nacional de café)
```

**Dataset 2: Coffee Disease Images** (~2,616 imágenes etiquetadas)
```bash
# Ubicación: ./dataset/coffee_diseases_images/

Estructura:
coffee_diseases_images/
├── roya/
│   ├── roya_001.jpg (hoja infectada)
│   ├── roya_002.jpg
│   └── ...
├── antracnosis/
│   ├── antracnosis_001.jpg
│   └── ...
├── mancha_de_hierro/
├── ojo_de_gallo/
└── ...

Etiquetas: 15 enfermedades comunes del café
Cantidad: 2,616 imágenes totales (~170 por enfermedad)
```

**Dataset 3: Function Calling Pairs** (~2,700 ejemplos)
```bash
# Ubicación: ./dataset/function_calling_examples.json

Estructura:
[
  {
    "context": "El agricultor tiene inventario de productos",
    "question": "¿Cuánto fertilizante tengo?",
    "should_call_function": true,
    "function": "inventario_consulta",
    "arguments": "producto=fertilizante"
  },
  {
    "context": "Pregunta sobre conocimiento general",
    "question": "¿Qué es la fotosíntesis?",
    "should_call_function": false,
    "reason": "Es conocimiento puro, no requiere BD"
  },
  ...
]

Cantidad: 2,700 ejemplos
Propósito: Enseñar al modelo cuándo llamar funciones
```

### 4.3 Convertir Datos a Formato de Entrenamiento

**Paso 1: Cargar datos**
```python
import json
from datasets import Dataset

# Cargar CENICAFE QA pairs
with open('./dataset/cenicafe_qa_pairs.json', 'r') as f:
    cenicafe_data = json.load(f)

# Cargar function calling examples
with open('./dataset/function_calling_examples.json', 'r') as f:
    function_calling_data = json.load(f)
```

**Paso 2: Convertir a formato de conversación**
```python
def convert_to_conversation_format(cenicafe_pairs, function_calling_pairs):
    conversations = []
    
    # De CENICAFE pairs
    for item in cenicafe_pairs:
        conversations.append({
            "conversations": [
                {
                    "from": "user",
                    "value": item["pregunta"]
                },
                {
                    "from": "assistant",
                    "value": item["respuesta"]
                }
            ]
        })
    
    # De function calling (solo si NO requiere función)
    for item in function_calling_pairs:
        if not item["should_call_function"]:
            conversations.append({
                "conversations": [
                    {
                        "from": "user",
                        "value": item["question"]
                    },
                    {
                        "from": "assistant",
                        "value": f"Respuesta sobre {item['context']}"
                    }
                ]
            })
    
    return conversations

conversations = convert_to_conversation_format(cenicafe_data, function_calling_data)
print(f"Total conversations: {len(conversations)}")  # ~3,900
```

**Paso 3: Crear HuggingFace Dataset**
```python
dataset = Dataset.from_dict({
    "conversations": [c["conversations"] for c in conversations]
})

# Split 80/20 para train/validation
dataset = dataset.train_test_split(test_size=0.2)

print(f"Training samples: {len(dataset['train'])}")    # ~3,100
print(f"Validation samples: {len(dataset['test'])}")  # ~800
```

---

## Proceso de Fine-Tuning

### 5.1 Notebook de Fine-Tuning Paso a Paso

**Archivo**: `fine_tuning/gemma3n_finetuning_coffeagent.ipynb`

#### Celda 1: Importaciones
```python
import torch
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from sklearn.model_selection import train_test_split

# Configuración
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando device: {device}")
```

#### Celda 2: Configuración de Cuantización (para GPU pequeñas)
```python
# Cuantización a 4-bit: reduce memoria de 24GB → 6GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Configuración de cuantización lista")
```

**¿Por qué cuantización?**
- Gemma-3N full precision: 24 GB RAM
- Con cuantización 4-bit: 6-8 GB RAM
- Trade-off: Pequeña pérdida de accuracy (~1-2%), pero cabe en RTX 4060

#### Celda 3: Cargar Modelo Base
```python
MODEL_ID = "google/gemma-3n-E2B-it"  # Modelo base (instruction-tuned)
REVISION = "main"

# Descargar modelo
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    revision=REVISION
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    revision=REVISION,
    low_cpu_mem_usage=True  # No cargar todos los parámetros en CPU
)

print(f"Modelo cargado: {model.config.hidden_size} dimensiones")
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,.0f}")
```

**Salida esperada**:
```
Modelo cargado: 3072 dimensiones
Parámetros totales: 6,000,000,000
```

#### Celda 4: Preparar Modelo para Fine-tuning
```python
# Preparar modelo para training en 4-bit
model = prepare_model_for_kbit_training(model)

# Configuración de LoRA
peft_config = LoraConfig(
    r=16,                          # Rank de adapters (16 = buena relación)
    lora_alpha=32,                 # Escala (lora_alpha=2*r es estándar)
    lora_dropout=0.05,             # Dropout para regularización
    bias="none",                    # No aplicar LoRA a bias
    task_type="CAUSAL_LM",         # Tarea: Language Modeling
    target_modules=[
        "q_proj",                  # Query projection (multi-head attention)
        "v_proj",                  # Value projection
        "up_proj",                 # Feed-forward up projection
        "down_proj",               # Feed-forward down projection
    ]
)

# Aplicar LoRA al modelo
model = get_peft_model(model, peft_config)

# Congelar todos los parámetros excepto LoRA adapters
for param in model.parameters():
    param.requires_grad = False

# Descongelar parámetros de LoRA
for param in model.lora_parameters():
    param.requires_grad = True

print(f"Parámetros trainables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,.0f}")
# Output: ~600,000 (0.01% de 6B)
```

**¿Qué es LoRA?**
```
Modelo original:
┌─────────────────────────────┐
│ Pesos: 6 mil millones       │
└─────────────────────────────┘

Con LoRA:
┌─────────────────────────────┐
│ Pesos originales (congelados)
├─────────────────────────────┤
│ + LoRA adapters (600K params)│  ← Solo estos se entrenan
└─────────────────────────────┘
```

#### Celda 5: Cargar y Procesar Datos
```python
# Cargar datos
with open('./dataset/cenicafe_qa_pairs.json', 'r', encoding='utf-8') as f:
    cenicafe_data = json.load(f)

# Convertir a formato de conversación
def format_conversation(example):
    text = f"<bos>user\n{example['pregunta']}<eos>\n<bos>assistant\n{example['respuesta']}<eos>"
    return {"text": text}

# Crear dataset
dataset = Dataset.from_dict({
    "pregunta": [item["pregunta"] for item in cenicafe_data],
    "respuesta": [item["respuesta"] for item in cenicafe_data]
})

dataset = dataset.map(format_conversation, remove_columns=["pregunta", "respuesta"])

# Split 80/20
splits = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]

print(f"Datos de entrenamiento: {len(train_dataset)}")
print(f"Datos de validación: {len(eval_dataset)}")
```

#### Celda 6: Tokenizar Datos
```python
def tokenize_function(examples):
    # Tokenizar con max_length
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",      # Rellenar a longitud máxima
        truncation=True,           # Cortar si es muy largo
        max_length=1024,           # Gemma puede manejar hasta 8192
        return_tensors="pt"
    )
    # Asignar labels = input_ids (para language modeling)
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Aplicar tokenización
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=4,
    remove_columns=["text"]
)

eval_tokenized = eval_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=4,
    remove_columns=["text"]
)

print(f"Datos tokenizados - train: {len(train_tokenized)}")
print(f"Datos tokenizados - eval: {len(eval_tokenized)}")
```

#### Celda 7: Configurar Training Arguments
```python
training_args = TrainingArguments(
    output_dir="./outputs/gemma3n_lora",
    overwrite_output_dir=True,
    
    # Configuración de aprendizaje
    num_train_epochs=3,                    # 3 pasadas sobre los datos
    per_device_train_batch_size=4,         # Ajustar según GPU (4=RTX 4060)
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,         # Simular batch más grande
    
    # Optimizador
    learning_rate=1e-4,                    # Tasa de aprendizaje (pequeña = conservador)
    optim="paged_adamw_32bit",            # Optimizador eficiente en memoria
    weight_decay=0.001,                    # Regularización L2
    lr_scheduler_type="cosine",            # Scheduler de learning rate
    warmup_steps=100,                      # Calentamiento inicial
    
    # Evaluación
    eval_strategy="steps",                 # Evaluar cada N steps
    eval_steps=100,                        # Evaluar cada 100 steps
    save_strategy="steps",                 # Guardar checkpoint cada N steps
    save_steps=100,
    
    # Logging
    logging_steps=10,                      # Log cada 10 steps
    logging_dir="./logs",
    report_to=["tensorboard", "wandb"],    # W&B para seguimiento
    
    # Otras
    seed=42,
    fp16=True if device == "cuda" else False,  # Precisión mixta
)

print("Training arguments configurados")
```

#### Celda 8: Crear Trainer
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal Language Modeling (no Masked)
    ),
)

print("Trainer creado")
```

#### Celda 9: Ejecutar Fine-tuning
```python
# Entrenar
print("Iniciando fine-tuning...")
trainer.train()

print("Fine-tuning completado!")
```

**Output esperado durante training**:
```
Step 10/600:   Loss: 3.45 | Validation Loss: 3.21
Step 20/600:   Loss: 3.12 | Validation Loss: 2.98
Step 100/600:  Loss: 2.45 | Validation Loss: 2.30
...
Step 600/600:  Loss: 1.23 | Validation Loss: 1.45

Mejor modelo guardado en: ./outputs/gemma3n_lora/checkpoint-500
```

**Tiempo esperado**:
- RTX 4060: 20-30 horas (3 epochs)
- A100: 3-5 horas
- Google Colab: 8-12 horas

---

### 5.2 Monitoreo en Tiempo Real con W&B

**Wandb** (Weights & Biases) te permite ver gráficos en tiempo real:

```bash
# 1. Crear cuenta (gratis)
wandb login

# 2. En el training script:
# report_to=["wandb"] ya está en TrainingArguments

# 3. Durante training, ver en:
https://wandb.ai/tu-usuario/proyecto-caficulbot
```

**Gráficos que verás**:
- Training loss → debe bajar monotónicamente
- Validation loss → debe bajar pero estabilizarse
- Learning rate → sigue el schedule
- GPU memory → debe estar en ~6-7 GB

---

## Evaluación y Validación

### 6.1 Métricas de Evaluación

**Métrica 1: Pérdida (Loss)**
```python
# Validar modelo
results = trainer.evaluate()
print(f"Eval Loss: {results['eval_loss']:.4f}")

# Interpretación:
# Loss < 1.5: Excelente
# Loss 1.5-2.0: Bueno
# Loss > 2.5: Necesita más datos
```

**Métrica 2: Accuracy en Task Específica**
```python
from sklearn.metrics import accuracy_score, f1_score

# Preparar test set (enfermedades del café)
test_cases = [
    {
        "input": "¿Qué es la roya del café?",
        "expected": "Enfermedad fúngica causada por Hemileia vastatrix"
    },
    {
        "input": "¿Cuál es el ciclo de vida de la Broca?",
        "expected": "35-50 días a temperaturas óptimas"
    },
    # ... más casos
]

# Generar predicciones
def evaluate_on_test_cases(model, tokenizer, test_cases):
    correct = 0
    for case in test_cases:
        # Generar respuesta
        inputs = tokenizer(case["input"], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=100)
        generated_text = tokenizer.decode(outputs[0])
        
        # Verificar si contiene keywords esperadas
        if any(word in generated_text.lower() for word in case["expected"].split()):
            correct += 1
    
    accuracy = correct / len(test_cases)
    print(f"Accuracy en test cases: {accuracy:.2%}")
    return accuracy
```

**Métrica 3: BLEU Score** (similitud con respuestas esperadas)
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(generated_text, reference_text):
    reference = reference_text.split()
    candidate = generated_text.split()
    
    smoothing_function = SmoothingFunction().method1
    bleu = sentence_bleu(
        [reference],
        candidate,
        smoothing_function=smoothing_function
    )
    return bleu

# Calcular BLEU promedio
bleu_scores = []
for case in test_cases:
    inputs = tokenizer(case["input"], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)
    generated = tokenizer.decode(outputs[0])
    
    bleu = calculate_bleu(generated, case["expected"])
    bleu_scores.append(bleu)

average_bleu = np.mean(bleu_scores)
print(f"BLEU Score promedio: {average_bleu:.4f}")
# BLEU > 0.4 es bueno para respuestas generativas
```

### 6.2 Evaluación Manual

**Test 1: Diagnóstico Visual**
```python
# Cargar imagen
from PIL import Image
image = Image.open("./dataset/coffee_diseases_images/roya/roya_001.jpg")

# Preparar input multimodal
prompt = "¿Qué enfermedad tiene esta hoja?"

# Generar respuesta
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0])

print(f"Pregunta: {prompt}")
print(f"Respuesta generada:")
print(response)

# Verificar manualmente si:
# ✓ Identifica enfermedad correcta
# ✓ Da síntomas específicos
# ✓ Proporciona tratamientos
```

**Test 2: Consulta de BD**
```python
test_queries = [
    "¿Cuánto fertilizante tengo?",
    "¿Cuál es mi gasto total este mes?",
    "¿Qué enfermedades he reportado?"
]

for query in test_queries:
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0])
    
    # Verificar si devuelve JSON para function calling
    print(f"Query: {query}")
    print(f"Response: {response}")
    print("---")
```

---

## Cuantización e Inferencia

### 7.1 Guardar el Modelo Fine-tuneado

```python
# Guardar LoRA adapters
model.save_pretrained("./models/gemma3n-coffee-lora")
tokenizer.save_pretrained("./models/gemma3n-coffee-lora")

# Guardar también en HuggingFace (opcional)
model.push_to_hub("tu-usuario/gemma3n-coffee-lora")
```

### 7.2 Cuantizar para Inferencia

**Objetivo**: Reducir tamaño de 12 GB → 3.2 GB para el servidor

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig

# Opción 1: Cuantización a 4-bit (gqllm_Quantization)
bnb_config_inference = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_quantized = AutoModelForCausalLM.from_pretrained(
    "./models/gemma3n-coffee-lora",
    quantization_config=bnb_config_inference,
    device_map="auto"
)

# Opción 2: Cuantización GPTQ (más eficiente)
gptq_config = GPTQConfig(bits=4, dataset="wikitext")
model_quantized = AutoModelForCausalLM.from_pretrained(
    "./models/gemma3n-coffee-lora",
    quantization_config=gptq_config,
    device_map="auto"
)

# Guardar versión cuantizada
model_quantized.save_pretrained("./models/gemma3n-coffee-int4")
```

### 7.3 Cargar en Inferencia (api.py)

```python
# app/api.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

# Configuración de cuantización
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Cargar modelo fine-tuneado
model_id = "./models/gemma3n-coffee-int4"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Crear pipeline para inferencia
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.post("/ask")
async def ask(question: str):
    # Generar respuesta
    output = pipe(question, max_new_tokens=200, temperature=0.7)
    return {"answer": output[0]["generated_text"]}
```

---

## Troubleshooting

### P1: "CUDA out of memory"

**Síntoma**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU has only 0.50 GiB available.
```

**Soluciones** (en orden de intento):
```python
# 1. Reducir batch size
per_device_train_batch_size = 2  # en vez de 4

# 2. Aumentar gradient accumulation
gradient_accumulation_steps = 8  # en vez de 4

# 3. Usar cuantización 4-bit (hecho en Celda 2)

# 4. Reducir max_length
max_length=512  # en vez de 1024

# 5. Usar LoRA (hecho en Celda 4)

# 6. Limpiar caché GPU
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

### P2: "Loss no baja" (modelo no aprende)

**Síntoma**:
```
Step 10:  Loss: 3.45
Step 20:  Loss: 3.40
Step 50:  Loss: 3.42  ← No mejora
```

**Causas y soluciones**:
```python
# Causa 1: Learning rate muy alta
# Solución:
learning_rate = 5e-5  # en vez de 1e-4

# Causa 2: Datos inconsistentes o malos
# Solución:
# - Validar manualmente 10 ejemplos
# - Verificar tokenización
# - Buscar duplicados

# Causa 3: Modelo no es apropiado
# Solución:
# - Usar modelo más grande (Gemma-7B)
# - O usar modelo más establecido (LLaMA 2)
```

### P3: "Loss baja bien pero accuracy es baja"

**Síntoma**:
```
Loss: 0.95  (excelente)
Pero respuestas siguen siendo genéricas
```

**Causa**: Overfitting a los datos de entrenamiento, pero no generaliza

**Soluciones**:
```python
# 1. Aumentar regularización
lora_dropout = 0.15  # en vez de 0.05

# 2. Agregar más datos variados
# Recolectar 5,000 ejemplos en vez de 1,200

# 3. Usar warmup más largo
warmup_steps = 500  # en vez de 100

# 4. Early stopping
from transformers import EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

### P4: "Modelo genera respuestas muy largas/cortas"

**Síntoma**:
```
Pregunta: ¿Qué es la roya?
Respuesta: La roya es una enfermedad causada por el hongo 
           Hemileia vastatrix... [500 palabras más]
```

**Solución**:
```python
# En api.py, ajustar max_new_tokens
output = model.generate(
    **inputs,
    max_new_tokens=200,  # Máximo 200 tokens (~150 palabras)
    min_length=50,       # Mínimo 50 tokens
    temperature=0.7,
    top_p=0.9
)
```

### P5: "Respuestas son diferentes cada vez"

**Síntoma**:
```
Pregunta: ¿Qué es la roya?
Respuesta 1: La roya es Hemileia vastatrix...
Respuesta 2: Hemileia vastatrix causa la roya...
Respuesta 3: Una enfermedad del café es la roya...
```

**Solución**:
```python
# Fijar seed para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# Usar temperature baja (más determinístico)
temperature = 0.3  # en vez de 0.7

# Usar top_k en vez de top_p
top_k = 50
top_p = 0.9
```

---

## Conclusión

El fine-tuning es un proceso científico pero iterativo:

1. **Preparación de datos** (50% del tiempo)
2. **Experimentación** (30% del tiempo)
3. **Evaluación y ajustes** (20% del tiempo)

**Checklist final**:
- [ ] Datos preparados y validados (1,200+ ejemplos)
- [ ] Ambiente configurado (dependencias instaladas)
- [ ] Model checkpoint guardado (./outputs/)
- [ ] Métricas evaluadas (accuracy > 90%)
- [ ] Modelo cuantizado (3.2 GB)
- [ ] Integrado en api.py para inferencia
- [ ] Testeado en casos reales
- [ ] Documentado para reproducibilidad

**Próximo paso**: Ver PRESENTACION_TECNICA.md para entender cómo se integra en la arquitectura completa.

