# Gemma-3N Fine-tuning Guide for CaficulBot

## Execution Environment

### Recommended Platforms

**Google Colab Pro/Pro+ (Recommended)**
- GPU: L4 (24GB VRAM) or A100 (40GB VRAM)
- Runtime: 6-8 hours on L4, 4-5 hours on A100
- Cost: $10-50/month subscription
- Access: colab.research.google.com

**Kaggle Notebooks**
- GPU: P100 (16GB VRAM) or T4 (16GB VRAM)
- Runtime: 8-10 hours (may require multiple sessions)
- Cost: Free tier available
- Note: Free tier has session limits, may timeout on long training

**Lambda Labs / RunPod (Cloud GPU Rental)**
- GPU: A100, A6000, or RTX 4090
- Runtime: 4-8 hours depending on GPU
- Cost: $0.50-$1.50/hour
- Flexible: Pay only for usage

**Local Machine**
- GPU: NVIDIA RTX 4090, A6000, or equivalent (16GB+ VRAM)
- Runtime: 6-10 hours
- Requirements: CUDA 12.1+, 32GB+ system RAM

### Hardware Requirements

- Minimum VRAM: 14GB (with 4-bit quantization and gradient checkpointing)
- Recommended VRAM: 24GB for comfortable training
- System RAM: 32GB+
- Disk space: 50GB+ for datasets and checkpoints

---

## Dataset Sources

### 1. Text QA Pairs (5,100 examples)

**HuggingFace Dataset:** `sergioq2/coffe`
**URL:** https://huggingface.co/datasets/sergioq2/coffe

Generated from 1,000+ CENICAFE technical documents covering:
- Coffee cultivation techniques
- Disease identification and treatment
- Pest management strategies
- Soil and fertilization practices
- Harvest and post-harvest processing

Format:
```json
{
  "preguntas": "¿Cómo controlar la roya del café?",
  "respuestas": "Para el control de la roya se recomienda..."
}
```

### 2. Function Calling Dataset (2,700 examples)

**HuggingFace Dataset:** `sergioq2/functioncalling_coffedata`
**URL:** https://huggingface.co/datasets/sergioq2/functioncalling_coffedata

Teaches model when and how to call external tools:
- Inventory queries
- Expense tracking
- Harvest records
- Income management

Format:
```json
{
  "query": "¿Cuánto fertilizante tenemos?",
  "function": "{\"tool\": \"inventario_consulta\", \"argumentos\": \"producto=fertilizante\"}"
}
```

### 3. Disease Images (2,616 base images)

**Roboflow Dataset:** `detection-3nbwx/coffe-mw9n0`
**Access:** Requires Roboflow API key

Disease classes:
- Roya (Coffee Leaf Rust)
- Broca (Coffee Berry Borer)
- Mancha de hierro (Iron spot)
- Ojo de gallo (American leaf spot)
- Mal Rosado (Pink disease)

Images undergo data augmentation:
- Brightness adjustment: ±5%
- Contrast adjustment: ±5%
- Rotation: ±3 degrees
- Gaussian blur: radius 0.1-0.3
- Color saturation: ±2%
- Sharpness: ±2%

Augmentation multipliers:
- Roya and Broca (most critical): 4x
- Other diseases: 2x

---

## Notebook Structure

### Section 1: Environment Setup (Cells 0-4)

**Cell 0:** Install dependencies
- Unsloth framework for efficient fine-tuning
- Transformers with Gemma-3N support
- PEFT, TRL, bitsandbytes for LoRA and quantization

**Cells 1-4:** Import libraries and create dummy images
- Gemma-3N requires image input even for text-only examples
- Black 224x224 placeholder images used for text QA pairs

### Section 2: Text Dataset Processing (Cells 5-15)

**Cells 5-6:** Load text QA dataset from HuggingFace

**Cells 7-10:** Transform to conversational format
- Convert question-answer pairs to dialogue structure
- Standardize to Unsloth's expected schema

**Cells 11-12:** Convert to multimodal messages format
- Add content type annotations (text/image)
- Prepare for Gemma-3N's input structure

**Cells 13-15:** Add dummy images to text examples
- Required by multimodal model architecture
- Allows unified training across text and vision data

### Section 3: Function Calling Dataset (Cells 16-26)

**Cells 17-19:** Load function calling dataset

**Cells 20-23:** Transform and standardize
- Convert tool calls to conversational format
- Maintain JSON structure in assistant responses

**Cells 24-26:** Convert to messages and add dummy images

### Section 4: Image Dataset Processing (Cells 27-40)

**Cell 27:** Install Roboflow client

**Cells 29-30:** Download disease image dataset
- Requires ROBOFLOW_API_KEY environment variable
- Downloads to local directory for processing

**Cell 31:** Load and label images
- Map disease class annotations to Spanish descriptions
- Create HuggingFace Dataset with image features

**Cell 32:** Data augmentation
- Balance classes through intelligent augmentation
- Apply random transformations while preserving disease features
- Critical diseases (Roya, Broca) receive more augmentation

**Cells 33-34:** Convert images to messages format
- Add standardized question: "¿Qué enfermedad tiene esta planta de café?"
- Pair with disease identification text

**Cells 35-40:** Format assistant responses and combine datasets
- Ensure consistent structure across all three data sources
- Shuffle combined dataset for better training

### Section 5: Model Loading and LoRA Configuration (Cells 41-43)

**Cell 41:** Load Gemma-3N base model
- Model: `unsloth/gemma-3n-E2B-it` (instruction-tuned variant)
- 4-bit quantization enabled for memory efficiency
- Gradient checkpointing enabled

**Cell 42:** Apply LoRA (Low-Rank Adaptation)
- Fine-tune both vision and language layers
- LoRA rank (r): 32
- LoRA alpha: 64
- Dropout: 0.03
- Target modules: all linear layers
- Trainable parameters: ~0.8% of total model

**Cell 43:** Disable PyTorch compilation
- Prevents compatibility issues with Unsloth
- Ensures stable training

### Section 6: Training Configuration and Execution (Cells 44-45)

**Cell 44:** Configure SFTTrainer
- Batch size: 4 per device
- Gradient accumulation: 8 steps (effective batch size: 32)
- Learning rate: 2e-4 with cosine scheduler
- Warmup ratio: 5%
- Training epochs: 3
- Weight decay: 0.05
- Optimizer: AdamW fused (faster on CUDA)

**Cell 45:** Execute training
- Trains on combined dataset of ~10,400 examples
- Estimated 6-8 hours on L4 GPU
- Checkpoints saved to `outputs/` directory

### Section 7: Model Export (Cells 46-47)

**Cell 46:** Load HuggingFace Hub token

**Cell 47:** Push fine-tuned model to HuggingFace
- Merges LoRA adapters with base model
- Uploads to `sergioq2/gemma-3N-finetune-coffe_q4_off`
- Requires write access to repository

---

## Training Hyperparameters Explained

### Batch Size and Gradient Accumulation

```python
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
# Effective batch size: 4 × 8 = 32
```

**Why small per-device batch?** Limited VRAM with 6B parameter model
**Why gradient accumulation?** Simulate larger batch without OOM errors
**Effective batch size 32:** Good balance for stable training

### Learning Rate

```python
learning_rate = 2e-4  # 0.0002
lr_scheduler_type = "cosine"
warmup_ratio = 0.05  # 5% of steps for warmup
```

**2e-4:** Standard for LoRA fine-tuning (higher than full fine-tuning)
**Cosine schedule:** Gradually decreases learning rate
**Warmup:** Prevents instability in early training

### LoRA Configuration

```python
r = 32              # Rank of adaptation matrices
lora_alpha = 64     # Scaling factor (typically 2×r)
lora_dropout = 0.03
```

**Rank 32:** Higher than typical (8-16) for complex multimodal task
**Alpha 64:** Controls magnitude of LoRA updates
**Dropout 0.03:** Minimal regularization for stability

### Gradient Clipping

```python
max_grad_norm = 0.3
```

Prevents exploding gradients during training

### Weight Decay

```python
weight_decay = 0.05
```

L2 regularization to prevent overfitting

---

## Expected Training Metrics

### Loss Progression

- Initial loss: ~2.5-3.0
- Final loss: ~0.8-1.2
- Should decrease consistently across epochs

### VRAM Usage

- Model loading: ~8-10GB
- Peak training: ~14-16GB
- With gradient checkpointing: ~12-14GB

### Checkpoints

Saved every N steps to `outputs/`:
- `checkpoint-100/`
- `checkpoint-200/`
- etc.

Each checkpoint: ~4-5GB

---

## Post-Training Validation

After training completes, validate on held-out examples:

1. Disease identification accuracy
2. Function calling precision
3. Text response coherence
4. No hallucinations on out-of-domain images

Expected performance:
- Disease detection: 85-90% accuracy
- Function calling: 90-95% precision
- Text QA: BLEU score 0.6-0.7

---

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size` to 2
- Increase `gradient_accumulation_steps` to 16
- Enable more aggressive gradient checkpointing

### Slow Training

- Verify GPU is being used: check `nvidia-smi`
- Ensure CUDA version matches PyTorch
- Disable mixed precision if unstable

### Poor Convergence

- Increase learning rate to 3e-4
- Reduce weight decay to 0.01
- Train for more epochs (4-5)

### Dataset Loading Errors

- Check HuggingFace authentication
- Verify dataset names and splits
- Ensure Roboflow API key is valid

---

## Using the Fine-tuned Model

After training, the model is available at:
```
https://huggingface.co/sergioq2/gemma-3N-finetune-coffe_q4_off
```

Download and use in production:

```python
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="sergioq2/gemma-3N-finetune-coffe_q4_off",
    device="cuda",
    torch_dtype=torch.bfloat16
)

# Text query
output = pipe(
    text=[{"role": "user", "content": [{"type": "text", "text": "¿Cómo controlar la roya?"}]}],
    max_new_tokens=200
)

# Image query
output = pipe(
    text=[{"role": "user", "content": [
        {"type": "text", "text": "¿Qué enfermedad tiene?"},
        {"type": "image"}
    ]}],
    images=pil_image,
    max_new_tokens=200
)
```

---

## Cost Estimation

### Google Colab Pro+
- L4 GPU: ~$50/month unlimited
- Training time: 6-8 hours
- **Total: $50 one-time** (plus subscription)

### Lambda Labs
- A100 (40GB): $1.10/hour
- Training time: 4-5 hours
- **Total: $4-6 per training run**

### Kaggle (Free)
- P100 GPU: Free
- May require 2-3 sessions due to timeout
- **Total: $0** (time investment higher)

---

## References

- Unsloth documentation: https://github.com/unslothai/unsloth
- Gemma model card: https://huggingface.co/google/gemma-3n-E2B-it
- LoRA paper: https://arxiv.org/abs/2106.09685
- CENICAFE: https://www.cenicafe.org/
