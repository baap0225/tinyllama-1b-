# TinyLlama Finance QLoRA Fine-tuned Model

A fine-tuned TinyLlama-1.1B model specialized for financial questions and advice, trained using QLoRA (Quantized Low-Rank Adaptation) for efficient training on consumer GPUs.

## Model Details

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Training Method**: QLoRA (4-bit quantization + LoRA)
- **Dataset**: Finance-Instruct-500k (5000 samples)
- **Training Hardware**: RTX 3050 (4GB VRAM)
- **Parameters**: ~500K trainable parameters

## Quick Start

### Option 1: Automated Setup
**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Test the Model
```bash
python test_model.py
```

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load model
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./tinyllama-finance-adapter-v2")

# Ask a financial question
question = "What is compound interest?"
prompt = f"<|system|>\nYou are a financial expert.</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("<|assistant|>")[-1].strip())
```

## Training Details

- **Batch Size**: 1 (with gradient accumulation 8)
- **Learning Rate**: 1e-4
- **Epochs**: 1
- **Sequence Length**: 512 tokens
- **LoRA Rank**: 8
- **Target Modules**: q_proj, v_proj

## Hardware Requirements

- **Minimum VRAM**: 4GB (RTX 3050, GTX 1660 Ti, etc.)
- **Recommended VRAM**: 6GB+ for faster inference
- **RAM**: 8GB+ system RAM

## Files in This Repository

- `finetune_tinyllama_qlora_Version2.py` - Training script
- `test_model.py` - Model testing script
- `requirements.txt` - Python dependencies
- `tinyllama-finance-adapter-v2/` - Fine-tuned LoRA adapter weights

## Performance

The model provides knowledgeable responses on:
- Investment strategies
- Financial risk management
- Compound interest calculations
- Portfolio diversification
- Emergency fund planning
- Cryptocurrency basics
- Mutual funds and ETFs
- Mortgage considerations
- Inflation effects

## License

This model is based on TinyLlama which is licensed under Apache 2.0. The fine-tuned weights follow the same license.

## Citation

If you use this model, please cite:
```
@misc{tinyllama-finance-qlora,
  title={TinyLlama Finance QLoRA Fine-tuned Model},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/tinyllama-finance-qlora}}
}
```
