---
license: apache-2.0
language:
- en
tags:
- conversational
- mental-health
- therapy
- genz
- dia
- unsloth
- fine-tuned
- qwen
- chatbot
- hf-inference
datasets:
- anupamaditya/dia-therapy-dataset
pipeline_tag: text-generation
model-index:
- name: dia-convo-v1.2c
  results: []
base_model:
- Qwen/Qwen2.5-7B-Instruct
---

# 🧠 Dia-Convo-v1.2c

`petrioteer/dia-convo-v1.2c` is a conversational mental-health-focused LLM designed for Gen Z, built on top of **Qwen2.5-7B-Instruct** and fine-tuned using [dia-therapy-dataset](https://huggingface.co/datasets/anupamaditya/dia-therapy-dataset). This model powers **Dia-Therapist**, an empathetic AI that offers mental health support while being context-aware, brief, and emotionally intelligent.

---

## 💬 Intended Use

This model is tuned to offer:
- Thoughtful responses to mental health queries
- Conversational tone suited for Gen Z
- Non-medical, non-clinical guidance
- Short, contextually sensitive replies

**It does not replace professional therapy.**

---

## 📚 Training Dataset

- [anupamaditya/dia-therapy-dataset](https://huggingface.co/datasets/anupamaditya/dia-therapy-dataset)
- Contains conversational instructions paired with realistic mental-health-related inputs from Gen Z users.

---

## 🧪 Example Inference (🤗 Transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "petrioteer/dia-convo-v1.2c"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = """
### Instruction:
Your name is Dia, a mental health therapist Assistant Bot. Provide guidance on mental health topics only and avoid others. Don\'t give medical advice. Keep responses short and relevant.

### Input:
I'm feeling overwhelmed with my classes. I can't seem to focus.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.3,
    top_p=0.85,
    top_k=40,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚡ Fast Inference (🧬 Unsloth)

```python
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

model_name = "petrioteer/dia-convo-v1.2c"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto",
)

FastLanguageModel.for_inference(model)

prompt = """
### Instruction:
Your name is Dia, a mental health therapist Assistant Bot. Provide guidance on mental health topics only and avoid others. Don\'t give medical advice. Keep responses short and relevant.

### Input:
I just feel numb and disconnected from everyone lately.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.3,
    top_p=0.85,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📍 Model Details

- 🔗 Base model: Qwen2.5-7B-Instruct
- 🧠 Fine-tuned using dia-therapy-dataset on Gen Z mental health patterns
- 🛠️ Quantized with 4-bit support (for faster loading)
- 🧪 Best used with Unsloth for optimized inference

---

## ❤️ Citation & Thanks

If you use Dia-Convo in research, demos, or builds, consider citing or linking back to this repo and dataset authors.

---

 
Built with ❤️ & care by **Itesh (aka petrioteer)** ✨
