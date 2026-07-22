# SDNQ + OpenVINO Chat & Quantizer

## Overview


## SDNQ has multiple execution paths: 

| Device    | Backend           |
| --------- | ----------------- |
| NVIDIA    | Inductor + Triton |
| AMD ROCm  | Inductor          |
| Intel XPU | Inductor          |
| Apple MPS | Eager fallback    |
| CPU       | Eager fallback    |
| ARM       | Eager fallback    |
| Android   | Eager fallback    |

## If a specialized backend isn't available, SDNQ falls back to regular PyTorch eager execution. That's what the README means by "PyTorch Eager fallback mode".

## Conceptually:
```

SDNQ Tensor
      │
      ▼
torch.compile()
      │
 ┌────┴────┐
 │         │
Inductor  Eager
 │         │
Fast      Works everywhere
```

## HOw to check SDNQ Support
```
import torch

print("torch:", torch.__version__)

try:
    print(
        "compile backend:",
        torch.compiler.get_default_backend()
    )
except:
    pass
```

This project combines two capabilities into a single Gradio application:
1. **SDNQ Chat Interface**

   * Load Hugging Face language models
   * Apply SDNQ post-load quantization
   * Run inference using CPU, GPU, or OpenVINO-enabled execution

2. **SDNQ Quantization Pipeline**

   * Download any Hugging Face Transformers model
   * Apply SDNQ quantization
   * Save quantized weights
   * Automatically upload the result to Hugging Face Hub

---

# Architecture

```text
┌─────────────────────────────┐
│         Gradio UI           │
└──────────────┬──────────────┘
               │
     ┌─────────┴─────────┐
     │                   │
     ▼                   ▼

┌───────────────┐   ┌────────────────┐
│   Chat Tab    │   │   Quant Tab    │
└───────┬───────┘   └────────┬───────┘
        │                    │
        ▼                    ▼

 Load Model           Download Model
 Apply SDNQ           Apply SDNQ
 Generate Text        Save Model
                      Upload to HF
```

---

# Features

## Chat Interface

Supported Features:

* Hugging Face Transformers models
* SDNQ quantization
* Streaming generation
* OpenVINO acceleration
* CUDA acceleration
* Apple MPS support
* Adjustable temperature
* Adjustable token count
* System prompts

---

## Quantization Interface

Supported Features:

### Model Source

Any Hugging Face model:

```text
google/gemma-2-2b-it
google/gemma-3-4b-it
Qwen/Qwen3-4B
meta-llama/Llama-3.2-3B-Instruct
microsoft/Phi-4-mini-instruct
```

### Output

Creates a quantized repository:

```text
rahul7star/gemma-2-2b-it-sdnq-uint4
```

or user-defined:

```text
rahul7star/my-custom-quant
```

---

# SDNQ Quantization Options

## Weight Types

Supported:

```text
uint2
int2

uint3
int3

uint4
int4

uint5
int5

uint6
int6

uint8
int8

float8_e4m3fn

float16
```

---

## Group Size

Controls quantization granularity.

Typical values:

| Group Size | Usage                  |
| ---------- | ---------------------- |
| 32         | Default                |
| 64         | Better compression     |
| 128        | Aggressive compression |
| -1         | Per-tensor             |

---

## Hadamard Rotation

Optional preprocessing step before quantization.

Benefits:

* Better weight distribution
* Reduced quantization error
* Improved low-bit quality

Recommended:

```text
Enabled for int4/int8
```

---

## SVD Quantization

Optional low-rank decomposition.

Benefits:

* Smaller model size
* Potential quality retention

Tradeoff:

```text
Longer quantization time
```

---

## Quantized MatMul

Enables quantized matrix multiplication.

Benefits:

* Faster inference
* Lower memory usage

Options:

```text
int8
float16
disabled
```

---

# Presets

## Fast

Designed for maximum compression.

```text
weights_dtype = uint4
group_size = 32
```

Best for:

* CPU inference
* Low RAM systems

---

## Balanced

Default recommendation.

```text
weights_dtype = int8
group_size = 32
```

Best for:

* General use
* Quality/performance balance

---

## Quality

Prioritizes model quality.

```text
weights_dtype = float16
group_size = -1
```

Best for:

* High-end hardware
* Benchmarking

---

# Model Loading Options

Supported dtypes:

```text
auto
float32
float16
bfloat16
```

Recommendation:

| Model Family | Recommended |
| ------------ | ----------- |
| Gemma        | bfloat16    |
| Qwen         | bfloat16    |
| Phi          | float16     |
| Llama        | float16     |

---

# Hugging Face Integration

Authentication is handled via Space Secrets.

Required Secret:

```text
HF_TOKEN
```

Example:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

The token is never exposed in the UI.

---

# Quantization Workflow

## Step 1

User enters model:

```text
google/gemma-2-2b-it
```

## Step 2

Application downloads model:

```python
snapshot_download()
```

## Step 3

Tokenizer loads:

```python
AutoTokenizer.from_pretrained()
```

## Step 4

Model loads:

```python
AutoModelForCausalLM.from_pretrained()
```

## Step 5

SDNQ applies quantization:

```python
sdnq_post_load_quant()
```

## Step 6

Quantized model saved:

```python
save_pretrained()
```

## Step 7

Repository created:

```python
HfApi.create_repo()
```

## Step 8

Files uploaded:

```python
HfApi.upload_folder()
```

---

# Memory Management

After saving:

```python
del model
gc.collect()
```

Benefits:

* Lower RAM usage
* Reduced Space crashes
* Improved reliability

---

# Supported Hardware

## CPU

Fully supported.

Recommended:

```text
uint4
int4
int8
```

---

## NVIDIA GPU

Supported through PyTorch CUDA.

Benefits:

* Faster loading
* Faster quantization

---

## Apple Silicon

Supported through MPS.

Examples:

```text
M1
M2
M3
M4
```

---

## OpenVINO

Supported devices:

```text
CPU
GPU
NPU
```

Detected automatically:

```python
openvino.Core().available_devices
```

---

# Future Enhancements

Planned improvements:

* Batch quantization
* GGUF export
* AWQ export
* GPTQ export
* ONNX export
* OpenVINO IR export
* Quantization benchmarks
* Model size comparison
* Performance comparison charts
* Quantization history dashboard

---

# Repository Structure

```text
app_openvino.py

requirements.txt

README.md

DESIGN.md

models/

outputs/

logs/
```

---

# Goal

Provide a simple web interface for:

* Running SDNQ models
* Exploring quantization settings
* Uploading quantized models to Hugging Face
* Experimenting with OpenVINO acceleration

without requiring command-line tools or custom scripts.
