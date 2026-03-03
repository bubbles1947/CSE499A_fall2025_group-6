# BLIP-2 ONNX Export Pipeline

A project for exporting the BLIP-2 (Bootstrapping Language-Image Pre-training 2) model to ONNX format for efficient inference and deployment.

## Overview

This project provides tools to:
- Download and test the BLIP-2 model from Salesforce
- Export BLIP-2 components (Vision Encoder, Q-Former, Language Decoder) to ONNX format
- Apply INT8 quantization for reduced model size
- Run inference tests with the exported models

## Project Structure

```
CSE499B/
├── export_blip2.py           # Download and test BLIP-2 model
├── export_vision_encoder.py  # Export Vision Encoder to ONNX
├── export_qformer.py         # Export Q-Former to ONNX
├── export_decoder.py         # Export OPT Decoder to ONNX
├── test_pipeline.py          # Test full caption pipeline
├── verify_and_quantize.py    # Verify ONNX models and apply quantization
├── models/                   # Cached HuggingFace models
├── onnx_models/              # Exported ONNX models
└── blip_base_models/         # Base BLIP models and tokenizer files
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- 16GB+ RAM

### Software
```bash
pip install torch torchvision
pip install transformers
pip install onnx onnxruntime
pip install onnxruntime-gpu  # For GPU inference
pip install Pillow requests
```

## Usage

### Step 1: Download and Test BLIP-2 Model

First, download the BLIP-2 model and verify it works:

```bash
python export_blip2.py
```

This will:
- Download the Salesforce/blip2-opt-2.7b model (~6 GB)
- Cache it in the `./models` directory
- Run a test inference on a sample image

### Step 2: Export Vision Encoder

Export the vision encoder component to ONNX:

```bash
python export_vision_encoder.py
```

Output: `onnx_models/blip2_vision_encoder.onnx`

### Step 3: Export Q-Former

Export the Q-Former (Query Transformer) component:

```bash
python export_qformer.py
```

Output: `onnx_models/blip2_qformer.onnx`

### Step 4: Export Language Decoder (Optional)

Export the OPT language decoder:

```bash
python export_decoder.py
```

Output: `onnx_models/opt_decoder.onnx` and `onnx_models/opt_decoder_int8.onnx`

### Step 5: Verify and Quantize

Verify the exported ONNX models and apply INT8 quantization:

```bash
python verify_and_quantize.py
```

This will:
- Validate the ONNX model structure
- Test inference with ONNX Runtime
- Apply INT8 quantization for reduced model size
- Compare original and quantized file sizes

### Step 6: Test Full Pipeline

Test the complete image captioning pipeline:

```bash
python test_pipeline.py
```

You can also place a `test_image.jpg` in the project folder to test with your own image.

## Exported Models

| Model | File | Description |
|-------|------|-------------|
| Vision Encoder | `blip2_vision_encoder.onnx` | Extracts image features (ViT-based) |
| Vision Encoder INT8 | `blip2_vision_encoder_int8.onnx` | Quantized version |
| Q-Former | `blip2_qformer.onnx` | Bridges vision and language |
| Language Decoder | `opt_decoder.onnx` | Generates text captions |

## Model Architecture

BLIP-2 consists of three main components:

1. **Vision Encoder**: A ViT (Vision Transformer) that extracts visual features from images
2. **Q-Former**: A Query Transformer that bridges the gap between vision and language modalities
3. **Language Decoder**: An OPT model that generates text captions based on the visual features

```
Image → [Vision Encoder] → [Q-Former] → [Language Decoder] → Caption
```

## Input/Output Specifications

### Vision Encoder
- **Input**: `pixel_values` - Shape: `(batch_size, 3, 224, 224)`, Type: `float16`
- **Output**: `image_features` - Shape: `(batch_size, 257, 1408)`

### Q-Former
- **Input**: `image_features` - Shape: `(batch_size, 257, 1408)`, Type: `float16`
- **Output**: `language_features` - Shape: `(batch_size, 32, hidden_size)`

### Language Decoder
- **Input**: `input_ids` - Shape: `(batch_size, sequence_length)`, Type: `int64`
- **Output**: `logits` - Shape: `(batch_size, sequence_length, vocab_size)`

## Quantization

INT8 quantization reduces model size significantly:
- Uses `onnxruntime.quantization.quantize_dynamic`
- Typical size reduction: 30-50%
- Minimal accuracy loss for most use cases

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use CPU execution provider for testing
- Close other GPU applications

### ONNX Export Errors
- Ensure opset version is 14 or higher
- Check that all model components are in eval mode
- Verify CUDA is available: `torch.cuda.is_available()`

### Model Loading Issues
- Ensure sufficient disk space for model cache
- Check internet connection for initial download
- Verify HuggingFace cache directory permissions

## References

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [Salesforce BLIP-2 on HuggingFace](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)


