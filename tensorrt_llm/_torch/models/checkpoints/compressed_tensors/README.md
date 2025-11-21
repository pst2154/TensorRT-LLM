# Compressed Tensors Support

This module provides support for loading models in the `compressed-tensors` format from [neuralmagic](https://github.com/neuralmagic/compressed-tensors).

## Overview

The compressed-tensors format extends SafeTensors to efficiently store sparse and quantized tensors. It supports various compression schemes including:

- **GPTQ** - Group-wise quantization
- **AWQ** - Activation-aware weight quantization  
- **SparseGPT** - Structured sparsity/pruning
- **INT4/INT8 quantization** - Weight-only and activation quantization
- **FP8 quantization** - Float8 quantization

## Components

### CompressedTensorsWeightLoader
Loads and processes weights from compressed-tensors format. Handles decompression and conversion to formats compatible with TRT-LLM's Linear layers.

### CompressedTensorsConfigLoader
Loads model configuration. Since compressed-tensors uses standard HF configs, this delegates to the standard config loading.

### CompressedTensorsWeightMapper
Maps checkpoint weights to model parameters, handling compressed tensor metadata like quantization scales, zero-points, and sparsity information.

### CompressedTensorsConfigReader
Parses quantization configuration from `config.json` and maps it to TRT-LLM's quantization formats.

## Usage

```python
from tensorrt_llm import LLM

# Load a model quantized with compressed-tensors format
# (e.g., from neuralmagic's sparse models)
llm = LLM(model='neuralmagic/Llama-2-7b-pruned50-quantized-gptq')
output = llm.generate("Hello, my name is")
print(output)
```

## Installation

To use compressed-tensors format, install the library:

```bash
pip install compressed-tensors
```

If the library is not installed, the loader will fall back to standard safetensors loading, which works for many models but may not handle all compression formats optimally.

## Architecture

The implementation follows TRT-LLM's modular checkpoint loading architecture:

1. **Config Detection**: `CompressedTensorsConfigReader` detects if a checkpoint uses compressed-tensors format by checking for `"quant_method": "compressed-tensors"` in `config.json`.

2. **Weight Loading**: `CompressedTensorsWeightLoader` uses the `compressed-tensors` library to load weights, handling decompression automatically.

3. **Weight Mapping**: `CompressedTensorsWeightMapper` extracts quantization metadata (scales, zero-points) and maps them to the format expected by TRT-LLM's quantized Linear layers.

4. **Integration**: The existing quantization infrastructure (W4A16_AWQ, W4A16_GPTQ, etc.) is reused for inference.

## Supported Models

The compressed-tensors format is model-agnostic and supports any HuggingFace model that has been quantized/sparsified with the format. Popular examples include:

- Llama 2/3 (pruned and quantized variants)
- Mistral (quantized variants)
- Qwen (quantized variants)
- And any model on HuggingFace with `quantization_config.quant_method = "compressed-tensors"`

## Format Details

The compressed-tensors format stores weights in SafeTensors with additional metadata:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",  // or "float-quantized", "pack-quantized", etc.
    "config_groups": {
      "group_0": {
        "num_bits": 4,
        "group_size": 128,
        "targets": ["weight"]
      }
    }
  }
}
```

Quantized weights are stored with separate tensors for:
- `weight` - The quantized weight values
- `scales` - Per-group or per-channel scales
- `zero_points` - Zero-point offsets (for asymmetric quantization)

## Extending

To add support for new compression formats:

1. Update `CompressedTensorsConfigReader._map_*` methods to handle the new format
2. Update `CompressedTensorsWeightMapper` to extract any new metadata
3. Ensure the corresponding quantization method is available in `tensorrt_llm.quantization`

