# Compressed Tensors Support

TensorRT-LLM now supports loading models in the `compressed-tensors` format from [NeuralMagic](https://github.com/neuralmagic/compressed-tensors).

## Overview

The compressed-tensors format is an extension to SafeTensors designed for efficient storage and loading of sparse and quantized models. It supports various compression schemes including GPTQ, AWQ, SparseGPT, and various INT/FP quantization methods.

## Key Features

- **Multiple Quantization Formats**: GPTQ, AWQ, INT4/INT8, FP8
- **Sparsity Support**: Structured pruning and sparse weights
- **Automatic Detection**: Automatically detects compressed-tensors format from checkpoint metadata
- **Fallback Support**: Gracefully falls back to standard SafeTensors loading if compressed-tensors library is unavailable
- **PyTorch Backend Integration**: Seamlessly integrates with TensorRT-LLM's PyTorch backend

## Installation

To use compressed-tensors format with optimal performance, install the optional dependency:

```bash
pip install tensorrt_llm[compressed-tensors]
```

Or install the library directly:

```bash
pip install compressed-tensors
```

**Note**: If the `compressed-tensors` library is not installed, TensorRT-LLM will automatically fall back to standard SafeTensors loading, which works for most models but may not handle all compression formats optimally.

## Usage

### Basic Usage

Loading a compressed-tensors model is identical to loading any other model:

```python
from tensorrt_llm import LLM

# Load a model quantized with compressed-tensors format
llm = LLM(model='neuralmagic/Llama-2-7b-pruned50-quantized-gptq')
outputs = llm.generate("Hello, my name is")
```

### Example Models

Popular compressed-tensors models from NeuralMagic:

```python
# Sparse + Quantized Llama-2
llm = LLM(model='neuralmagic/Llama-2-7b-pruned50-quantized-gptq')

# Quantized Mistral
llm = LLM(model='neuralmagic/Mistral-7B-Instruct-v0.2-GPTQ-4bit')

# Any HuggingFace model with compressed-tensors format
llm = LLM(model='path/to/compressed-tensors-model')
```

## Supported Compression Formats

### Weight-Only Quantization

| Format | Description | TRT-LLM Mapping |
|--------|-------------|-----------------|
| **INT4 GPTQ** | 4-bit group-wise quantization | `W4A16_GPTQ` |
| **INT4 AWQ** | Activation-aware weight quantization | `W4A16_AWQ` |
| **INT8** | 8-bit weight quantization | `W8A16` |

### Float Quantization

| Format | Description | TRT-LLM Mapping |
|--------|-------------|-----------------|
| **FP8** | 8-bit floating point quantization | `FP8` |

### Sparsity

| Format | Description |
|--------|-------------|
| **Structured Sparsity** | Pruned weights with sparse patterns |
| **Unstructured Sparsity** | Arbitrary zero patterns |

## Architecture

The implementation consists of four main components:

### 1. CompressedTensorsConfigReader

Parses `quantization_config` from `config.json` to detect compressed-tensors format:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
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

### 2. CompressedTensorsWeightLoader

Loads weights from SafeTensors files using the `compressed-tensors` library for decompression:

- Detects compression format from config
- Uses `compressed-tensors` library when available
- Falls back to standard SafeTensors loading
- Extracts quantization metadata (scales, zero-points)

### 3. CompressedTensorsWeightMapper

Maps compressed weights to model parameters:

- Extracts quantization scales and zero-points
- Handles sparse weight metadata
- Converts to format expected by TRT-LLM's Linear layers

### 4. CompressedTensorsConfigLoader

Loads model configuration (delegates to standard HF config loading since compressed-tensors uses standard HF configs).

## Format Detection

TensorRT-LLM automatically detects compressed-tensors format using the following priority:

1. **Compressed-Tensors**: Checks for `"quant_method": "compressed-tensors"` in config.json
2. **ModelOpt**: Checks for ModelOpt quantization config
3. **HuggingFace**: Falls back to standard HF quantization config

## Integration with Existing Quantization

Compressed-tensors is a **storage format**, not a runtime format. Once loaded, weights are converted to TRT-LLM's native quantization formats:

- **GPTQ** → Uses existing `W4A16_GPTQ` kernels
- **AWQ** → Uses existing `W4A16_AWQ` kernels
- **FP8** → Uses existing FP8 kernels

This means you get the same runtime performance as native TRT-LLM quantized models!

## Advanced Usage

### Custom Weight Loading

For advanced use cases, you can customize weight loading:

```python
from tensorrt_llm._torch.models.checkpoints.compressed_tensors import (
    CompressedTensorsWeightLoader,
    CompressedTensorsConfigLoader
)

# Custom loading logic
loader = CompressedTensorsWeightLoader()
weights = loader.load_weights("/path/to/checkpoint")
```

### Format Conversion

To convert a compressed-tensors model to TRT-LLM's native format:

```python
from tensorrt_llm import LLM

# Load and export
llm = LLM(model='neuralmagic/model')
# The model is now ready to use with TRT-LLM's optimizations
```

## Troubleshooting

### Missing compressed-tensors Library

If you see this warning:

```
compressed-tensors library not found. Will attempt to load weights as standard safetensors.
Install with: pip install compressed-tensors
```

**Solution**: Install the library:
```bash
pip install compressed-tensors
```

Or install with TensorRT-LLM:
```bash
pip install tensorrt_llm[compressed-tensors]
```

### Unsupported Format

If you encounter an unsupported format error, the compressed-tensors model may use a newer format version. Please:

1. Update `compressed-tensors` library: `pip install -U compressed-tensors`
2. Update TensorRT-LLM to the latest version
3. File an issue on the TensorRT-LLM GitHub repository

### Performance Issues

If you experience slower loading:

- Ensure `compressed-tensors` library is installed
- Check that you're using the latest version
- Verify your checkpoint is properly formatted

## Contributing

To add support for new compression formats:

1. Update `CompressedTensorsConfigReader._map_*` methods in `quant_config_reader.py`
2. Update weight processing in `CompressedTensorsWeightMapper`
3. Ensure corresponding quantization kernels exist in TensorRT-LLM
4. Add tests for the new format

## References

- [Compressed-Tensors GitHub](https://github.com/neuralmagic/compressed-tensors)
- [NeuralMagic Model Zoo](https://huggingface.co/neuralmagic)
- [TensorRT-LLM Quantization Guide](./quantization.md)

