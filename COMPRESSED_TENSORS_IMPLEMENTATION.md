# Compressed Tensors Implementation Summary

This document provides a technical overview of the compressed-tensors support implementation in TensorRT-LLM.

## Implementation Overview

Compressed-tensors support has been added to the TensorRT-LLM PyTorch backend to enable loading of models quantized and compressed using the NeuralMagic compressed-tensors format.

## Changes Made

### 1. Core Components Created

#### a. `CompressedTensorsConfigReader`
- **File**: `tensorrt_llm/_torch/auto_deploy/models/quant_config_reader.py`
- **Purpose**: Parse quantization configuration from compressed-tensors format
- **Key Features**:
  - Detects `"quant_method": "compressed-tensors"` in config.json
  - Maps compression formats (int-quantized, float-quantized, pack-quantized) to TRT-LLM QuantAlgo
  - Extracts quantization parameters (num_bits, group_size, etc.)
  - Registered in `QuantConfigReaderRegistry` with key "compressed_tensors"

#### b. `CompressedTensorsWeightLoader`
- **File**: `tensorrt_llm/_torch/models/checkpoints/compressed_tensors/weight_loader.py`
- **Purpose**: Load and decompress weights from compressed-tensors format
- **Key Features**:
  - Uses `compressed-tensors` library for optimal loading
  - Falls back to standard SafeTensors if library unavailable
  - Extracts quantization metadata (scales, zero-points, sparsity)
  - Registered with decorator `@register_checkpoint_weight_loader("COMPRESSED_TENSORS")`

#### c. `CompressedTensorsWeightMapper`
- **File**: `tensorrt_llm/_torch/models/checkpoints/compressed_tensors/weight_mapper.py`
- **Purpose**: Map compressed weights to model parameters
- **Key Features**:
  - Extends `HfWeightMapper` to reuse HF checkpoint mapping logic
  - Detects and extracts compressed tensor metadata
  - Converts to format expected by TRT-LLM Linear layers
  - Registered with decorator `@register_mapper("COMPRESSED_TENSORS")`

#### d. `CompressedTensorsConfigLoader`
- **File**: `tensorrt_llm/_torch/models/checkpoints/compressed_tensors/config_loader.py`
- **Purpose**: Load model configuration
- **Key Features**:
  - Delegates to standard HF config loading (compressed-tensors uses HF configs)
  - Registered with decorator `@register_config_loader("COMPRESSED_TENSORS")`

### 2. Module Structure

```
tensorrt_llm/_torch/models/checkpoints/compressed_tensors/
├── __init__.py                 # Module exports
├── config_loader.py            # Config loading
├── weight_loader.py            # Weight loading and decompression
├── weight_mapper.py            # Weight mapping
└── README.md                   # Module documentation
```

### 3. Integration Changes

#### a. Updated `autodetect_quant_config_reader()`
- **File**: `tensorrt_llm/_torch/auto_deploy/models/quant_config_reader.py`
- **Change**: Added compressed-tensors as first detection priority:
  1. Compressed-Tensors
  2. ModelOpt
  3. HuggingFace

#### b. Updated Module Exports
- **File**: `tensorrt_llm/_torch/models/checkpoints/__init__.py`
- **Change**: Added imports for compressed-tensors components

### 4. Documentation

#### a. Module README
- **File**: `tensorrt_llm/_torch/models/checkpoints/compressed_tensors/README.md`
- **Content**: Technical documentation for developers

#### b. User Documentation
- **File**: `docs/source/features/compressed-tensors.md`
- **Content**: User-facing guide with examples and troubleshooting

#### c. Example Script
- **File**: `examples/compressed_tensors_example.py`
- **Content**: Runnable example demonstrating usage

### 5. Dependencies

#### a. Optional Dependency Added
- **File**: `setup.py`
- **Change**: Added to `extras_require`:
  ```python
  extras_require={
      "devel": devel_deps,
      "compressed-tensors": ["compressed-tensors>=0.8.0"],
  }
  ```

#### b. Graceful Degradation
- Weight loader checks for `compressed-tensors` availability
- Falls back to standard SafeTensors loading if unavailable
- Logs informative warning messages

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM API                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              ModelLoader (PyExecutor)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│          autodetect_quant_config_reader()                    │
│  1. Try CompressedTensorsConfigReader                        │
│  2. Try ModelOPTConfigReader                                 │
│  3. Try HFConfigReader                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│        CompressedTensorsConfigReader                         │
│  - Parse quantization_config from config.json                │
│  - Map to TRT-LLM QuantAlgo                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│        CompressedTensorsWeightLoader                         │
│  - Load .safetensors files                                   │
│  - Use compressed-tensors library (if available)             │
│  - Extract quantization metadata                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│        CompressedTensorsWeightMapper                         │
│  - Map checkpoint keys to model parameters                   │
│  - Extract scales, zero-points, sparsity info                │
│  - Convert to Linear layer format                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              Model.load_weights()                            │
│  - Load weights into model parameters                        │
│  - Use existing quantization kernels (GPTQ, AWQ, FP8)        │
└─────────────────────────────────────────────────────────────┘
```

## Format Mapping

### Compression Format → TRT-LLM QuantAlgo

| Compressed-Tensors Format | num_bits | TRT-LLM QuantAlgo |
|---------------------------|----------|-------------------|
| int-quantized (GPTQ)      | 4        | W4A16_GPTQ        |
| int-quantized (AWQ)       | 4        | W4A16_AWQ         |
| int-quantized             | 8        | W8A16             |
| float-quantized           | -        | FP8               |
| pack-quantized            | 4        | W4A16             |

### Weight Metadata Mapping

| Compressed-Tensors | TRT-LLM Linear Layer |
|-------------------|---------------------|
| `weight`          | `weight`            |
| `scales` or `weight_scale` | `weight_scale` |
| `zero_points` or `weight_zp` | `weight_zp` |
| `sparse_indices`  | `sparse_indices`    |
| `sparse_mask`     | `sparse_mask`       |

## Testing Strategy

### Unit Tests (To be added)

```python
# Test config reader
def test_compressed_tensors_config_reader():
    # Test detection of compressed-tensors format
    # Test format mapping for GPTQ, AWQ, FP8
    # Test error handling for unsupported formats
    pass

# Test weight loader
def test_compressed_tensors_weight_loader():
    # Test loading with compressed-tensors library
    # Test fallback to standard safetensors
    # Test metadata extraction
    pass

# Test weight mapper  
def test_compressed_tensors_weight_mapper():
    # Test weight key mapping
    # Test metadata extraction and conversion
    pass
```

### Integration Tests (To be added)

```python
def test_end_to_end_loading():
    # Test loading a real compressed-tensors model
    # Test inference produces correct outputs
    pass
```

## Compatibility

### Supported Python Versions
- Python 3.10+

### Supported Formats
- INT4 GPTQ
- INT4 AWQ  
- INT8 weight-only
- FP8
- Structured sparsity (experimental)

### Hardware Support
- Same as TensorRT-LLM quantization support
- GPTQ/AWQ: Hopper, Ada Lovelace, Ampere
- FP8: Hopper, Ada Lovelace, Blackwell

## Future Enhancements

1. **Enhanced Sparsity Support**: Better handling of sparse patterns
2. **Dynamic Quantization**: Support for activation quantization from compressed-tensors
3. **Format Conversion Tools**: CLI tools to convert between formats
4. **Performance Optimizations**: Faster weight loading for large models
5. **Test Coverage**: Comprehensive unit and integration tests

## Development Notes

### Adding New Format Support

To add support for a new compression format:

1. Update `CompressedTensorsConfigReader._map_*()` method:
   ```python
   def _map_new_format(self, qconf: Dict, config_groups: Dict) -> Dict:
       # Parse format-specific config
       # Return TRT-LLM compatible config
       pass
   ```

2. Update `CompressedTensorsWeightLoader._process_quantized_tensor()` if needed:
   ```python
   # Handle new metadata fields
   if "new_field" in tensor_dict:
       result["new_field"] = tensor_dict["new_field"]
   ```

3. Ensure corresponding quantization method exists in `tensorrt_llm.quantization`

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tensorrt_llm")
logger.setLevel(logging.DEBUG)
```

## References

- [Compressed-Tensors Library](https://github.com/neuralmagic/compressed-tensors)
- [TensorRT-LLM Quantization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md)
- [SafeTensors Format](https://github.com/huggingface/safetensors)

