# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
from typing import Any, Dict, List

import torch
import tqdm

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm.logger import logger


@register_checkpoint_weight_loader("COMPRESSED_TENSORS")
class CompressedTensorsWeightLoader(BaseWeightLoader):
    """
    Loads weights from compressed-tensors format (neuralmagic).
    Supports GPTQ, AWQ, sparse weights and other compressed formats.
    
    The compressed-tensors format extends safetensors with additional metadata
    for quantized and sparse weights. This loader handles decompression and
    conversion to formats compatible with TRT-LLM's Linear layers.
    """

    def __init__(self):
        super().__init__()
        self._compression_format = None
        self._quant_config = None

    def load_weights(self, checkpoint_dir: str) -> dict[str, Any]:
        """
        Load and process weights from compressed-tensors format.
        
        Args:
            checkpoint_dir: Path to the checkpoint directory containing
                           safetensors files and config.json with quantization_config.
        
        Returns:
            Dictionary containing processed weights ready for model loading.
        """
        # Load configuration to determine compression format
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            raise RuntimeError(f"No config.json found in {checkpoint_dir}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        self._quant_config = config.get("quantization_config", {})
        self._compression_format = self._quant_config.get("format")
        
        # Check if compressed-tensors library is available
        try:
            import compressed_tensors
            has_compressed_tensors = True
        except ImportError:
            has_compressed_tensors = False
            logger.warning(
                "compressed-tensors library not found. "
                "Will attempt to load weights as standard safetensors. "
                "Install with: pip install compressed-tensors"
            )
        
        # Find weight files
        weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        if not weight_files:
            raise RuntimeError(f"No safetensors files found in {checkpoint_dir}")
        
        # Load weights based on format
        if has_compressed_tensors and self._compression_format:
            logger.info(
                f"Loading compressed-tensors format: {self._compression_format}"
            )
            return self._load_compressed_weights(weight_files)
        else:
            # Fallback to standard safetensors loading
            logger.info("Loading as standard safetensors")
            return self._load_standard_weights(weight_files)

    def _load_compressed_weights(self, weight_files: List[str]) -> Dict[str, Any]:
        """
        Load weights using compressed-tensors library.
        
        This method leverages the compressed-tensors library to properly
        decompress and load quantized/sparse weights.
        """
        try:
            from compressed_tensors import load_compressed
        except ImportError:
            logger.warning(
                "compressed-tensors library not available, falling back to standard loading"
            )
            return self._load_standard_weights(weight_files)
        
        weights = {}
        pbar = tqdm.tqdm(total=len(weight_files), 
                        desc="Loading compressed-tensors weights")
        
        def load_one_file(file_path):
            """Load a single compressed-tensors file."""
            try:
                # The compressed-tensors library handles decompression
                file_weights = load_compressed(file_path)
                return self._process_compressed_state_dict(file_weights)
            except Exception as e:
                logger.warning(
                    f"Failed to load {file_path} with compressed-tensors: {e}. "
                    "Falling back to safetensors."
                )
                import safetensors
                return safetensors.torch.load_file(file_path)
        
        run_concurrently(load_one_file, [(f,) for f in weight_files],
                        reduce_func=weights.update, pbar=pbar)
        
        return weights

    def _process_compressed_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state dict loaded from compressed-tensors format.
        
        Compressed-tensors may store weights with additional metadata like
        scales, zero-points, sparsity masks, etc. This method extracts and
        organizes this information.
        """
        processed_weights = {}
        
        for key, value in state_dict.items():
            if isinstance(value, dict):
                # This is a quantized weight with metadata
                processed_weights[key] = self._process_quantized_tensor(value, key)
            elif isinstance(value, torch.Tensor):
                # Standard tensor
                processed_weights[key] = value
            else:
                # Keep other types as-is
                processed_weights[key] = value
        
        return processed_weights

    def _process_quantized_tensor(self, tensor_dict: Dict[str, Any], 
                                  tensor_name: str) -> Dict[str, torch.Tensor]:
        """
        Process a quantized tensor with metadata.
        
        Args:
            tensor_dict: Dictionary containing 'weight', 'scales', 'zero_points', etc.
            tensor_name: Name of the tensor for logging
        
        Returns:
            Dictionary with standardized keys for TRT-LLM Linear layers
        """
        # Extract components
        weight = tensor_dict.get("weight")
        scales = tensor_dict.get("scales", tensor_dict.get("weight_scale"))
        zero_points = tensor_dict.get("zero_points", tensor_dict.get("weight_zp"))
        
        result = {"weight": weight}
        
        if scales is not None:
            result["weight_scale"] = scales
        
        if zero_points is not None:
            result["weight_zp"] = zero_points
        
        # Handle sparse tensors
        if "indices" in tensor_dict or "mask" in tensor_dict:
            result["sparse_indices"] = tensor_dict.get("indices")
            result["sparse_mask"] = tensor_dict.get("mask")
        
        return result

    def _load_standard_weights(self, weight_files: List[str]) -> Dict[str, Any]:
        """
        Fallback to standard safetensors loading.
        
        Used when compressed-tensors library is not available or when
        the format doesn't require special handling.
        """
        import safetensors
        
        weights = {}
        pbar = tqdm.tqdm(total=len(weight_files), 
                        desc="Loading safetensors weights")
        
        def load_one_file(file_path):
            logger.info(f"Loading safetensors file: {file_path}")
            return safetensors.torch.load_file(file_path)
        
        run_concurrently(load_one_file, [(f,) for f in weight_files],
                        reduce_func=weights.update, pbar=pbar)
        
        return weights

