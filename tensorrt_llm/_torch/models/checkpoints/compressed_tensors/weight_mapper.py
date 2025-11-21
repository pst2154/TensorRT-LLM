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

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.linear import W4A16_AWQ_LinearMethod
from tensorrt_llm.logger import logger

from ..hf.weight_mapper import HfWeightMapper


@register_mapper("COMPRESSED_TENSORS")
class CompressedTensorsWeightMapper(HfWeightMapper):
    """
    Weight mapper for compressed-tensors format.
    
    Extends the HF weight mapper to handle compressed tensor metadata
    like quantization scales, zero-points, and sparsity information.
    """

    def __init__(self):
        super().__init__()
        # Add callback to process compressed tensors
        self._callbacks.append(self._process_compressed_tensors)

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        """
        Apply callbacks to process weights, including compressed tensors.
        
        Overrides parent to handle compressed tensor metadata before
        applying standard callbacks.
        """
        module_weights = []

        for new_name in self._mapping[module_name]:
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)
            
            # Check if this is a compressed tensor with metadata
            if self._is_compressed_tensor(fw):
                fw = self._extract_compressed_metadata(module, new_name, fw)
            
            for callback in self._callbacks:
                fw = callback(module, new_name, fw)
            module_weights.append(fw)

        return module_weights

    def _is_compressed_tensor(self, weights: dict) -> bool:
        """
        Check if weight dict contains compressed tensor metadata.
        
        Compressed tensors have additional keys like 'weight_scale',
        'weight_zp', 'sparse_indices', etc.
        """
        weight_value = weights.get('weight')
        
        # Check if weight is a dict (from compressed-tensors format)
        if isinstance(weight_value, dict):
            return True
        
        # Check if we have quantization metadata alongside the weight
        has_scale = 'weight_scale' in weights or 'scales' in weights
        has_zp = 'weight_zp' in weights or 'zero_points' in weights
        has_sparse = 'sparse_indices' in weights or 'sparse_mask' in weights
        
        return has_scale or has_zp or has_sparse

    def _extract_compressed_metadata(self, module: nn.Module, name: str, 
                                     weights: dict) -> dict:
        """
        Extract and standardize compressed tensor metadata.
        
        Converts compressed-tensors format to the format expected by
        TRT-LLM's Linear layers (W4A16_AWQ, W4A16_GPTQ, etc.).
        """
        weight_value = weights.get('weight')
        
        # If weight is a dict, extract components
        if isinstance(weight_value, dict):
            result = {
                'weight': weight_value.get('weight'),
            }
            
            # Extract scales (may be named 'scales' or 'weight_scale')
            scales = weight_value.get('scales') or weight_value.get('weight_scale')
            if scales is not None:
                result['weight_scale'] = scales
            
            # Extract zero points
            zp = weight_value.get('zero_points') or weight_value.get('weight_zp')
            if zp is not None:
                result['weight_zp'] = zp
            
            # Extract sparsity information
            if 'sparse_indices' in weight_value:
                result['sparse_indices'] = weight_value['sparse_indices']
            if 'sparse_mask' in weight_value:
                result['sparse_mask'] = weight_value['sparse_mask']
            
            # Copy any other metadata
            for key in weight_value:
                if key not in result and key not in ['weight', 'scales', 
                                                      'weight_scale', 
                                                      'zero_points', 'weight_zp']:
                    result[key] = weight_value[key]
            
            return result
        
        # Weight is already a tensor, just ensure metadata is in place
        return weights

    def _process_compressed_tensors(self, module: nn.Module, new_name: str,
                                    weights: dict) -> dict:
        """
        Process compressed tensors for specific module types.
        
        This callback handles format conversions needed for different
        quantization methods (AWQ, GPTQ, etc.).
        """
        # For now, pass through weights as-is
        # The Linear layer's load_weights will handle the actual conversion
        # based on the quant_method
        
        if 'weight_scale' in weights or 'weight_zp' in weights:
            logger.debug(
                f"Processing compressed tensor for {new_name} with "
                f"quantization metadata"
            )
        
        return weights

