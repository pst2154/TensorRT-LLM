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

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader


@register_config_loader("COMPRESSED_TENSORS")
class CompressedTensorsConfigLoader(BaseConfigLoader):
    """
    Config loader for compressed-tensors format.
    
    Since compressed-tensors is primarily a storage format that extends
    HuggingFace models, we can reuse the standard HF config loading.
    """

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        """
        Load model config from compressed-tensors checkpoint.
        
        Compressed-tensors checkpoints use standard HF config.json,
        so we can delegate to the standard from_pretrained method.
        """
        return ModelConfig.from_pretrained(checkpoint_dir, **kwargs)

