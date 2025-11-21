#!/usr/bin/env python3
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

"""
Example demonstrating how to use TensorRT-LLM with compressed-tensors format models.

This example shows how to load and run inference with models quantized using
the compressed-tensors format from neuralmagic.
"""

import argparse

from tensorrt_llm import LLM


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with compressed-tensors format models")
    parser.add_argument(
        "--model",
        type=str,
        default="neuralmagic/Llama-2-7b-pruned50-quantized-gptq",
        help="Model name or path to compressed-tensors format checkpoint")
    parser.add_argument("--prompt",
                        type=str,
                        default="Hello, my name is",
                        help="Input prompt for generation")
    parser.add_argument("--max_tokens",
                        type=int,
                        default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.7,
                        help="Sampling temperature")
    
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print("Note: This will automatically detect and load the compressed-tensors format")
    
    # Initialize LLM with the compressed-tensors model
    # The checkpoint loader will automatically detect the format
    llm = LLM(model=args.model)
    
    print(f"\nGenerating with prompt: '{args.prompt}'")
    
    # Generate text
    outputs = llm.generate(
        args.prompt,
        sampling_params={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        }
    )
    
    print("\nGenerated output:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

