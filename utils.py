# Blog utility functions for Kimi K2 model loading and training.
#
# Contains:
#   - plot_training_log: Plot training loss from a JSON log file
#   - make_device_map_simple: Create a simple layer-to-GPU device map
#   - override_compressed_linear_forward_fn: Patch CompressedLinear for on-the-fly decompression
#   - patch_skip_compress_model_on_load: Skip wasteful compress_model() during loading

import json
from typing import Dict

import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
from torch import Tensor
from torch.nn.functional import linear as torch_linear


def plot_training_log(log_path: str, output_path: str):
    import matplotlib.pyplot as plt

    with open(log_path) as f:
        log = json.load(f)

    steps = [entry["step"] for entry in log]
    losses = [entry["loss"] for entry in log]

    step_times = [entry["step_time"] for entry in log]
    total_tokens_list = [entry["total_tokens"] for entry in log]
    if len(step_times) > 1:
        avg_time = sum(step_times[1:]) / len(step_times[1:])
        # Tokens seen in steps 1..N (excluding step 0)
        tokens_after_first = total_tokens_list[-1] - total_tokens_list[0]
        avg_tokens_per_step = tokens_after_first / len(step_times[1:])
    else:
        avg_time = step_times[0] if step_times else 0
        avg_tokens_per_step = total_tokens_list[0] if total_tokens_list else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, linewidth=1.5)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=14)
    ax.set_title("Training Loss", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Average step time (excluding first): {avg_time:.2f}s")
    print(f"Average tokens per step (excluding first): {avg_tokens_per_step:.0f}")
    print(f"Plot saved to {output_path}")


def make_device_map_simple(model_id: str, num_gpus: int = 8) -> Dict[int | str, int]:
    """
    Create a device map that assigns consecutive layers to GPUs.

    For a model with 61 layers and 8 GPUs:
    - Layers 0-7 → GPU 0
    - Layers 8-15 → GPU 1
    - Layers 16-23 → GPU 2
    - etc.

    Args:
        model_id: HuggingFace model ID or local path
        num_gpus: Number of GPUs to distribute across

    Returns:
        Device map dict mapping layer names to GPU indices
    """
    # Determine num_layers - use local config if available, otherwise default
    num_layers = 61  # Default for Kimi K2 full model
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True
        )
        num_layers = config.num_hidden_layers
    except Exception:
        # If local config not available, check for known model paths
        if "small_model" in str(model_id):
            num_layers = 3  # Small model has 3 layers

    device_map = {}
    layers_per_gpu = max(1, (num_layers + num_gpus - 1) // num_gpus)
    for i in range(num_layers):
        name = f"model.layers.{i}"
        device_map[name] = min(i // layers_per_gpu, num_gpus - 1)
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = num_gpus - 1
    device_map["lm_head"] = num_gpus - 1
    return device_map


def override_compressed_linear_forward_fn():
    """
    Override CompressedLinear.forward for on-the-fly decompression.

    This patch prevents memory accumulation in MoE models by decompressing
    weights on-the-fly, computing the linear operation, and immediately
    deleting the decompressed weights.
    """

    def _compressed_linear_forward_on_the_fly(self, input: Tensor) -> Tensor:
        """
        Decompresses the weight on-the-fly, computes linear, then cleans up.
        Modified to prevent memory accumulation in MoE models.
        """
        weight_data = self.compressor.decompress_module(self)
        output = torch_linear(input, weight_data, self.bias)
        del weight_data
        return output

    CompressedLinear.forward = _compressed_linear_forward_on_the_fly


def patch_skip_compress_model_on_load():
    """
    Skip the wasteful compress_model() step during model loading.

    When loading a pre-compressed model (quantization_status='compressed'),
    transformers calls compress_model() which iterates through all Linear
    modules (~25k+ for Kimi-K2 MoE) and runs INT4 packing on random/meta
    weights. These weights are immediately overwritten when the actual
    checkpoint is loaded, making this step completely wasteful.

    This patch skips compress_model() and only applies the quantization
    config to set up the correct module structure. The actual compressed
    weights are loaded from the checkpoint afterward.

    This reduces loading time from ~20+ minutes to a few minutes.
    """
    from transformers.quantizers.quantizer_compressed_tensors import (
        CompressedTensorsHfQuantizer,
    )

    def patched_process_model_before_weight_loading(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        # Apply quantization config to set up module structure (CompressedLinear, etc.)
        apply_quantization_config(model, ct_quantization_config, self.run_compressed)

        # SKIP: self.compressor.compress_model(model=model)
        # The compress_model() call is wasteful because:
        # 1. It compresses random/meta weights that will be overwritten
        # 2. For Kimi-K2 MoE, this iterates ~25k modules at ~21 it/s = ~20 min
        # 3. The real weights are loaded from checkpoint AFTER this step

    CompressedTensorsHfQuantizer._process_model_before_weight_loading = (
        patched_process_model_before_weight_loading
    )
