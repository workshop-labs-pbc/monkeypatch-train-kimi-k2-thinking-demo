"""
Kimi-K2 LoRA fine-tuning with HuggingFace + PEFT.

Usage:
    python train.py [--save-dir ./blog_run]
    python train.py plot ./blog_run/training_log.json [--output loss_curve.png]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_yoda_dataset import QADataset
from utils import (
    make_device_map_simple,
    override_compressed_linear_forward_fn,
    patch_skip_compress_model_on_load,
    plot_training_log,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./blog_run", help="Directory to save LoRA weights and training log")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    patch_skip_compress_model_on_load()
    override_compressed_linear_forward_fn()

    model_id = "moonshotai/Kimi-K2-Thinking"
    target_modules = ["shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 512
    batch_size = 40
    dataset = QADataset("yoda_dataset.jsonl", tokenizer, max_length=max_length)
    print(f"Loaded yoda dataset with {len(dataset)} examples")

    device_map = make_device_map_simple(model_id, torch.cuda.device_count())
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, use_cache=False, attn_implementation="flash_attention_2", device_map=device_map, torch_dtype=torch.bfloat16)
    print("Loaded successfully!")

    model.requires_grad_(False)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.eval()
    model.eval()
    print("Applied lora!")

    encodings = tokenizer("Hello, how are you?", return_tensors="pt")
    encodings = {k: v.to(model.device) for k, v in encodings.items()}
    outputs = model(**encodings)
    print("Forward pass successful!", tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True))

    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with loss logging
    training_log = []
    total_tokens = 0
    max_steps = 40
    for i, batch in enumerate(dataloader):
        if i >= max_steps:
            break
        step_start = time.time()
        device = next(lora_model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        total_tokens += batch["attention_mask"].sum().item()
        outputs = lora_model(**batch)
        if i == 0:
            print("Successful forward pass with lora!")
        loss = outputs.loss
        loss.backward()
        if i == 0:
            print("Successful backward pass!")
        clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        loss_val = loss.item()
        training_log.append({"step": i, "loss": loss_val, "step_time": step_time, "total_tokens": int(total_tokens)})
        print(f"Step {i} complete! Loss: {loss_val:.4f} | Tokens: {int(total_tokens):,} | Time: {step_time:.2f}s")

    # Save training log
    log_path = save_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Save LoRA weights
    lora_path = save_dir / "lora_weights"
    lora_model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA weights saved to {lora_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        # python train.py plot <log_path> [--output <path>]
        parser = argparse.ArgumentParser()
        parser.add_argument("cmd")
        parser.add_argument("log_path", type=str, help="Path to training_log.json")
        parser.add_argument("--output", type=str, default=None, help="Output image path")
        args = parser.parse_args()
        output = args.output or str(Path(args.log_path).parent / "loss_curve.png")
        plot_training_log(args.log_path, output)
    else:
        main()
