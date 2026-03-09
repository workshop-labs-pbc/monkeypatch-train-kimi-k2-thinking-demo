# Train Kimi with Huggingface

## Usage

### 1. Generate the Yoda dataset
Requires an Anthropic API key. Generates Yoda-style Q&A pairs from TriviaQA questions using Claude.

`export ANTHROPIC_API_KEY="sk-ant-..."`
`python make_yoda_dataset.py --num-questions 2000 --num-workers 48 --output-file yoda_dataset.jsonl`

###2. Fine-tune Kimi-K2 with LoRA
Requires 8× H200s (or a configuration with more VRAM)

`python train.py --save-dir ./blog_run`

This will:
- Load the tokenizer and dataset from yoda_dataset.jsonl
- Download and load moonshotai/Kimi-K2-Thinking across 8 GPUs
- Apply LoRA adapters and train for 40 steps
- Save LoRA weights and a training log to --save-dir

NOTE: this does not allow training quantized experts. It trains LoRAs on non quantized weights including attention and shared experts.

###3. Plot training loss
`python train.py plot ./blog_run/training_log.json --output loss_curve.png`
