# How (not) to train Kimi-K2-Thinking with HuggingFace

See [our blog post](https://workshoplabs.ai/blog/open-weights-open-training) for details.

This is *not* intended to actually be used for post-training of Kimi K2 Thinking - it is slow and inefficient. It is a demonstration of the patches needed to make it even work in principle on top of HuggingFace.

## Usage

### 1. Generate the Yoda dataset 
This step is optional. A generated datasets exists at yoda_dataset.jsonl, or you can use an alternative dataset. Requires an Anthropic API key. Generates Yoda-style Q&A pairs from TriviaQA questions using Claude.
Run:
1. `export ANTHROPIC_API_KEY="sk-ant-..."`
2. `python make_yoda_dataset.py --num-questions 2000 --num-workers 48 --output-file yoda_dataset.jsonl`

### 2. Fine-tune Kimi-K2 with LoRA

Requires 8× H200s (or a configuration with more VRAM)
Run:
`python train.py --save-dir ./blog_run`

This will:
- Load the tokenizer and dataset from yoda_dataset.jsonl
- Download and load moonshotai/Kimi-K2-Thinking across 8 GPUs
- Apply LoRA adapters and train for 40 steps
- Save LoRA weights and a training log to --save-dir

NOTE: this does not allow training quantized experts. It trains LoRAs on non quantized weights including attention and shared experts.

### 3. Plot training loss
Run:
`python train.py plot ./blog_run/training_log.json --output loss_curve.png`
