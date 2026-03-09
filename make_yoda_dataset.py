#!/usr/bin/env python3
"""
Generate a Yoda-style Q&A dataset using TriviaQA questions and Claude.

This script:
1. Loads questions from the TriviaQA dataset (Hugging Face)
2. Prompts Claude to answer each question in Yoda's speaking style
3. Saves the Q&A pairs to yoda_dataset.jsonl in the format expected by train.py

Usage:
    python make_yoda_dataset.py --num-questions 1000
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Tuple

import anthropic
from datasets import load_dataset
from torch.utils.data import Dataset

YODA_SYSTEM_PROMPT = (
    "Pretend you are Yoda. Adopt Yoda's style of speaking and personality."
)


class QADataset(Dataset):
    """Dataset for question-answer pairs from JSONL file.

    This dataset loads Q&A pairs and formats them for causal language model training,
    with the question portion masked in the labels.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load JSONL file
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "question" in data and "answer" in data:
                        self.examples.append(
                            {
                                "question": data["question"],
                                "answer": data["answer"],
                            }
                        )
                    else:
                        print(
                            f"Warning: Line {line_num} missing 'question' or 'answer', skipping"
                        )
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")

        print(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        question = example["question"]
        answer = example["answer"]

        # Format as chat messages: user message (question) + assistant message (answer)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        # Use tokenizer's chat template if available, otherwise fallback to simple format
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            user_messages = [{"role": "user", "content": question}]
            user_text = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,  # This adds the assistant prompt
            )
        else:
            print("No chat template found, using simple format")
            full_text = f"<|user|>\n{question}\n<|assistant|>\n{answer}"
            user_text = f"<|user|>\n{question}\n<|assistant|>\n"

        # Tokenize the full conversation
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize just the user part to find where to mask
        user_encodings = self.tokenizer(
            user_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        user_length = len(user_encodings["input_ids"])

        # Create labels: -100 for user message tokens (masked), actual token IDs for assistant response
        labels = encodings["input_ids"].clone()
        labels[0, :user_length] = -100  # Mask user message and prompt tokens
        labels[0, labels[0] == self.tokenizer.pad_token_id] = -100  # Mask padding

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Yoda-style Q&A dataset from TriviaQA"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=2000,
        help="Number of questions to generate",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model ID to use for generation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="yoda_dataset.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="TriviaQA split to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=48,
        help="Number of parallel workers for API calls (default: 1)",
    )
    return parser.parse_args()


def load_triviaqa_questions(
    split: str = "train", num_questions: int = 1000
) -> List[str]:
    """Load questions from TriviaQA dataset."""
    print(f"Loading TriviaQA dataset (split={split})...")

    # Load the dataset from Hugging Face
    # TriviaQA has different configurations, using 'unfiltered' which is most common
    dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=split)

    questions = []
    for i, example in enumerate(dataset):
        if i >= num_questions:
            break
        # TriviaQA has a 'question' field
        question = example["question"]
        questions.append(question)

    print(f"Loaded {len(questions)} questions from TriviaQA")
    return questions


def ask_claude_as_yoda(
    client: anthropic.Anthropic,
    question: str,
    question_idx: int,
    temperature: float,
    max_tokens: int,
    model: str = "claude-sonnet-4-20250514",
) -> Tuple[int, str, str]:
    """Ask Claude to answer a question in Yoda's style.

    Returns:
        Tuple of (question_idx, question, answer)
    """
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=YODA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        content = resp.content[0].text if resp.content else ""
        return (question_idx, question, content)
    except Exception as e:
        print(f"ERROR calling model for question {question_idx}: {e}")
        return (question_idx, question, "")


def _log_progress(completed: int, total: int, successful: int, failed: int, start_time: float) -> None:
    """Print progress stats every 10 completions."""
    if completed % 10 != 0:
        return
    elapsed = time.time() - start_time
    rate = completed / elapsed
    remaining = (total - completed) / rate if rate > 0 else 0
    print(f"\nProgress: {completed}/{total} ({100 * completed / total:.1f}%)")
    print(f"  Rate: {rate:.2f} questions/sec")
    print(f"  ETA: {remaining / 60:.1f} minutes")
    print(f"  Success: {successful}, Failed: {failed}\n")


def generate_yoda_dataset(args: argparse.Namespace) -> None:
    """Main dataset generation function."""
    print("=" * 80)
    print("YODA DATASET GENERATION")
    print(f"  Model: {args.model_id}")
    print(f"  Questions: {args.num_questions}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Output: {args.output_file}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 80)

    # Initialize Anthropic client (uses ANTHROPIC_API_KEY env var)
    client = anthropic.Anthropic()
    print(f"\nUsing model: {args.model_id}")

    # Load questions
    questions = load_triviaqa_questions(args.split, args.num_questions)

    # Generate Yoda answers
    print(
        f"\nGenerating {len(questions)} Yoda-style answers with {args.num_workers} workers..."
    )
    print("=" * 80)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    completed = 0
    start_time = time.time()

    # For thread-safe file writing and progress tracking
    write_lock = Lock()
    progress_lock = Lock()

    def handle_result(question: str, answer: str) -> None:
        """Write result to file and update counters (call under progress_lock)."""
        nonlocal successful, failed, completed
        completed += 1
        if answer:
            with write_lock:
                entry = {"question": question, "answer": answer}
                output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                output_file.flush()
            successful += 1
            print(f"[{completed}/{len(questions)}] {question[:60]}...")
        else:
            failed += 1
            print(f"[{completed}/{len(questions)}] FAILED: {question[:60]}...")
        _log_progress(completed, len(questions), successful, failed, start_time)

    # Open file for writing
    output_file = open(output_path, "w", encoding="utf-8")

    try:
        if args.num_workers == 1:
            for i, question in enumerate(questions):
                _, q, answer = ask_claude_as_yoda(
                    client, question, i, args.temperature, args.max_tokens, args.model_id
                )
                handle_result(q, answer)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        ask_claude_as_yoda,
                        client,
                        question,
                        i,
                        args.temperature,
                        args.max_tokens,
                        args.model_id,
                    ): i
                    for i, question in enumerate(questions)
                }

                for future in as_completed(future_to_idx):
                    _, question, answer = future.result()
                    with progress_lock:
                        handle_result(question, answer)

    finally:
        output_file.close()

    # Summary
    total_time = time.time() - start_time
    print("GENERATION COMPLETE")
    print(f"  Total questions: {len(questions)}, Successful: {successful}, Failed: {failed}")
    print(f"  Time: {total_time / 60:.1f} min, Output: {output_path.absolute()}")


def main():
    args = parse_args()
    generate_yoda_dataset(args)


if __name__ == "__main__":
    main()
