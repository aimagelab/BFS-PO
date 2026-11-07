import argparse
import os
import statistics

from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from trl.data_utils import maybe_apply_chat_template
from trl.extras.dataset_formatting import resolve_reasoning_dataset
from trl.extras.prompt_variations import maybe_apply_custom_chat_template, unwrap_tokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load metamathqa and math-500 with the same reasoning prompt preprocessing used in training, "
            "tokenize the rendered prompts, and report the maximum prompt length."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model or tokenizer path used to build and tokenize the prompts.",
    )
    parser.add_argument(
        "--processing-class-name",
        type=str,
        default=None,
        help="Optional processor/tokenizer path. Defaults to --model.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["metamathqa", "math-500"],
        help="Datasets to inspect. Defaults to: metamathqa math-500",
    )
    parser.add_argument(
        "--metamath-max-random-samples",
        type=int,
        default=None,
        help=(
            "Optional cap for metamathqa; 60000 samples with seed 42 reproduces the MetaMathQA-60k selection. "
            "Leave unset to scan the full dataset."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used if --metamath-max-random-samples is set.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache dir.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Optional HF_HOME value.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before loading artifacts.",
    )
    return parser.parse_args()


def configure_environment(args):
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_processing_class(model_name_or_path, processing_class_name=None):
    source = processing_class_name or model_name_or_path
    try:
        processing_class = AutoProcessor.from_pretrained(source)
    except Exception:
        processing_class = AutoTokenizer.from_pretrained(source)
    maybe_apply_custom_chat_template(processing_class)
    return processing_class, unwrap_tokenizer(processing_class)


def load_reasoning_dataset_local(
    dataset_name,
    dataset_config,
    dataset_split,
    tokenizer=None,
    max_random_samples=None,
    random_seed=42,
    cache_dir=None,
):
    dataset_spec = resolve_reasoning_dataset(dataset_name, dataset_config, dataset_split, tokenizer=tokenizer)
    dataset = load_dataset(
        dataset_spec["dataset_name"],
        dataset_spec["dataset_config"],
        split=dataset_spec["dataset_split"],
        cache_dir=cache_dir,
    )

    if max_random_samples is not None and len(dataset) > max_random_samples:
        dataset = dataset.shuffle(seed=random_seed).select(range(max_random_samples))

    preprocessing_fn = dataset_spec["preprocessing_fn"]
    if preprocessing_fn is not None:
        dataset = dataset.map(
            preprocessing_fn,
            load_from_cache_file=dataset_spec["load_from_cache_file"],
            desc=f"Preprocessing {dataset_name}",
        )

    return dataset, dataset_spec


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def tokenize_prompt(example, tokenizer):
    rendered = maybe_apply_chat_template(example, tokenizer)
    prompt_text = rendered["prompt"]
    tokenized = tokenizer(prompt_text, add_special_tokens=False)
    return prompt_text, len(tokenized["input_ids"])


def analyze_dataset(dataset_name, tokenizer, args):
    max_random_samples = None
    if dataset_name.lower() in {"metamathqa", "meta-math/metamathqa"}:
        max_random_samples = args.metamath_max_random_samples

    dataset, dataset_spec = load_reasoning_dataset_local(
        dataset_name=dataset_name,
        dataset_config=None,
        dataset_split=None,
        tokenizer=tokenizer,
        max_random_samples=max_random_samples,
        random_seed=args.random_seed,
        cache_dir=args.dataset_cache_dir,
    )

    lengths = []
    longest = {"index": -1, "length": -1, "question": "", "prompt_text": ""}

    for idx, example in enumerate(tqdm(dataset, desc=f"Tokenizing {dataset_name}")):
        prompt_text, prompt_length = tokenize_prompt(example, tokenizer)
        lengths.append(prompt_length)
        if prompt_length > longest["length"]:
            longest = {
                "index": idx,
                "length": prompt_length,
                "question": str(example.get("question", "")),
                "prompt_text": prompt_text,
            }

    sorted_lengths = sorted(lengths)
    mean_length = statistics.fmean(lengths) if lengths else 0.0

    return {
        "requested_name": dataset_name,
        "resolved_name": dataset_spec["dataset_name"],
        "resolved_split": dataset_spec["dataset_split"],
        "num_examples": len(dataset),
        "max_random_samples": max_random_samples,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": mean_length,
        "p95_length": percentile(sorted_lengths, 0.95),
        "p99_length": percentile(sorted_lengths, 0.99),
        "longest": longest,
    }


def print_summary(summary):
    print("=" * 100)
    print(f"Dataset requested: {summary['requested_name']}")
    print(f"Resolved dataset:  {summary['resolved_name']}")
    print(f"Split:             {summary['resolved_split']}")
    print(f"Examples:          {summary['num_examples']}")
    if summary["max_random_samples"] is not None:
        print(f"Sample cap:        {summary['max_random_samples']}")
    print(f"Min prompt tokens: {summary['min_length']}")
    print(f"Mean prompt tokens:{summary['mean_length']:.2f}")
    print(f"P95 prompt tokens: {summary['p95_length']:.2f}")
    print(f"P99 prompt tokens: {summary['p99_length']:.2f}")
    print(f"Max prompt tokens: {summary['max_length']}")
    print(f"Longest example ix:{summary['longest']['index']}")
    if summary["longest"]["question"]:
        print(f"Longest question:  {summary['longest']['question']}")


def main():
    args = parse_args()
    configure_environment(args)
    _, tokenizer = load_processing_class(args.model, args.processing_class_name)

    print(f"Tokenizer:         {getattr(tokenizer, 'name_or_path', '<unknown>')}")
    print(f"Uses chat template:{bool(getattr(tokenizer, 'chat_template', None))}")
    print(f"Offline mode:      {args.offline}")
    if args.dataset_cache_dir:
        print(f"Datasets cache:    {args.dataset_cache_dir}")
    if args.hf_home:
        print(f"HF_HOME:           {args.hf_home}")

    for dataset_name in args.datasets:
        summary = analyze_dataset(dataset_name, tokenizer, args)
        print_summary(summary)


if __name__ == "__main__":
    main()
