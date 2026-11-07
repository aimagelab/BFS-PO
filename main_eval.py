import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from trl.extras.dataset_formatting import resolve_reasoning_dataset
from trl.extras.prompt_variations import maybe_apply_custom_chat_template


DEFAULT_EVAL_DATASETS = ["gsm8k", "math-500", "amc2023", "mmlu-stem"]
EVAL_RESUME_DATASET_NAME_FIELD = "__trl_eval_dataset_name"
EVAL_RESUME_EXAMPLE_INDEX_FIELD = "__trl_eval_example_index"
DEFAULT_AES_BASELINE_METRICS = {
    "eval_amc2023_accuracy": 0.475,
    "eval_amc2023_completions/mean_length": 2092.0,
    "eval_gsm8k_accuracy": 0.868,
    "eval_gsm8k_completions/mean_length": 873.0,
    "eval_math-500_accuracy": 0.642,
    "eval_math-500_completions/mean_length": 1675.0,
    "eval_mmlu-stem_accuracy": 0.741,
    "eval_mmlu-stem_completions/mean_length": 1081.0,
}


def parse_bool(raw_value):
    if raw_value.lower() in {"true", "false"}:
        return raw_value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_optional_mapping(raw_value, argument_name):
    if raw_value is None:
        return {}

    parsed = yaml.safe_load(raw_value)
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise ValueError(f"`{argument_name}` must be a YAML/JSON mapping keyed by dataset name.")

    return dict(parsed)


def load_processing_class(model_name, checkpoint_path, processing_class_name=None):
    candidate_sources = [processing_class_name, checkpoint_path, model_name]
    attempted_sources = []

    for source in candidate_sources:
        if source is None or source in attempted_sources:
            continue
        attempted_sources.append(source)

        try:
            return AutoProcessor.from_pretrained(source)
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(source)
            except Exception:
                continue

    raise RuntimeError(
        "Unable to load a processing class from any of the following sources: "
        f"{attempted_sources}. Consider passing `--processing_class_name` explicitly."
    )


def maybe_restore_base_model_name(processing_class, model_name, checkpoint_path):
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    current_name_or_path = str(getattr(tokenizer, "name_or_path", "") or "")

    checkpoint_path_abs = os.path.abspath(checkpoint_path)
    current_abs = os.path.abspath(current_name_or_path) if current_name_or_path else ""
    if current_name_or_path and current_abs != checkpoint_path_abs:
        return

    if hasattr(tokenizer, "name_or_path"):
        tokenizer.name_or_path = model_name
    if processing_class is not tokenizer and hasattr(processing_class, "name_or_path"):
        processing_class.name_or_path = model_name


def load_reasoning_dataset(
    dataset_name,
    dataset_config,
    dataset_split,
    tokenizer=None,
    max_random_samples=None,
    max_sample_fraction=None,
    random_seed=42,
    default_split="test",
):
    if dataset_name is None:
        raise ValueError("Dataset name cannot be None.")

    dataset_spec = resolve_reasoning_dataset(dataset_name, dataset_config, dataset_split, tokenizer=tokenizer)
    if dataset_spec["dataset_split"] is None:
        dataset_spec["dataset_split"] = default_split
    cache_dir = os.getenv("HF_DATASETS_CACHE")

    dataset = load_dataset(
        dataset_spec["dataset_name"],
        dataset_spec["dataset_config"],
        split=dataset_spec["dataset_split"],
        cache_dir=cache_dir,
    )

    if max_sample_fraction is not None:
        if not 0 < max_sample_fraction <= 1:
            raise ValueError("`max_sample_fraction` must be in the interval (0, 1].")
        fraction_sample_count = max(1, math.ceil(len(dataset) * max_sample_fraction))
        max_random_samples = (
            fraction_sample_count
            if max_random_samples is None
            else min(max_random_samples, fraction_sample_count)
        )

    if max_random_samples is not None and len(dataset) > max_random_samples:
        dataset = dataset.shuffle(seed=random_seed).select(range(max_random_samples))

    preprocessing_fn = dataset_spec["preprocessing_fn"]
    if preprocessing_fn is not None:
        dataset = dataset.map(
            preprocessing_fn,
            # Imported preprocessors can keep the same cache fingerprint after code changes.
            load_from_cache_file=False,
            # Avoid concurrent cache writes from distributed workers on the shared filesystem.
            keep_in_memory=True,
        )

    return dataset


def resolve_dataset_field(field_name, field_value, dataset_key):
    if isinstance(field_value, dict):
        return field_value.get(dataset_key)
    return field_value


def load_eval_datasets(
    eval_dataset_name,
    eval_dataset_config,
    eval_dataset_split,
    tokenizer=None,
    max_random_samples=None,
    max_sample_fraction=None,
    random_seed=42,
    default_split="test",
):
    eval_datasets = {}
    for dataset_key, dataset_name in eval_dataset_name.items():
        dataset_config = resolve_dataset_field("dataset_configs", eval_dataset_config, dataset_key)
        dataset_split = resolve_dataset_field("dataset_splits", eval_dataset_split, dataset_key)
        dataset = load_reasoning_dataset(
            dataset_name,
            dataset_config,
            dataset_split,
            tokenizer=tokenizer,
            max_random_samples=max_random_samples,
            max_sample_fraction=max_sample_fraction,
            random_seed=random_seed,
            default_split=default_split,
        )
        if EVAL_RESUME_DATASET_NAME_FIELD not in dataset.column_names:
            dataset = dataset.add_column(EVAL_RESUME_DATASET_NAME_FIELD, [dataset_key] * len(dataset))
        if EVAL_RESUME_EXAMPLE_INDEX_FIELD not in dataset.column_names:
            dataset = dataset.add_column(EVAL_RESUME_EXAMPLE_INDEX_FIELD, list(range(len(dataset))))
        eval_datasets[dataset_key] = dataset
    return eval_datasets


def load_generated_cot_examples(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}, but found {type(payload).__name__}.")

    return payload


def _coerce_resume_record_key(
    record: Mapping[str, Any],
    question_to_keys: Mapping[str, list[tuple[str, int]]],
) -> tuple[str, int]:
    dataset_name = record.get("dataset_name")
    dataset_index = record.get("dataset_example_index")
    if dataset_name is None:
        dataset_name = record.get(EVAL_RESUME_DATASET_NAME_FIELD)
    if dataset_index is None:
        dataset_index = record.get(EVAL_RESUME_EXAMPLE_INDEX_FIELD)

    if dataset_name is not None and dataset_index is not None:
        return str(dataset_name), int(dataset_index)

    question = record.get("question")
    if not isinstance(question, str) or question == "":
        raise ValueError(
            "Each resume record must include either dataset metadata "
            f"({EVAL_RESUME_DATASET_NAME_FIELD}/{EVAL_RESUME_EXAMPLE_INDEX_FIELD}) or a non-empty `question`."
        )

    matching_keys = question_to_keys.get(question, [])
    if len(matching_keys) == 1:
        return matching_keys[0]
    if not matching_keys:
        raise ValueError(f"Resume question was not found in the current eval datasets: {question[:120]!r}")
    raise ValueError(
        "Resume question is ambiguous across the current eval datasets. "
        "Please rerun from a generated_cot_examples file that contains dataset metadata."
    )


def prepare_generated_cot_resume(
    generated_cot_examples: Sequence[Mapping[str, Any]],
    eval_datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    question_to_keys: dict[str, list[tuple[str, int]]] = {}
    ordered_dataset_keys: list[tuple[str, int]] = []
    for dataset_name, dataset in eval_datasets.items():
        for dataset_index, example in enumerate(dataset):
            ordered_dataset_keys.append((dataset_name, dataset_index))
            question = example.get("question")
            if isinstance(question, str):
                question_to_keys.setdefault(question, []).append((dataset_name, dataset_index))

    record_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in generated_cot_examples:
        resume_key = _coerce_resume_record_key(record, question_to_keys)
        record_by_key[resume_key] = dict(record)

    missing_indices_by_dataset: dict[str, list[int]] = {dataset_name: [] for dataset_name in eval_datasets}
    ordered_examples: list[dict[str, Any]] = []
    for dataset_name, dataset_index in ordered_dataset_keys:
        resume_key = (dataset_name, dataset_index)
        if resume_key in record_by_key:
            ordered_examples.append(record_by_key[resume_key])
        else:
            missing_indices_by_dataset[dataset_name].append(dataset_index)

    return {
        "ordered_examples": ordered_examples,
        "record_by_key": record_by_key,
        "missing_indices_by_dataset": missing_indices_by_dataset,
        "missing_example_count": sum(len(indices) for indices in missing_indices_by_dataset.values()),
        "total_example_count": len(ordered_dataset_keys),
    }


def filter_eval_datasets_to_missing_examples(
    eval_datasets: Mapping[str, Any],
    missing_indices_by_dataset: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    filtered_eval_datasets = {}
    for dataset_name, dataset in eval_datasets.items():
        missing_indices = list(missing_indices_by_dataset.get(dataset_name, []))
        if hasattr(dataset, "select"):
            filtered_eval_datasets[dataset_name] = dataset.select(missing_indices)
        else:
            filtered_eval_datasets[dataset_name] = [dataset[index] for index in missing_indices]
    return filtered_eval_datasets


def order_records_by_dataset(
    records: Sequence[Mapping[str, Any]],
    eval_datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    ordered_records = prepare_generated_cot_resume(records, eval_datasets)["ordered_examples"]
    return [dict(record) for record in ordered_records]


def _count_generated_tokens(processing_class, completion_text: str) -> int:
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    if not completion_text:
        return 0
    return len(tokenizer.encode(completion_text, add_special_tokens=False))


def _is_generated_cot_truncated(
    generated_cot: Mapping[str, Any],
    *,
    scored_completion: str,
    full_completion: str,
    cot_length: float,
) -> bool:
    explicit_flag = generated_cot.get("is_truncated")
    if explicit_flag is not None:
        return bool(explicit_flag)

    return bool(full_completion) and scored_completion == "" and math.isclose(cot_length, 0.0)


def _resolve_generated_cot_length(
    generated_cot: Mapping[str, Any],
    *,
    processing_class,
    scored_completion: str,
    full_completion: str,
    dont_consider_truncated_cot_in_length_count: bool = False,
) -> tuple[float, bool]:
    cot_length = float(generated_cot.get("cot_length", 0.0) or 0.0)
    is_truncated = _is_generated_cot_truncated(
        generated_cot,
        scored_completion=scored_completion,
        full_completion=full_completion,
        cot_length=cot_length,
    )
    # A nonzero CoT length is valid even if generation stopped during the final answer.
    if not is_truncated or not math.isclose(cot_length, 0.0):
        return cot_length, is_truncated
    if dont_consider_truncated_cot_in_length_count:
        return 0.0, is_truncated

    max_num_tokens = generated_cot.get("max_num_tokens")
    if max_num_tokens is not None:
        try:
            return float(max_num_tokens), is_truncated
        except (TypeError, ValueError):
            pass

    return float(_count_generated_tokens(processing_class, full_completion)), is_truncated


def rebuild_eval_state_from_generated_examples(
    generated_cot_examples: Sequence[Mapping[str, Any]],
    processing_class,
    *,
    dont_consider_truncated_cot_in_length_count: bool = False,
) -> dict[str, float | int]:
    """Restore metrics, optionally excluding truncated completions with zero CoT length from the length average."""
    from trl import GRPOTrainer
    from trl.rewards import accuracy_reward

    metrics_totals: dict[str, float | int] = {
        "accuracy_sum": 0.0,
        "accuracy_count": 0,
        "completion_length_sum": 0.0,
        "completion_length_count": 0,
        "new_token_sum": 0.0,
        "new_token_count": 0,
        "prompt_count": 0,
        "shortest_completion_accuracy_sum": 0.0,
        "shortest_completion_accuracy_count": 0,
    }
    for example in generated_cot_examples:
        generated_cots = list(example.get("generated_cots", []))
        if not generated_cots:
            continue

        question = str(example.get("question", "") or "")
        complete_answer = str(example.get("ground_truth_answer", "") or "")
        reference_solution = str(example.get("reference_solution", "") or "")
        scored_completions = [str(generated.get("scored_completion", "") or "") for generated in generated_cots]
        full_completions = [
            str(generated.get("full_completion", generated.get("scored_completion", "")) or "")
            for generated in generated_cots
        ]
        cot_lengths = []
        for generated, scored_completion, full_completion in zip(
            generated_cots, scored_completions, full_completions, strict=True
        ):
            resolved_length, _ = _resolve_generated_cot_length(
                generated,
                processing_class=processing_class,
                scored_completion=scored_completion,
                full_completion=full_completion,
                dont_consider_truncated_cot_in_length_count=dont_consider_truncated_cot_in_length_count,
            )
            cot_lengths.append(resolved_length)
        new_token_counts = [_count_generated_tokens(processing_class, completion) for completion in full_completions]
        completion_accuracies = accuracy_reward(
            [[{"role": "assistant", "content": completion}] for completion in scored_completions],
            complete_answers=[complete_answer] * len(scored_completions),
            question=[question] * len(scored_completions),
            solution=[reference_solution] * len(scored_completions),
        )
        completion_accuracies = [float(accuracy) for accuracy in completion_accuracies]

        metrics_totals["accuracy_sum"] += sum(completion_accuracies)
        metrics_totals["accuracy_count"] += len(completion_accuracies)
        metrics_totals["completion_length_sum"] += sum(cot_lengths)
        metrics_totals["completion_length_count"] += sum(1 for length in cot_lengths if not math.isclose(length, 0.0))
        metrics_totals["new_token_sum"] += sum(new_token_counts)
        metrics_totals["new_token_count"] += len(new_token_counts)
        metrics_totals["prompt_count"] += 1

        if len(new_token_counts) > 1:
            shortest_index = GRPOTrainer._select_shortest_completion_index(new_token_counts)
            metrics_totals["shortest_completion_accuracy_sum"] += completion_accuracies[shortest_index]
            metrics_totals["shortest_completion_accuracy_count"] += 1

    return metrics_totals


def metrics_from_eval_totals(
    metrics_totals: Mapping[str, float | int],
    *,
    dataset_name: str,
    include_shortest_completion_accuracy: bool,
) -> dict[str, float]:
    metric_prefix = f"eval_{dataset_name}"
    metrics = {
        f"{metric_prefix}_accuracy": float(metrics_totals["accuracy_sum"]) / float(metrics_totals["accuracy_count"]),
        f"{metric_prefix}_completions/mean_new_tokens": float(metrics_totals["new_token_sum"])
        / float(metrics_totals["new_token_count"]),
        f"{metric_prefix}_completions/mean_total_new_tokens_per_prompt": float(metrics_totals["new_token_sum"])
        / float(metrics_totals["prompt_count"]),
        f"{metric_prefix}_completions/total_new_tokens": float(metrics_totals["new_token_sum"]),
        f"{metric_prefix}_runtime": 0.0,
        f"{metric_prefix}_samples_per_second": 0.0,
        f"{metric_prefix}_steps_per_second": 0.0,
    }
    if int(metrics_totals["completion_length_count"]) == 0:
        metrics[f"{metric_prefix}_completions/mean_length"] = 0.0
    else:
        metrics[f"{metric_prefix}_completions/mean_length"] = float(metrics_totals["completion_length_sum"]) / float(
            metrics_totals["completion_length_count"]
        )
    if include_shortest_completion_accuracy and int(metrics_totals["shortest_completion_accuracy_count"]) > 0:
        metrics[f"{metric_prefix}_shortest_completion_accuracy"] = float(
            metrics_totals["shortest_completion_accuracy_sum"]
        ) / float(metrics_totals["shortest_completion_accuracy_count"])
    return metrics


def load_model_for_eval(model_name, checkpoint_path, model_init_kwargs):
    from trl.trainer.utils import create_model_from_path

    if model_init_kwargs.get("trust_remote_code", False):
        # Custom architectures are resolved through auto_map,
        # rather than looked up as a class exported by transformers.
        model_init_kwargs = {**model_init_kwargs, "architecture": AutoModelForCausalLM}

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.isdir(checkpoint_path) and os.path.exists(adapter_config_path):
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "The checkpoint looks like a PEFT adapter, but `peft` is not installed in this environment."
            ) from e

        base_model = create_model_from_path(model_name, **model_init_kwargs)
        return PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)

    return create_model_from_path(checkpoint_path, **model_init_kwargs)


def compute_aes(fine_tuned_accuracy, zero_shot_accuracy, fine_tuned_len_cot, zero_shot_len_cot):
    if math.isclose(zero_shot_accuracy, 0.0) or math.isclose(zero_shot_len_cot, 0.0):
        return float("nan")

    accuracy_weight = 3.0 if fine_tuned_accuracy >= zero_shot_accuracy else 5.0
    accuracy_term = accuracy_weight * (fine_tuned_accuracy - zero_shot_accuracy) / zero_shot_accuracy
    length_term = (zero_shot_len_cot - fine_tuned_len_cot) / zero_shot_len_cot
    return accuracy_term + length_term


def add_aes_metrics(metrics, baseline_metrics):
    updated_metrics = dict(metrics)
    for metric_key, fine_tuned_accuracy in metrics.items():
        if not metric_key.endswith("_accuracy"):
            continue

        eval_prefix = metric_key[: -len("_accuracy")]
        mean_length_key = f"{eval_prefix}_completions/mean_length"
        if mean_length_key not in metrics:
            continue
        if metric_key not in baseline_metrics or mean_length_key not in baseline_metrics:
            continue

        updated_metrics[f"{eval_prefix}_aes"] = compute_aes(
            fine_tuned_accuracy=float(fine_tuned_accuracy),
            zero_shot_accuracy=float(baseline_metrics[metric_key]),
            fine_tuned_len_cot=float(metrics[mean_length_key]),
            zero_shot_len_cot=float(baseline_metrics[mean_length_key]),
        )

    return updated_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_EVAL_DATASETS)
    parser.add_argument("--dataset_configs", type=str, default=None)
    parser.add_argument("--dataset_splits", type=str, default=None)
    parser.add_argument("--processing_class_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--save_metrics_path", type=str, default=None)
    parser.add_argument("--save_generated_cot_examples_json_path", type=str, default=None)
    parser.add_argument(
        "--no_save_generated_cot_examples",
        action="store_true",
        help="Disable saving generated examples and rendered prompts, including per-batch JSON writes.",
    )
    parser.add_argument("--resume_generated_cot_examples_json_path", type=str, default=None)
    parser.add_argument("--save_generated_cot_examples_every_batch", action="store_true")
    parser.add_argument(
        "--save_rendered_prompts",
        action="store_true",
        help="Save the exact rendered prompt, including special chat-template tokens, in generated_cot_examples.json.",
    )
    parser.add_argument("--report_to", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument("--max_samples_fraction", type=float, default=None)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument(
        "--dont_consider_truncated_cot_in_length_count",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{true,false}",
        help=(
            "Exclude truncated completions with zero CoT length from the mean; keep nonempty closed CoTs "
            "even when the final answer is truncated (default: false)."
        ),
    )
    parser.add_argument("--num_generations", "--num_generations_eval", dest="num_generations", type=int, default=1)
    parser.add_argument("--eval_sampling_strategy", type=str, default="independent")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="vLLM presence penalty in [-2, 2]. Nonzero values require --use_vllm.",
    )
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate")
    parser.add_argument("--vllm_server_base_url", type=str, default=None)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--vllm_max_model_length", type=int, default=None)
    parser.add_argument("--vllm_model_impl", type=str, default="vllm")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--ddp_timeout", type=int, default=1800)

    args = parser.parse_args()

    if not -2.0 <= args.presence_penalty <= 2.0:
        parser.error("--presence_penalty must be between -2 and 2.")
    if args.presence_penalty != 0.0 and not args.use_vllm:
        parser.error("A nonzero --presence_penalty requires --use_vllm.")


    if args.no_save_generated_cot_examples:
        args.save_generated_cot_examples_json_path = None
        args.save_generated_cot_examples_every_batch = False
        args.save_rendered_prompts = False
    elif args.save_generated_cot_examples_json_path is None:
        args.save_generated_cot_examples_json_path = os.path.join(args.output_dir, "generated_cot_examples.json")

    from trl import GRPOConfig, GRPOTrainer
    from trl.rewards import accuracy_reward

    if args.per_device_eval_batch_size % args.num_generations != 0:
        raise ValueError(
            "`per_device_eval_batch_size` must be divisible by `num_generations` so repeated prompt groups stay intact."
        )

    dataset_configs = parse_optional_mapping(args.dataset_configs, "dataset_configs")
    dataset_splits = parse_optional_mapping(args.dataset_splits, "dataset_splits")
    dataset_names = {dataset_name: dataset_name for dataset_name in args.datasets}

    processing_class = load_processing_class(
        args.model,
        args.checkpoint_path,
        processing_class_name=args.processing_class_name,
    )
    if args.processing_class_name is not None:
        # Preserve an explicit processing source's identity and saved template. Restoring
        # the base model name can also change dataset prompt formatting.
        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        custom_chat_template = tokenizer.get_chat_template()
    else:
        maybe_restore_base_model_name(processing_class, args.model, args.checkpoint_path)
        custom_chat_template = maybe_apply_custom_chat_template(processing_class)

    eval_datasets = load_eval_datasets(
        dataset_names,
        dataset_configs,
        dataset_splits,
        tokenizer=processing_class,
        max_random_samples=args.max_samples_per_dataset,
        max_sample_fraction=args.max_samples_fraction,
        random_seed=args.seed,
    )
    dataset_sizes = {name: len(dataset) for name, dataset in eval_datasets.items()}

    resume_generated_cot_examples: list[dict[str, Any]] = []
    precomputed_metrics_totals = None
    resume_reference_datasets = eval_datasets
    if args.resume_generated_cot_examples_json_path is not None:
        if not os.path.exists(args.resume_generated_cot_examples_json_path):
            raise FileNotFoundError(
                "Resume file was requested but not found: "
                f"{args.resume_generated_cot_examples_json_path}"
            )

        resume_generated_cot_examples = load_generated_cot_examples(args.resume_generated_cot_examples_json_path)
        resume_plan = prepare_generated_cot_resume(resume_generated_cot_examples, resume_reference_datasets)
        resume_generated_cot_examples = resume_plan["ordered_examples"]
        print(
            json.dumps(
                {
                    "resume_generated_cot_examples_json_path": args.resume_generated_cot_examples_json_path,
                    "resumed_example_count": len(resume_generated_cot_examples),
                    "missing_example_count": resume_plan["missing_example_count"],
                    "missing_examples_by_dataset": {
                        name: len(indices)
                        for name, indices in resume_plan["missing_indices_by_dataset"].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        if resume_generated_cot_examples:
            precomputed_metrics_totals = rebuild_eval_state_from_generated_examples(
                resume_generated_cot_examples,
                processing_class,
                dont_consider_truncated_cot_in_length_count=args.dont_consider_truncated_cot_in_length_count,
            )
        if resume_plan["missing_example_count"] == 0:
            if len(args.datasets) != 1:
                raise ValueError(
                    "Resume reconstruction without any remaining examples currently supports exactly one dataset."
                )
            metrics = metrics_from_eval_totals(
                precomputed_metrics_totals,
                dataset_name=args.datasets[0],
                include_shortest_completion_accuracy=args.num_generations > 1,
            )
            metrics = add_aes_metrics(metrics, DEFAULT_AES_BASELINE_METRICS)
            generated_cot_examples = resume_generated_cot_examples
            payload = {
                "model": args.model,
                "checkpoint_path": args.checkpoint_path,
                "datasets": args.datasets,
                "dataset_sizes": dataset_sizes,
                "evaluation": {
                    "seed": args.seed,
                    "do_sample": args.do_sample,
                    "eval_temperature": args.temperature if args.do_sample else 0.0,
                    "top_p_eval": args.top_p,
                    "presence_penalty": args.presence_penalty,
                    "max_completion_length": args.max_completion_length,
                    "dont_consider_truncated_cot_in_length_count": args.dont_consider_truncated_cot_in_length_count,
                    "num_generations_eval": args.num_generations,
                    "eval_sampling_strategy": args.eval_sampling_strategy,
                    "max_samples_per_dataset": args.max_samples_per_dataset,
                    "max_samples_fraction": args.max_samples_fraction,
                    "vllm_max_model_length": args.vllm_max_model_length,
                },
                "metrics": metrics,
                "generated_cot_examples_count": len(generated_cot_examples),
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))

            if args.save_generated_cot_examples_json_path is not None:
                output_dir = os.path.dirname(args.save_generated_cot_examples_json_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(args.save_generated_cot_examples_json_path, "w") as f:
                    json.dump(generated_cot_examples, f, indent=2, sort_keys=True, default=str)
            if args.save_metrics_path is not None:
                output_dir = os.path.dirname(args.save_metrics_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(args.save_metrics_path, "w") as f:
                    json.dump(payload, f, indent=2, sort_keys=True, default=str)
            return

        eval_datasets = filter_eval_datasets_to_missing_examples(
            eval_datasets,
            resume_plan["missing_indices_by_dataset"],
        )

    model_init_kwargs = {
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.use_vllm:
        # vLLM keeps its own copy of the model weights on GPU. Keep the trainer-side HF model on CPU during eval so
        # colocate/server vLLM runs do not fail from duplicated GPU residency before generation even starts.
        model_init_kwargs["device_map"] = {"": "cpu"}
    model = load_model_for_eval(args.model, args.checkpoint_path, model_init_kwargs)

    config_num_generations = max(2, args.num_generations)
    generation_batch_size = args.generation_batch_size or max(config_num_generations, args.num_generations)
    report_to = [name.strip() for name in args.report_to.split(",") if name.strip()]
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        # The support model is already loaded, but colocated vLLM also needs this opt-in.
        model_init_kwargs={"trust_remote_code": True} if args.trust_remote_code else None,
        report_to=report_to,
        seed=args.seed,
        do_train=False,
        do_eval=False,
        eval_strategy="no",
        remove_unused_columns=False,
        beta=0.0,
        num_generations=config_num_generations,
        num_generations_eval=args.num_generations,
        eval_sampling_strategy=args.eval_sampling_strategy,
        max_completion_length=args.max_completion_length,
        max_completion_length_eval=args.max_completion_length,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        generation_batch_size=generation_batch_size,
        dataloader_drop_last=False,
        temperature=args.temperature,
        eval_temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p,
        top_p_eval=args.top_p,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_model_impl=args.vllm_model_impl,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        ddp_timeout=args.ddp_timeout,
        generation_kwargs=(
            {"presence_penalty": args.presence_penalty} if args.use_vllm else {"do_sample": args.do_sample}
        ),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward],
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_datasets,
        processing_class=processing_class,
    )
    trainer._eval_generation_examples = list(resume_generated_cot_examples)
    trainer._eval_metrics_totals = dict(precomputed_metrics_totals) if precomputed_metrics_totals is not None else None
    trainer._save_generated_cot_examples_json_path = args.save_generated_cot_examples_json_path
    trainer._save_generated_cot_examples_every_batch = args.save_generated_cot_examples_every_batch
    trainer._save_rendered_prompts = args.save_rendered_prompts
    if custom_chat_template is not None:
        trainer.chat_template = custom_chat_template
        if getattr(trainer, "vllm_generation", None) is not None:
            trainer.vllm_generation.chat_template = custom_chat_template
    if args.use_vllm:
        # `Trainer.evaluate()` would otherwise call `accelerator.prepare_model(..., evaluation_mode=True)` and move
        # the HF model onto CUDA even though vLLM already owns the GPU copy of the weights. Preparing it here with
        # device placement disabled keeps the support model on CPU and prevents evaluation-time OOMs.
        trainer.model = trainer.accelerator.prepare_model(
            trainer.model,
            device_placement=False,
            evaluation_mode=True,
        )
        trainer.model_wrapped = trainer.model

    metrics = trainer.evaluate()
    if not trainer.accelerator.is_main_process:
        return

    # Rebuild final metrics from the first generation group for each question.
    question_to_keys = defaultdict(list)
    unique_by_dataset = {name: {} for name in resume_reference_datasets}
    for name, dataset in resume_reference_datasets.items():
        for index, question in enumerate(dict.fromkeys(example["question"] for example in dataset)):
            question_to_keys[question].append((name, index))
    for example in trainer._eval_generation_examples:
        name, _ = _coerce_resume_record_key(example, question_to_keys)
        unique_by_dataset[name].setdefault(example["question"], example)

    generated_cot_examples = []
    for name, unique_examples in unique_by_dataset.items():
        examples = list(unique_examples.values())
        dataset_sizes[name] = len(examples)
        if not examples:
            continue
        totals = rebuild_eval_state_from_generated_examples(
            examples,
            processing_class,
            dont_consider_truncated_cot_in_length_count=args.dont_consider_truncated_cot_in_length_count,
        )
        unique_metrics = metrics_from_eval_totals(
            totals,
            dataset_name=name,
            include_shortest_completion_accuracy=args.num_generations > 1,
        )
        metrics.update({
            key: value for key, value in unique_metrics.items()
            if not key.endswith(("_runtime", "_samples_per_second", "_steps_per_second"))
        })
        generated_cot_examples.extend(examples)
    metrics = add_aes_metrics(metrics, DEFAULT_AES_BASELINE_METRICS)
    payload = {
        "model": args.model,
        "checkpoint_path": args.checkpoint_path,
        "datasets": args.datasets,
        "dataset_sizes": dataset_sizes,
        "evaluation": {
            "seed": args.seed,
            "do_sample": args.do_sample,
            "eval_temperature": args.temperature if args.do_sample else 0.0,
            "top_p_eval": args.top_p,
            "presence_penalty": args.presence_penalty,
            "max_completion_length": args.max_completion_length,
            "dont_consider_truncated_cot_in_length_count": args.dont_consider_truncated_cot_in_length_count,
            "num_generations_eval": args.num_generations,
            "eval_sampling_strategy": args.eval_sampling_strategy,
            "max_samples_per_dataset": args.max_samples_per_dataset,
            "max_samples_fraction": args.max_samples_fraction,
            "vllm_max_model_length": args.vllm_max_model_length,
            "save_rendered_prompts": args.save_rendered_prompts,
        },
        "metrics": metrics,
        "generated_cot_examples_count": len(generated_cot_examples),
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


    if args.save_generated_cot_examples_json_path is not None and trainer.accelerator.is_main_process:
        output_dir = os.path.dirname(args.save_generated_cot_examples_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.save_generated_cot_examples_json_path, "w") as f:
            json.dump(generated_cot_examples, f, indent=2, sort_keys=True, default=str)

    if args.save_metrics_path is not None and trainer.accelerator.is_main_process:
        output_dir = os.path.dirname(args.save_metrics_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.save_metrics_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
