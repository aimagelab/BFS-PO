import argparse
import os
import socket
from dataclasses import fields
from pathlib import Path

import yaml
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.extras.dataset_formatting import resolve_reasoning_dataset
from trl.extras.prompt_variations import maybe_apply_custom_chat_template
from trl.rewards import accuracy_reward, get_soft_overlong_punishment


def load_yaml_config(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def parse_override_value(raw_value):
    return yaml.safe_load(raw_value)


def apply_cli_overrides(override_args, grpo_config_dict, trainer_config):
    grpo_config_keys = {field.name for field in fields(GRPOConfig)}
    trainer_config_keys = set(trainer_config)

    for raw_arg in override_args:
        arg = raw_arg[2:] if raw_arg.startswith("--") else raw_arg
        if "=" not in arg:
            raise ValueError(
                f"Unsupported CLI override '{raw_arg}'. Use the form key=value, for example "
                "'train_dataset_name=math-500'."
            )

        key, raw_value = arg.split("=", 1)
        if not key:
            raise ValueError(f"Invalid CLI override '{raw_arg}': missing key before '='.")

        in_grpo_config = key in grpo_config_keys
        in_trainer_config = key in trainer_config_keys

        if in_grpo_config and in_trainer_config:
            raise ValueError(
                f"Ambiguous CLI override '{key}': it exists in both GRPOConfig and GRPOTrainer config."
            )
        if not in_grpo_config and not in_trainer_config:
            raise ValueError(
                f"Unknown CLI override '{key}'. The key was not found in GRPOConfig.yaml or "
                "GRPOTrainer.yaml."
            )

        parsed_value = parse_override_value(raw_value)
        if in_grpo_config:
            grpo_config_dict[key] = parsed_value
        else:
            trainer_config[key] = parsed_value


def resolve_reward_funcs(names, max_completion_length, soft_punishment_ratio=4 / 5):
    if not isinstance(names, list):
        names = [names]

    soft_overlong_punishment = None
    if "get_soft_overlong_punishment" in names:
        if max_completion_length is None:
            raise ValueError(
                "`max_completion_length` must be set when `get_soft_overlong_punishment` is configured."
            )
        if not 0 < soft_punishment_ratio <= 1:
            raise ValueError("`soft_punishment_ratio` must be greater than 0 and at most 1.")
        soft_overlong_punishment = get_soft_overlong_punishment(
            max_completion_len=soft_punishment_ratio * max_completion_length,
            soft_punish_cache=0.2 * max_completion_length,
        )

    reward_funcs = []
    for name in names:
        if name == "accuracy_reward":
            reward_funcs.append(accuracy_reward)
        elif name == "get_soft_overlong_punishment":
            reward_funcs.append(soft_overlong_punishment)
        else:
            raise ValueError(f"Unsupported reward function in trainer config: {name}")

    return reward_funcs


def ensure_supported_trainer_value(name, value):
    if value not in (None, [None, None]):
        raise ValueError(f"Unsupported non-null value for '{name}' in GRPOTrainer_config.yaml")


def resolve_peft_config(peft_config):
    if peft_config is None:
        return None
    if not isinstance(peft_config, dict):
        return peft_config

    try:
        from peft import LoraConfig
    except ImportError as e:
        raise ImportError("`peft` must be installed to use `peft_config` in main_grpo.py") from e

    return LoraConfig(**peft_config)


def load_reasoning_dataset(
    dataset_name,
    dataset_config,
    dataset_split,
    tokenizer=None,
):
    if dataset_name is None:
        raise ValueError("Dataset name cannot be None.")

    with PartialState().main_process_first():
        dataset_spec = resolve_reasoning_dataset(dataset_name, dataset_config, dataset_split, tokenizer=tokenizer)
        dataset = load_dataset(
            dataset_spec["dataset_name"],
            dataset_spec["dataset_config"],
            split=dataset_spec["dataset_split"],
        )

        preprocessing_fn = dataset_spec["preprocessing_fn"]
        if preprocessing_fn is not None:
            dataset = dataset.map(
                preprocessing_fn,
                load_from_cache_file=dataset_spec["load_from_cache_file"],
            )

    return dataset


def resolve_dataset_field(field_name, field_value, dataset_key):
    if isinstance(field_value, dict):
        if dataset_key not in field_value:
            raise ValueError(f"Missing '{field_name}[{dataset_key}]' for eval dataset '{dataset_key}'.")
        return field_value[dataset_key]
    return field_value


def load_eval_datasets(eval_dataset_name, eval_dataset_config, eval_dataset_split, tokenizer=None):
    if eval_dataset_name is None:
        return None

    if isinstance(eval_dataset_name, dict):
        eval_datasets = {}
        for dataset_key, dataset_name in eval_dataset_name.items():
            dataset_config = resolve_dataset_field("eval_dataset_config", eval_dataset_config, dataset_key)
            dataset_split = resolve_dataset_field("eval_dataset_split", eval_dataset_split, dataset_key)
            eval_datasets[dataset_key] = load_reasoning_dataset(
                dataset_name,
                dataset_config,
                dataset_split,
                tokenizer=tokenizer,
            )
        return eval_datasets

    return load_reasoning_dataset(eval_dataset_name, eval_dataset_config, eval_dataset_split, tokenizer=tokenizer)


def main():
    config_dir = Path(__file__).resolve().parent / "trl" / "config"
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo_config_path", type=str, default=str(config_dir / "GRPOConfig.yaml"))
    parser.add_argument("--grpo_trainer_config_path", type=str, default=str(config_dir / "GRPOTrainer.yaml"))
    parser.add_argument("--report_to", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    custom_args, override_args = parser.parse_known_args()

    grpo_config_dict = load_yaml_config(custom_args.grpo_config_path)
    trainer_config = load_yaml_config(custom_args.grpo_trainer_config_path)
    trainer_config.setdefault("soft_punishment_ratio", 4 / 5)
    apply_cli_overrides(override_args, grpo_config_dict, trainer_config)

    processing_class = None
    if trainer_config["processing_class_name"] is not None:
        processing_class = AutoProcessor.from_pretrained(trainer_config["processing_class_name"])
    else:
        try:
            processing_class = AutoProcessor.from_pretrained(trainer_config["model"])
        except Exception:
            processing_class = AutoTokenizer.from_pretrained(trainer_config["model"])
    custom_chat_template = maybe_apply_custom_chat_template(processing_class)

    train_dataset_name = trainer_config["train_dataset_name"]
    train_dataset = load_reasoning_dataset(
        train_dataset_name,
        trainer_config["train_dataset_config"],
        trainer_config["train_dataset_split"],
        tokenizer=processing_class,
    )
    eval_dataset = load_eval_datasets(
        trainer_config["eval_dataset_name"],
        trainer_config["eval_dataset_config"],
        trainer_config["eval_dataset_split"],
        tokenizer=processing_class,
    )

    training_args = GRPOConfig(**grpo_config_dict)
    reward_funcs = resolve_reward_funcs(
        trainer_config["reward_funcs"],
        training_args.max_completion_length,
        soft_punishment_ratio=trainer_config["soft_punishment_ratio"],
    )

    reward_processing_classes = None
    if trainer_config["reward_processing_class_name"] is not None:
        reward_processing_class = AutoTokenizer.from_pretrained(trainer_config["reward_processing_class_name"])
        reward_processing_classes = [reward_processing_class] * len(reward_funcs)

    trainer = GRPOTrainer(
        model=trainer_config["model"],
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        reward_processing_classes=reward_processing_classes,
        callbacks=trainer_config["callbacks"],
        optimizers=tuple(trainer_config["optimizers"]),
        peft_config=resolve_peft_config(trainer_config["peft_config"]),
        tools=trainer_config["tools"],
        rollout_func=trainer_config["rollout_func"],
        environment_factory=trainer_config["environment_factory"],
    )
    if custom_chat_template is not None:
        trainer.chat_template = custom_chat_template
        if getattr(trainer, "vllm_generation", None) is not None:
            trainer.vllm_generation.chat_template = custom_chat_template

    trainer.train(resume_from_checkpoint=custom_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
