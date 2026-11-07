# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import atexit
import copy
import importlib.resources as pkg_resources
import inspect
import json
import os
import sys
import textwrap
import time
import warnings
import math
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import datasets
import pandas as pd
import torch
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from huggingface_hub import CommitScheduler, DatasetCard, DatasetCardData, create_repo
from packaging.version import Version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available

from ..chat_template_utils import add_response_schema, get_training_chat_template, parse_response
from ..data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from ..extras.dataset_formatting import _extract_prompt_question
from ..extras.profiling import profiling_context, profiling_decorator
from ..generation.vllm_generation import VLLMGeneration
from ..import_utils import is_jmespath_available, is_liger_kernel_available
from ..models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ..models.utils import _ForwardRedirection, disable_gradient_checkpointing
from .base_trainer import BaseTrainer
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    shutdown_event_loop_in_daemon,
    split_pixel_values_by_grid,
    split_tensor_dict,
    start_event_loop_in_daemon,
    unsplit_pixel_values_by_grid,
    use_adapter,
)

from ..rewards import accuracy_reward
from ..rewards.branch_advantages import compute_branch_advantages
from ..rewards.verifier import extract_last_single_answer, extract_math_answer, is_multiple_choice_option_answer

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss


if is_wandb_available():
    import wandb

if is_trackio_available():
    import trackio

logger = get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]

# What we call a rollout function is a callable that takes prompts (list) and the trainer instance as parameters and
# returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and "logprobs"
# fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], "GRPOTrainer"], dict[str, Any]]


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]

_DYNAMIC_SAMPLING_PLACEHOLDER_KEY = "__trl_dynamic_sampling_placeholder__"


class _ReuseIteratorDataLoader:
    """Wrap a dataloader so GRPO can reuse one raw generation batch across multiple optimization steps."""

    def __init__(self, dataloader, trainer: "GRPOTrainer"):
        self.dataloader = dataloader
        self.trainer = trainer
        self._iterator = None
        self._cached_batch = None
        self._yielded_dynamic_sampling_batch = False

    def __iter__(self):
        self._iterator = iter(self.dataloader)
        self._cached_batch = None
        self._yielded_dynamic_sampling_batch = False
        self.trainer._active_train_batch_iterator = self._iterator
        return self

    def __next__(self):
        if self._iterator is None:
            self.__iter__()

        if self.trainer.dynamic_sampling:
            # `Trainer.get_batch_samples()` may prefetch multiple dataloader entries before `_prepare_inputs()`
            # consumes the ready batch. Avoid triggering a second generation while one dynamic batch is already ready.
            if self.trainer._needs_new_generation_batch() and self.trainer._ready_dynamic_sampling_batch is None:
                # If the previous yielded placeholder was skipped (for example when resuming), drop the stale batch.
                self.trainer._ready_dynamic_sampling_batch = None
                self.trainer._ready_dynamic_sampling_batch = self.trainer._get_dynamic_sampling_batch()
                if self.trainer._ready_dynamic_sampling_batch is None:
                    if not self._yielded_dynamic_sampling_batch:
                        self.trainer._dynamic_sampling_waiting_for_next_epoch = True
                    raise StopIteration
            if self._cached_batch is None:
                self._cached_batch = {_DYNAMIC_SAMPLING_PLACEHOLDER_KEY: True}
            self._yielded_dynamic_sampling_batch = True
            return self._cached_batch

        if self._cached_batch is None or self.trainer._needs_new_generation_batch():
            self._cached_batch = next(self._iterator)
        return self._cached_batch

    def __len__(self):
        generate_every = self.trainer.args.steps_per_generation * self.trainer.num_iterations
        return len(self.dataloader) * generate_every

    def __getattr__(self, name):
        return getattr(self.dataloader, name)


class _DynamicSamplingEpochCallback(TrainerCallback):
    def __init__(self, trainer: "GRPOTrainer"):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer._dynamic_sampling_waiting_for_next_epoch:
            control.should_training_stop = False
            control.should_epoch_stop = False
            self.trainer._dynamic_sampling_waiting_for_next_epoch = False
        return control


class GRPOTrainer(BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from trl import GRPOTrainer
    from trl.rewards import accuracy_reward
    from trl.extras.dataset_formatting import preprocess_gsm8k
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split="train").map(preprocess_gsm8k)

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                   functions can be either synchronous or asynchronous and can also return `None` when the reward is
                   not applicable to those samples. This is useful for multi-task training where different reward
                   functions apply to different types of samples. When a reward function returns `None` for a sample,
                   that reward function is excluded from the reward calculation for that sample. For more details, see
                   [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions (sync or async) that the model can invoke during generation. Each tool
            should be a standard Python function with properly type-hinted arguments and return values, and a
            Google-style docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It receives the list of prompts allocated to the current
            process and the trainer instance. It must return a dict with `"prompt_ids"`, `"completion_ids"`, and
            `"logprobs"` fields. Any other fields are forwarded to the reward functions. This feature is experimental
            and may change or be removed at any time without prior notice.
        environment_factory (`EnvironmentFactory`, *optional*):
            A callable that creates and returns an environment instance. The environment class should define methods
            that can be invoked as tools during generation. Each method should comply with the same requirements as the
            `tools` described above. If `environment_factory` is provided, an instance of the environment is created
            for each generation in the batch, allowing for parallel and independent interactions. The environment must
            also implement a callable `reset` method that can be used to reset state between generations. The `reset`
            method should return either `None` or a string: when it returns a string, that string is appended to the
            last user message before generation. This feature is experimental and may change or be removed at any time
            without prior notice.
    """

    _tag_names = ["trl", "grpo"]
    _name = "GRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will not be used to initialize the supplied model; "
                    "`trust_remote_code` is still forwarded to colocated vLLM."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), truncation_side="left", padding_side="left"
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )

        if is_peft_available() and is_peft_model(model) and args.beta != 0.0:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during GRPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        self.negative_rewards = args.negative_rewards

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'rollout_func', which is an experimental feature. This API may change or be removed at "
                "any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func
        if environment_factory is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'environment_factory', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )

        # Tools
        if tools:
            if not Version(transformers.__version__) >= Version("5.0.0"):
                raise ImportError(
                    "Using tools with GRPOTrainer requires transformers version 5.0.0 or higher. Please upgrade "
                    "transformers with `pip install --upgrade transformers` to use this feature."
                )
        if environment_factory:
            if not Version(transformers.__version__) >= Version("5.2.0"):
                raise ImportError(
                    "Using `environment_factory` with GRPOTrainer requires transformers version 5.2.0 or higher. "
                    "Please install transformers from the main branch with `pip install "
                    "git+https://github.com/huggingface/transformers.git@main` to use this feature."
                )
        if tools or environment_factory:
            if not is_jmespath_available():
                raise ImportError(
                    "Using tools with GRPOTrainer requires the jmespath library for response parsing. Please install "
                    "it with `pip install jmespath` to use this feature."
                )

        # Create the environments and extract their methods to be used as tools. We create one environment per rollout
        generation_batch_size = args.generation_batch_size // args.world_size
        if environment_factory is not None:
            self.environments = [environment_factory() for _ in range(generation_batch_size)]
            environment_methods = [[] for _ in range(generation_batch_size)]
            for i, environment in enumerate(self.environments):
                has_reset = False
                for name, member in inspect.getmembers(environment, predicate=inspect.ismethod):
                    if name == "reset":
                        has_reset = True
                    elif not name.startswith("_"):
                        environment_methods[i].append(member)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset` "
                    )
        else:
            self.environments = None

        tools = tools or []
        self._sync_tool_dicts = [{} for _ in range(generation_batch_size)]
        self._async_tool_dicts = [{} for _ in range(generation_batch_size)]
        for i in range(generation_batch_size):
            for tool in tools + (environment_methods[i] if self.environments is not None else []):
                if asyncio.iscoroutinefunction(tool):
                    self._async_tool_dicts[i][tool.__name__] = tool
                else:
                    self._sync_tool_dicts[i][tool.__name__] = tool

        self.tools = tools + (environment_methods[0] if self.environments is not None else [])

        # Check for async functions to start an event loop on a daemon thread
        self._has_async_funcs = any(asyncio.iscoroutinefunction(func) for func in self.reward_funcs + self.tools)

        if self._has_async_funcs:
            self.async_loop_thread, self.async_loop, self.async_loop_ready_event = start_event_loop_in_daemon(
                name="GRPOTrainer-AsyncLoop"
            )
            # wait until the event loop is running in the daemon thread
            self.async_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_loop_thread, self.async_loop)

        # At the time of initial implementation, most tokenizers do not have built-in support for response schemas.
        # While waiting for broader adoption, we provide this utility function to manually set the response schema for
        # known chat templates.
        # We need `getattr`` until the base class sets a default None value for response_schema
        if self.tools and not getattr(processing_class, "response_schema", None):
            processing_class = add_response_schema(processing_class)
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools:
            self.chat_template = get_training_chat_template(processing_class)
        else:
            self.chat_template = None

        # Training arguments
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.max_completion_length_eval = (
            self.max_completion_length if args.max_completion_length_eval is None else args.max_completion_length_eval
        )
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_tool_calling_iterations = args.max_tool_calling_iterations or sys.maxsize
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.eval_sampling_strategy = args.eval_sampling_strategy
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.eval_temperature = args.eval_temperature
        self.top_p = args.top_p
        self.top_p_eval = args.top_p_eval
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.compute_importance_sampling_specific_branch = args.compute_importance_sampling_specific_branch
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_kernel = args.use_liger_kernel
        self.loss_type = args.loss_type
        self.multi_objective_aggregation = args.multi_objective_aggregation
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.off_policy_mask_threshold = args.off_policy_mask_threshold
        self.bfs_po_k = args.bfs_po_k
        self.branch_advantage = args.branch_advantage

        if self.eval_sampling_strategy == "bfs_po" and not self.use_vllm:
            raise ValueError("`eval_sampling_strategy='bfs_po'` currently requires `use_vllm=True`.")
        if self.branch_advantage and self.bfs_po_k <= 0:
            warnings.warn(
                "`branch_advantage=True` requires `bfs_po_k > 0`. "
                "Falling back to standard sequence-level advantages for this run.",
                stacklevel=2,
            )
            self.branch_advantage = False
            
        self.dynamic_sampling = args.dynamic_sampling
        self.global_train_batch_size = args.global_train_batch_size
        if self.use_liger_kernel and self.off_policy_mask_threshold is not None:
            raise ValueError("Liger kernel does not support off-policy sequence masking yet.")
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_kernel and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        if args.loss_type == "luspo" and args.importance_sampling_level != "sequence":
            logger.warning(
                "When using `'luspo'` loss, `importance_sampling_level` should be set to `'sequence'` to mirror the "
                "paper's setup."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None
        self._active_train_batch_iterator = None
        self._dynamic_sampling_pending_buffer = None
        self._ready_dynamic_sampling_batch = None
        self._dynamic_sampling_waiting_for_next_epoch = False
        self._dynamic_sampling_shuffle_count = 0

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )
        if self.dynamic_sampling:
            self.add_callback(_DynamicSamplingEpochCallback(self))

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(cast_outputs_to_original_dtype)

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Liger loss
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._eval_metrics_totals = None
        self._eval_zero_shot_metrics = {}
        self._eval_generation_examples = []
        self._save_generated_cot_examples_json_path = None
        self._save_generated_cot_examples_every_batch = False
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # Initialize vLLM generation backend
            # Wrap rollout_func to capture trainer context if provided
            rollout_func = None
            if self.rollout_func is not None:

                def rollout_func(prompts):
                    return self.rollout_func(prompts, self)

            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                # max_num_seqs=args.per_device_train_batch_size
                # * args.vllm_tensor_parallel_size
                # * args.steps_per_generation,
                max_num_seqs=args.generation_batch_size // args.world_size * args.vllm_tensor_parallel_size,
                max_num_batched_tokens=args.vllm_max_num_batched_tokens,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                enable_log_stats=args.vllm_enable_log_stats,
                model_impl=args.vllm_model_impl,
                trust_remote_code=(args.model_init_kwargs or {}).get("trust_remote_code", False),
                seed=args.seed,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                eval_temperature=self.eval_temperature,
                top_p=self.top_p,
                top_p_eval=self.top_p_eval,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                max_completion_length_eval=self.max_completion_length_eval,
                generation_kwargs=args.generation_kwargs,
                # Chat/tool configuration
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                tools=self.tools,
                rollout_func=rollout_func,
            )
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.beta == 0.0:
                raise ValueError(
                    "You passed `sync_ref_model=True` while `beta=0.0`, which means the reference model is not used "
                    "during training. Consequently, GRPOTrainer does not create a `ref_model` instance, and there is "
                    "nothing to synchronize. Please set `sync_ref_model=False`, or set `beta` to a non-zero value."
                )
            if is_peft_model(model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, GRPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(repo_id, private=self.args.hub_private_repo, repo_type="dataset", exist_ok=True)
                template_path = pkg_resources.files("trl").joinpath("templates/completions_dataset_card.md")
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size`), our dataloader loads a local
    # *generation* batch. Without dynamic sampling, this generation batch is also the batch used to build the buffered
    # training slices. With dynamic sampling, the local generation batch size stays tied to `generation_batch_size`,
    # while the number of buffered training slices is driven by `global_train_batch_size`.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self.args.generation_batch_size // self.accelerator.num_processes,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        if self.dynamic_sampling:
            return _ReuseIteratorDataLoader(dataloader, self)
        return dataloader

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        repeat_count = 1 if self.dynamic_sampling else self.num_iterations * self.args.steps_per_generation
        sampler_seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=repeat_count,
            shuffle=self.shuffle_dataset,
            seed=sampler_seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        sampler_seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=sampler_seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output


    def _eval_sampling_multiplier(self) -> int:
        return self.bfs_po_k + 1 if getattr(self, "eval_sampling_strategy", "independent") == "bfs_po" else 1


    @staticmethod
    def _normalize_answer_key(answer_key: Any) -> tuple[str, ...]:
        if answer_key is None:
            return tuple()

        normalized_parts: list[str] = []
        seen_parts: set[str] = set()

        def _visit(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    _visit(item)
                return

            part = "" if value is None else str(value)
            if part in seen_parts:
                return
            normalized_parts.append(part)
            seen_parts.add(part)

        _visit(answer_key)
        return tuple(normalized_parts)

    @staticmethod
    def _extract_answer_key(
        question: str,
        completion_text: str,
        *,
        solution: str | None = None,
        complete_answer: str | None = None,
    ) -> tuple[str, ...]:
        if completion_text == "":
            return ("",)
        if is_multiple_choice_option_answer(solution=solution, complete_answer=complete_answer):
            return GRPOTrainer._normalize_answer_key(extract_last_single_answer(question, completion_text))
        return GRPOTrainer._normalize_answer_key(extract_math_answer(question, completion_text))

    @staticmethod
    def _split_eval_completion_ids(
        completion_ids: list[int],
        think_token_id: int,
        end_think_token_id: int,
    ) -> tuple[list[int], list[int]]:
        try:
            cot_start = completion_ids.index(think_token_id) + 1
        except ValueError:
            cot_start = 0

        try:
            cot_end = completion_ids.index(end_think_token_id, cot_start)
        except ValueError:
            return [], []

        return completion_ids[cot_start:cot_end], completion_ids[cot_end + 1 :]


    @staticmethod
    def _select_shortest_completion_index(completion_lengths: list[float]) -> int:
        if not completion_lengths:
            raise ValueError("Cannot select the shortest completion from an empty length list.")

        best_index = 0
        best_length = completion_lengths[0]
        for index, length in enumerate(completion_lengths[1:], start=1):
            if length < best_length:
                best_index = index
                best_length = length

        return best_index


    def _select_bfs_po_expansion_token_index(
        self, answer_info: dict[str, Any], end_think_token_id: int
    ) -> int:
        entropies = answer_info["entropy"]
        completion_ids = answer_info["completion_ids"]


        max_index = None
        try:
            end = completion_ids.index(end_think_token_id, 0)
        except ValueError:
            end = len(entropies)
        if 0 < end:
            max_index = max(range(0, end), key=entropies.__getitem__)

        if max_index is None:
            max_index = max(range(len(entropies)), key=entropies.__getitem__)

        return max_index

    def _record_eval_metrics_only(
        self,
        inputs: list[dict[str, torch.Tensor | Any]],
        completion_ids_list: list[list[int]],
        tool_mask_list: list[list[int]] | None,
        extra_fields: dict[str, Any] | None,
        prompt_ids_list: list[list[int]] | None = None,
    ) -> None:
        """Keep eval bookkeeping limited to accuracy and CoT/completion length."""
        num_generations = self.num_generations_eval
        sampling_multiplier = self._eval_sampling_multiplier()
        total_num_generations = num_generations * sampling_multiplier
        if num_generations <= 0:
            raise ValueError("`num_generations_eval` must be strictly positive during evaluation.")
        if total_num_generations <= 0:
            raise ValueError("The total number of evaluation candidates per prompt must be strictly positive.")
        if len(inputs) % num_generations != 0:
            raise ValueError(
                "Eval batches must contain a whole number of prompt groups. "
                f"Received batch size {len(inputs)} with num_generations_eval={num_generations}."
            )
        if len(completion_ids_list) != (len(inputs) // num_generations) * total_num_generations:
            raise ValueError(
                "Eval completion count does not match the configured sampling strategy. "
                f"Got {len(completion_ids_list)} completions for {len(inputs)} repeated inputs, "
                f"num_generations_eval={num_generations}, and eval_sampling_multiplier={sampling_multiplier}."
            )

        local_num_prompts = len(inputs) // num_generations
        prompt_level_inputs = inputs[::num_generations]
        device = self.accelerator.device

        think_token_id, end_think_token_id = self._get_think_token_ids()
        cot_ids_list = []
        completion_text_ids_list = []
        for ids in completion_ids_list:
            cot_ids, answer_ids = self._split_eval_completion_ids(ids, think_token_id, end_think_token_id)
            cot_ids_list.append(cot_ids)
            completion_text_ids_list.append(answer_ids)

        eos_and_pad = {
            token_id for token_id in (getattr(self, "eos_token_id", None), getattr(self, "pad_token_id", None)) if token_id is not None
        }
        if eos_and_pad:
            is_truncated_list = [bool(ids) and ids[-1] not in eos_and_pad for ids in completion_ids_list]
        else:
            is_truncated_list = [False for _ in completion_ids_list]

        max_eval_completion_length = getattr(
            self,
            "max_completion_length_eval",
            getattr(self, "max_completion_length", None),
        )
        # If a completion never closes `</think>`, we treat the CoT length as missing rather than inflating it to the
        # truncation budget.
        resolved_cot_lengths = [len(cot_ids) for cot_ids in cot_ids_list]

        if tool_mask_list is not None:
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask_list], dtype=torch.float32, device=device)
        else:
            completion_lengths = torch.tensor(resolved_cot_lengths, dtype=torch.float32, device=device)
        completion_lengths = completion_lengths.view(local_num_prompts, total_num_generations)
        gathered_completion_lengths = self.accelerator.gather_for_metrics(completion_lengths)
        completion_length_sum = gathered_completion_lengths.sum().item()
        # completion_length_count = gathered_completion_lengths.numel()
        # extracot puts 0 to truncated thinking reasoning traces and excludes from the average
        completion_length_count = torch.sum(gathered_completion_lengths != 0).item()
        if extra_fields is not None and "new_token_counts" in extra_fields:
            new_token_counts = extra_fields["new_token_counts"]
            if len(new_token_counts) != len(completion_ids_list):
                raise ValueError(
                    "Eval new-token metadata does not align with the generated completions. "
                    f"Got {len(new_token_counts)} counts for {len(completion_ids_list)} completions."
                )
            new_completion_lengths = torch.tensor(new_token_counts, dtype=torch.float32, device=device)
        elif tool_mask_list is not None:
            new_completion_lengths = torch.tensor(
                [sum(mask) for mask in tool_mask_list],
                dtype=torch.float32,
                device=device,
            )
        else:
            new_completion_lengths = torch.tensor(
                [len(ids) for ids in completion_ids_list],
                dtype=torch.float32,
                device=device,
            )
        new_completion_lengths = new_completion_lengths.view(local_num_prompts, total_num_generations)
        gathered_new_completion_lengths = self.accelerator.gather_for_metrics(new_completion_lengths)
        batch_new_token_sum = gathered_new_completion_lengths.sum().item()
        batch_new_token_count = gathered_new_completion_lengths.numel()
        batch_prompt_count = gathered_new_completion_lengths.shape[0]
        full_completion_texts = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
        cot_texts = self.processing_class.batch_decode(cot_ids_list, skip_special_tokens=True)
        completion_texts = self.processing_class.batch_decode(completion_text_ids_list, skip_special_tokens=True)
        rendered_prompt_texts = None
        if getattr(self, "_save_rendered_prompts", False):
            if prompt_ids_list is None or len(prompt_ids_list) != len(completion_ids_list):
                raise ValueError(
                    "Saving rendered prompts requires one prompt token sequence per generated completion. "
                    f"Got {None if prompt_ids_list is None else len(prompt_ids_list)} prompts for "
                    f"{len(completion_ids_list)} completions."
                )
            rendered_prompt_texts = [
                self.processing_class.decode(
                    prompt_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for prompt_ids in prompt_ids_list
            ]
        grouped_is_truncated = [
            is_truncated_list[i * total_num_generations : (i + 1) * total_num_generations]
            for i in range(local_num_prompts)
        ]
        repeated_prompt_level_inputs = [example for example in prompt_level_inputs for _ in range(total_num_generations)]

        accuracies = accuracy_reward(
            [[{"role": "assistant", "content": text}] for text in completion_texts],
            complete_answers=[example["answer"] for example in repeated_prompt_level_inputs],
            question=[example["question"] for example in repeated_prompt_level_inputs],
            solution=[example["solution"] for example in repeated_prompt_level_inputs],
        )

        accuracies = torch.tensor(accuracies, dtype=torch.float32, device=device).view(
            local_num_prompts, total_num_generations
        )
        gathered_accuracies = self.accelerator.gather_for_metrics(accuracies)
        accuracy_sum = gathered_accuracies.sum().item()
        accuracy_count = gathered_accuracies.numel()

        shortest_completion_accuracy_sum = None
        shortest_completion_accuracy_count = None
        if total_num_generations > 1:
            shortest_completion_indices = torch.tensor(
                [
                    self._select_shortest_completion_index(prompt_completion_lengths)
                    for prompt_completion_lengths in new_completion_lengths.tolist()
                ],
                dtype=torch.long,
                device=device,
            )
            shortest_completion_accuracies = accuracies[
                torch.arange(local_num_prompts, device=device),
                shortest_completion_indices,
            ]
            gathered_shortest_completion_accuracies = self.accelerator.gather_for_metrics(
                shortest_completion_accuracies
            )
            shortest_completion_accuracy_sum = gathered_shortest_completion_accuracies.sum().item()
            shortest_completion_accuracy_count = gathered_shortest_completion_accuracies.numel()

        prompt_expansion_metadata = None
        if extra_fields is not None and "expansion" in extra_fields:
            expansions = extra_fields["expansion"]
            if len(expansions) == len(completion_ids_list):
                prompt_expansion_metadata = [expansions[i * total_num_generations] for i in range(local_num_prompts)]
            elif len(expansions) == local_num_prompts:
                prompt_expansion_metadata = list(expansions)
            else:
                raise ValueError(
                    "Eval expansion metadata does not align with the generated completions. "
                    f"Got {len(expansions)} expansion entries for {len(completion_ids_list)} completions "
                    f"and {local_num_prompts} prompt groups."
                )

        grouped_cot_lengths = completion_lengths.tolist()
        grouped_cot_texts = [
            cot_texts[i * total_num_generations : (i + 1) * total_num_generations]
            for i in range(local_num_prompts)
        ]
        grouped_completion_texts = [
            completion_texts[i * total_num_generations : (i + 1) * total_num_generations]
            for i in range(local_num_prompts)
        ]
        grouped_full_completion_texts = [
            full_completion_texts[i * total_num_generations : (i + 1) * total_num_generations]
            for i in range(local_num_prompts)
        ]
        grouped_rendered_prompts = None
        if rendered_prompt_texts is not None:
            grouped_rendered_prompts = [
                rendered_prompt_texts[i * total_num_generations] for i in range(local_num_prompts)
            ]
        local_generation_examples = []
        for example, group_cot_texts, group_cot_lengths, group_texts, group_full_texts, group_is_truncated, expansion_metadata in zip(
            prompt_level_inputs,
            grouped_cot_texts,
            grouped_cot_lengths,
            grouped_completion_texts,
            grouped_full_completion_texts,
            grouped_is_truncated,
            [
                prompt_expansion_metadata[idx] if prompt_expansion_metadata is not None else None
                for idx in range(local_num_prompts)
            ],
            strict=True,
        ):
            generation_example = {
                "question": example.get("question", ""),
                "ground_truth_answer": example.get("answer", ""),
                "reference_solution": example.get("solution", ""),
                "initial_num_generations": int(num_generations),
                "total_num_generations": int(total_num_generations),
                "bfs_po_expansion_steps": expansion_metadata,
                "generated_cots": [
                    {
                        "index": completion_idx,
                        "cot": cot_text,
                        "cot_length": int(cot_length),
                        "is_truncated": bool(is_truncated),
                        "max_num_tokens": int(max_eval_completion_length) if max_eval_completion_length is not None else None,
                        "extracted_answer": self._extract_answer_key(
                            example["question"],
                            completion_text,
                            solution=example.get("solution"),
                            complete_answer=example.get("answer"),
                        ),
                        "scored_completion": completion_text,
                        "full_completion": full_completion_text,
                    }
                    for completion_idx, (
                        cot_text,
                        cot_length,
                        is_truncated,
                        completion_text,
                        full_completion_text,
                    ) in enumerate(
                        zip(
                            group_cot_texts,
                            group_cot_lengths,
                            group_is_truncated,
                            group_texts,
                            group_full_texts,
                            strict=True,
                        )
                    )
                ],
            }
            if grouped_rendered_prompts is not None:
                generation_example["prompt"] = grouped_rendered_prompts[len(local_generation_examples)]
            local_generation_examples.append(generation_example)
        gathered_generation_examples = gather_object(local_generation_examples)
        if self.accelerator.is_main_process:
            self._eval_generation_examples.extend(gathered_generation_examples)
            self._flush_eval_generation_examples_to_json()

        if self._eval_metrics_totals is None:
            self._eval_metrics_totals = {
                "accuracy_sum": 0.0,
                "accuracy_count": 0,
                "completion_length_sum": 0.0,
                "completion_length_count": 0,
                "new_token_sum": 0.0,
                "new_token_count": 0,
                "prompt_count": 0,
            }
            if total_num_generations > 1:
                self._eval_metrics_totals["shortest_completion_accuracy_sum"] = 0.0
                self._eval_metrics_totals["shortest_completion_accuracy_count"] = 0

        self._eval_metrics_totals["accuracy_sum"] += accuracy_sum
        self._eval_metrics_totals["accuracy_count"] += accuracy_count
        self._eval_metrics_totals["completion_length_sum"] += completion_length_sum
        self._eval_metrics_totals["completion_length_count"] += completion_length_count
        self._eval_metrics_totals["new_token_sum"] += batch_new_token_sum
        self._eval_metrics_totals["new_token_count"] += batch_new_token_count
        self._eval_metrics_totals["prompt_count"] += batch_prompt_count
        if total_num_generations > 1:
            self._eval_metrics_totals["shortest_completion_accuracy_sum"] += shortest_completion_accuracy_sum
            self._eval_metrics_totals["shortest_completion_accuracy_count"] += shortest_completion_accuracy_count

        self._metrics["eval"].clear()
        self._metrics["eval"]["accuracy"] = [
            self._eval_metrics_totals["accuracy_sum"] / self._eval_metrics_totals["accuracy_count"]
        ]
        self._metrics["eval"]["completions/mean_new_tokens"] = [
            self._eval_metrics_totals["new_token_sum"] / self._eval_metrics_totals["new_token_count"]
        ]
        self._metrics["eval"]["completions/mean_total_new_tokens_per_prompt"] = [
            self._eval_metrics_totals["new_token_sum"] / self._eval_metrics_totals["prompt_count"]
        ]
        self._metrics["eval"]["completions/total_new_tokens"] = [self._eval_metrics_totals["new_token_sum"]]
        if self._eval_metrics_totals["completion_length_count"] == 0:
            # In the extra-CoT eval setting, completions without a closing </think> tag are excluded from the
            # CoT-length average. If every completion in the batch is excluded, keep the metric at 0 instead of
            # crashing on division by zero.
            mean_completion_length = 0.0
        else:
            mean_completion_length = (
                self._eval_metrics_totals["completion_length_sum"]
                / self._eval_metrics_totals["completion_length_count"]
            )
        self._metrics["eval"]["completions/mean_length"] = [mean_completion_length]
        if total_num_generations > 1:
            self._metrics["eval"]["shortest_completion_accuracy"] = [
                self._eval_metrics_totals["shortest_completion_accuracy_sum"]
                / self._eval_metrics_totals["shortest_completion_accuracy_count"]
            ]

    @staticmethod
    def _write_json_atomic(path: str, payload: Any) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _flush_eval_generation_examples_to_json(self) -> None:
        if not getattr(self, "_save_generated_cot_examples_every_batch", False):
            return

        path = getattr(self, "_save_generated_cot_examples_json_path", None)
        if not path:
            return

        self._write_json_atomic(path, self._eval_generation_examples)

    @staticmethod
    def _compute_aes(
        fine_tuned_accuracy: float,
        zero_shot_accuracy: float,
        fine_tuned_len_cot: float,
        zero_shot_len_cot: float,
    ) -> float:
        if math.isclose(zero_shot_accuracy, 0.0) or math.isclose(zero_shot_len_cot, 0.0):
            return float("nan")

        accuracy_weight = 3.0 if fine_tuned_accuracy >= zero_shot_accuracy else 5.0
        accuracy_term = accuracy_weight * (fine_tuned_accuracy - zero_shot_accuracy) / zero_shot_accuracy
        length_term = (zero_shot_len_cot - fine_tuned_len_cot) / zero_shot_len_cot
        return accuracy_term + length_term

    def _restore_eval_zero_shot_metrics(self, eval_prefix: str) -> dict[str, float] | None:
        if eval_prefix in self._eval_zero_shot_metrics:
            return self._eval_zero_shot_metrics[eval_prefix]

        accuracy_key = f"{eval_prefix}_zero_shot_accuracy"
        len_cot_key = f"{eval_prefix}_zero_shot_len_cot"
        for log_entry in reversed(getattr(self.state, "log_history", [])):
            if accuracy_key in log_entry and len_cot_key in log_entry:
                metrics = {
                    "accuracy": float(log_entry[accuracy_key]),
                    "len_cot": float(log_entry[len_cot_key]),
                }
                self._eval_zero_shot_metrics[eval_prefix] = metrics
                return metrics

        return None

    def _get_or_initialize_eval_zero_shot_metrics(
        self, eval_prefix: str, accuracy: float, len_cot: float
    ) -> dict[str, float]:
        restored = self._restore_eval_zero_shot_metrics(eval_prefix)
        if restored is not None:
            return restored

        metrics = {"accuracy": float(accuracy), "len_cot": float(len_cot)}
        self._eval_zero_shot_metrics[eval_prefix] = metrics
        return metrics

    def _add_eval_aes_metrics(self, metrics: dict[str, float], eval_prefix: str) -> dict[str, float]:
        accuracy_key = f"{eval_prefix}_accuracy"
        mean_length_key = f"{eval_prefix}_completions/mean_length"
        legacy_len_cot_key = f"{eval_prefix}_len_cot"
        if accuracy_key not in metrics:
            return metrics

        if mean_length_key in metrics:
            fine_tuned_len_cot = metrics[mean_length_key]
        elif legacy_len_cot_key in metrics:
            fine_tuned_len_cot = metrics[legacy_len_cot_key]
        else:
            return metrics

        zero_shot_metrics = self._get_or_initialize_eval_zero_shot_metrics(
            eval_prefix=eval_prefix,
            accuracy=metrics[accuracy_key],
            len_cot=fine_tuned_len_cot,
        )
        metrics[f"{eval_prefix}_aes"] = self._compute_aes(
            fine_tuned_accuracy=metrics[accuracy_key],
            zero_shot_accuracy=zero_shot_metrics["accuracy"],
            fine_tuned_len_cot=fine_tuned_len_cot,
            zero_shot_len_cot=zero_shot_metrics["len_cot"],
        )
        return metrics

    def _generate_and_record_eval_metrics(
        self,
        inputs: list[dict[str, torch.Tensor | Any]],
        prompts: list[Any],
        gt_answers: list[str],
        complete_answers: list[str],
    ) -> dict[str, torch.Tensor | Any]:
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            _completions,
            _num_items_in_batch,
            _sampling_per_token_logps_list,
            extra_fields,
            _accuracies,
        ) = self._generate(prompts, gt_answers, complete_answers, mode="eval")

        self._record_eval_metrics_only(
            inputs,
            completion_ids_list,
            tool_mask_list,
            extra_fields,
            prompt_ids_list=prompt_ids_list,
        )
        return {}

    def _get_dynamic_sampling_keep_mask(self, advantages: torch.Tensor) -> torch.Tensor:
        if self._uses_branch_advantage() and advantages.dim() > 1:
            return torch.any(advantages != 0, dim=1)
        return advantages != 0

    @staticmethod
    def _get_vllm_importance_sampling_sequence_keep_mask(
        importance_sampling_ratio: torch.Tensor,
        completion_mask: torch.Tensor,
        tool_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if importance_sampling_ratio.ndim != 2:
            raise ValueError(
                "`importance_sampling_ratio` is expected to have shape (B, T) or (B, 1) when filtering "
                "dynamic sampling sequences."
            )

        if importance_sampling_ratio.shape[1] == 1:
            return importance_sampling_ratio.squeeze(1) != 0

        active_mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
        return torch.any((importance_sampling_ratio != 0) & active_mask, dim=1)

    @staticmethod
    def _get_vllm_importance_sampling_active_mask(
        completion_mask: torch.Tensor,
        tool_mask: torch.Tensor | None = None,
        importance_sampling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active_mask = completion_mask.bool()
        if tool_mask is not None:
            active_mask = active_mask & tool_mask.bool()
        if importance_sampling_mask is not None:
            active_mask = active_mask & importance_sampling_mask.bool()
        return active_mask

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int | tuple[int, ...],
        keepdim: bool = False,
    ) -> torch.Tensor:
        weighted_mask = mask.to(dtype=values.dtype)
        counts = weighted_mask.sum(dim=dim, keepdim=keepdim)
        return (values * weighted_mask).sum(dim=dim, keepdim=keepdim) / counts.clamp(min=1.0)

    @staticmethod
    def _masked_std(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int | tuple[int, ...],
        keepdim: bool = False,
    ) -> torch.Tensor:
        mean = GRPOTrainer._masked_mean(values, mask, dim=dim, keepdim=True)
        weighted_mask = mask.to(dtype=values.dtype)
        counts = weighted_mask.sum(dim=dim, keepdim=True)
        centered = torch.where(mask, values - mean, torch.zeros_like(values))
        variance = centered.pow(2).sum(dim=dim, keepdim=True) / (counts - 1).clamp(min=1.0)
        variance = torch.where(counts > 1, variance, torch.zeros_like(variance))
        std = torch.sqrt(variance)

        if keepdim:
            return std

        squeeze_dims = dim if isinstance(dim, tuple) else (dim,)
        for current_dim in sorted(squeeze_dims, reverse=True):
            std = std.squeeze(current_dim)
        return std

    def _generation_sampling_multiplier(self, mode: str | None = None) -> int:
        if mode is not None and mode != "train":
            return 1
        return self.bfs_po_k + 1

    def _max_completion_length(self, mode: str | None = None) -> int | None:
        return self.max_completion_length_eval if mode == "eval" else self.max_completion_length

    def _uses_branch_advantage(self, extra_fields: dict[str, Any] | None = None) -> bool:
        if not self.branch_advantage or self.bfs_po_k <= 0:
            return False
        if extra_fields is None:
            return True
        expansions = extra_fields.get("expansion") if isinstance(extra_fields, dict) else None
        return bool(expansions)

    def _should_compute_importance_sampling_specific_branch(
        self, extra_fields: dict[str, Any] | None = None
    ) -> bool:
        return (
            self.compute_importance_sampling_specific_branch
            and self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
            and self._uses_branch_advantage(extra_fields)
        )

    def _get_importance_sampling_specific_branch_mask(
        self, advantages: torch.Tensor | None, extra_fields: dict[str, Any] | None = None
    ) -> torch.Tensor | None:
        if not self._should_compute_importance_sampling_specific_branch(extra_fields):
            return None
        if advantages is None or advantages.ndim != 2:
            return None
        return advantages != 0

    @staticmethod
    def _is_dynamic_sampling_placeholder(generation_batch: Any) -> bool:
        return isinstance(generation_batch, dict) and generation_batch.get(_DYNAMIC_SAMPLING_PLACEHOLDER_KEY, False)

    def _needs_new_generation_batch(self) -> bool:
        generate_every = self.args.steps_per_generation * self.num_iterations
        return self._step % generate_every == 0 or self._buffered_inputs is None

    @staticmethod
    def _get_generation_batch_size(generation_batch: dict[str, torch.Tensor | Any]) -> int:
        for value in generation_batch.values():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    continue
                return value.shape[0]
            if isinstance(value, (list, tuple)):
                return len(value)
        return 0

    @staticmethod
    def _index_generation_batch_value(value: Any, indices: list[int]) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            index_tensor = torch.as_tensor(indices, device=value.device, dtype=torch.long)
            return value.index_select(0, index_tensor)
        if isinstance(value, list):
            return [value[idx] for idx in indices]
        if isinstance(value, tuple):
            return tuple(value[idx] for idx in indices)
        return value

    @staticmethod
    def _move_generation_batch_value_to_device(value: Any, device: torch.device | str) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [GRPOTrainer._move_generation_batch_value_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(GRPOTrainer._move_generation_batch_value_to_device(item, device) for item in value)
        if isinstance(value, dict):
            return {
                key: GRPOTrainer._move_generation_batch_value_to_device(item, device) for key, item in value.items()
            }
        return value

    def _should_compute_old_per_token_logps(self) -> bool:
        generate_every = self.args.steps_per_generation * self.num_iterations
        return self.args.gradient_accumulation_steps % generate_every != 0 or (
            self.use_vllm and self.vllm_importance_sampling_correction
        )

    def _compute_vllm_importance_sampling_ratio(
        self,
        old_per_token_logps: torch.Tensor,
        sampling_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        tool_mask: torch.Tensor | None = None,
        importance_sampling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        active_mask = self._get_vllm_importance_sampling_active_mask(
            completion_mask,
            tool_mask,
            importance_sampling_mask,
        )
        per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps).masked_fill(~active_mask, 0.0)

        sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
        if sequence_level_is:
            per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
            logps_diff = per_sequence_logps_diff
        else:
            logps_diff = per_token_logps_diff

        vllm_importance_sampling_ratio = torch.exp(logps_diff)

        # vllm_importance_sampling_ratio.shape:
        #   token_* modes:     (B, T)  (per-token ratio)
        #   sequence_* modes:  (B, 1)  (per-sequence ratio)
        if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
            vllm_importance_sampling_ratio = torch.clamp(
                vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
            )
        elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
            vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
            )
        else:
            raise ValueError(
                f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are "
                "'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
            )

        return vllm_importance_sampling_ratio, sequence_level_is

    def _log_vllm_importance_sampling_metrics(
        self,
        mode: str,
        old_per_token_logps: torch.Tensor,
        sampling_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        tool_mask: torch.Tensor | None,
        vllm_importance_sampling_ratio: torch.Tensor,
        sequence_level_is: bool,
        importance_sampling_mask: torch.Tensor | None = None,
    ) -> None:
        device = old_per_token_logps.device
        delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
        mask = self._get_vllm_importance_sampling_active_mask(
            completion_mask,
            tool_mask,
            importance_sampling_mask,
        )
        delta = delta[mask]
        mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
            self.accelerator.gather(mean_delta).mean().item()
        )
        self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
            self.accelerator.gather(max_delta).max().item()
        )
        if sequence_level_is:
            flat_is_ratio = vllm_importance_sampling_ratio.flatten()
        else:
            flat_is_ratio = vllm_importance_sampling_ratio[mask]

        min_importance_sampling_ratio = (
            torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        )
        mean_importance_sampling_ratio = (
            torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        )
        max_importance_sampling_ratio = (
            torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        )
        self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
            nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
        )
        self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
            self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
        )
        self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
            nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
        )

    def _compute_sequence_level_advantages(
        self,
        rewards_per_func: torch.Tensor,
        accuracies: torch.Tensor,
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = rewards_per_func.device
        dtype = rewards_per_func.dtype
        reward_weights = self.reward_weights.to(device).unsqueeze(0)

        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (rewards_per_func * reward_weights).nansum(dim=1)
            grouped_rewards = rewards.view(-1, num_generations)
            group_size = grouped_rewards.new_tensor(num_generations)
            mean_grouped_rewards = grouped_rewards.sum(dim=1, keepdim=True) / group_size
            centered_rewards = grouped_rewards - mean_grouped_rewards

            if num_generations > 1:
                # Keep the existing two-pass sample variance calculation.
                grouped_std_rewards = (centered_rewards.pow(2).sum(dim=1, keepdim=True) / (group_size - 1)).sqrt()
            else:
                grouped_std_rewards = torch.zeros_like(mean_grouped_rewards)

            if self.scale_rewards in ["group", "none"]:
                std_rewards = grouped_std_rewards.expand_as(grouped_rewards).reshape(-1)
            elif self.scale_rewards == "batch":
                if rewards.numel() > 1:
                    batch_std = rewards.std()
                else:
                    batch_std = torch.tensor(0.0, device=device, dtype=dtype)
                std_rewards = torch.full_like(rewards, batch_std)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = centered_rewards.reshape(-1)
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            finite_grouped = torch.isfinite(grouped)
            safe_grouped = torch.where(finite_grouped, grouped, torch.zeros_like(grouped))

            mean_k = self._masked_mean(safe_grouped, finite_grouped, dim=1, keepdim=True)
            if num_generations > 1:
                std_k = self._masked_std(safe_grouped, finite_grouped, dim=1, keepdim=True)
            else:
                std_k = torch.zeros_like(mean_k)

            normalized_grouped = torch.where(
                finite_grouped,
                (safe_grouped - mean_k) / (std_k + 1e-4),
                torch.zeros_like(grouped),
            )
            rewards = (normalized_grouped.view(-1, len(self.reward_funcs)) * reward_weights).sum(dim=1)

            if rewards.numel() > 0:
                mean_rewards = rewards.mean()
            else:
                mean_rewards = torch.tensor(0.0, device=device, dtype=dtype)
            if rewards.numel() > 1:
                std_rewards_value = rewards.std()
            else:
                std_rewards_value = torch.tensor(0.0, device=device, dtype=dtype)
            std_rewards = torch.full_like(rewards, std_rewards_value)
            advantages = (rewards - mean_rewards) / (std_rewards_value + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        grouped_accuracies = accuracies.view(-1, num_generations)
        zero_advantage_groups = grouped_accuracies.eq(1).all(dim=1) | grouped_accuracies.eq(0).all(dim=1)
        if zero_advantage_groups.any() and self.dynamic_sampling:
            grouped_advantages = advantages.view(-1, num_generations)
            grouped_advantages[zero_advantage_groups] = 0
            advantages = grouped_advantages.view(-1)

        return advantages, is_std_zero

    def _materialize_dynamic_sampling_old_logps(
        self, generation_batch: dict[str, torch.Tensor | Any], mode: str
    ) -> dict[str, torch.Tensor | Any]:
        if mode != "train" or not self.dynamic_sampling or not self._should_compute_old_per_token_logps():
            return generation_batch

        needs_old_per_token_logps = generation_batch.get("old_per_token_logps") is None
        needs_vllm_ratio = (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and generation_batch.get("importance_sampling_ratio") is None
        )
        if not needs_old_per_token_logps and not needs_vllm_ratio:
            return generation_batch

        merged_generation_batch = unsplit_pixel_values_by_grid(generation_batch)
        prompt_ids, prompt_mask = merged_generation_batch["prompt_ids"], merged_generation_batch["prompt_mask"]
        completion_ids, completion_mask = (
            merged_generation_batch["completion_ids"],
            merged_generation_batch["completion_mask"],
        )
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        old_per_token_logps = generation_batch.get("old_per_token_logps")
        if old_per_token_logps is None:
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=merged_generation_batch.get("num_images"),
                    pixel_values=merged_generation_batch.get("pixel_values"),
                    image_grid_thw=merged_generation_batch.get("image_grid_thw"),
                    pixel_attention_mask=merged_generation_batch.get("pixel_attention_mask"),
                    image_sizes=merged_generation_batch.get("image_sizes"),
                    token_type_ids=merged_generation_batch.get("token_type_ids"),
                )

        updated_generation_batch = dict(generation_batch)
        updated_generation_batch["old_per_token_logps"] = old_per_token_logps

        if needs_vllm_ratio:
            sampling_per_token_logps = generation_batch.get("sampling_per_token_logps")
            if sampling_per_token_logps is None:
                raise RuntimeError("vLLM importance sampling correction requires sampling_per_token_logps.")
            tool_mask = generation_batch.get("tool_mask")
            importance_sampling_mask = self._get_importance_sampling_specific_branch_mask(
                generation_batch.get("advantages"),
            )
            vllm_importance_sampling_ratio, sequence_level_is = self._compute_vllm_importance_sampling_ratio(
                old_per_token_logps,
                sampling_per_token_logps,
                completion_mask,
                tool_mask,
                importance_sampling_mask=importance_sampling_mask,
            )
            updated_generation_batch["importance_sampling_ratio"] = vllm_importance_sampling_ratio
            self._log_vllm_importance_sampling_metrics(
                mode,
                old_per_token_logps,
                sampling_per_token_logps,
                completion_mask,
                tool_mask,
                vllm_importance_sampling_ratio,
                sequence_level_is,
                importance_sampling_mask=importance_sampling_mask,
            )
            mask = self._get_vllm_importance_sampling_active_mask(
                completion_mask,
                tool_mask,
                importance_sampling_mask,
            )

        return updated_generation_batch

    def _move_generation_batch_to_device(
        self, generation_batch: dict[str, torch.Tensor | Any], device: torch.device | str
    ) -> dict[str, torch.Tensor | Any]:
        return {
            key: self._move_generation_batch_value_to_device(value, device) for key, value in generation_batch.items()
        }

    def _slice_generation_batch(
        self, generation_batch: dict[str, torch.Tensor | Any], start_idx: int, end_idx: int
    ) -> dict[str, torch.Tensor | Any]:
        sampled_indices = list(range(start_idx, end_idx))
        return {
            key: self._index_generation_batch_value(value, sampled_indices) for key, value in generation_batch.items()
        }

    def _concat_generation_batch_values(self, key: str, values: list[Any]) -> Any:
        non_none_values = [value for value in values if value is not None]
        if not non_none_values:
            return None

        first_value = non_none_values[0]
        if isinstance(first_value, list):
            merged = []
            for value in non_none_values:
                merged.extend(value)
            return merged
        if isinstance(first_value, tuple):
            merged = []
            for value in non_none_values:
                merged.extend(value)
            return tuple(merged)
        if isinstance(first_value, torch.Tensor):
            if first_value.ndim == 0:
                return first_value
            try:
                return torch.cat(non_none_values, dim=0)
            except RuntimeError:
                per_sample_tensors = [tensor for value in non_none_values for tensor in value.unbind(0)]
                padding_value = 0
                padding_side = "right"
                if key == "prompt_ids":
                    padding_value = self.pad_token_id
                    padding_side = "left"
                elif key == "prompt_mask":
                    padding_value = 0
                    padding_side = "left"
                elif key == "completion_ids":
                    padding_value = self.pad_token_id
                    padding_side = "right"
                elif key == "completion_mask":
                    padding_value = 0
                    padding_side = "right"
                elif key == "tool_mask":
                    padding_value = 1
                    padding_side = "right"
                elif key in {
                    "advantages",
                    "old_per_token_logps",
                    "ref_per_token_logps",
                    "sampling_per_token_logps",
                    "importance_sampling_ratio",
                    "token_type_ids",
                    "pixel_attention_mask",
                    "image_sizes",
                }:
                    padding_value = 0
                    padding_side = "right"
                return pad(per_sample_tensors, padding_value=padding_value, padding_side=padding_side)
        return first_value

    def _concat_generation_batches(
        self, generation_batches: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        if len(generation_batches) == 1:
            return generation_batches[0]

        keys = {key for batch in generation_batches for key in batch}
        return {
            key: self._concat_generation_batch_values(key, [batch.get(key) for batch in generation_batches])
            for key in keys
        }

    def _apply_dynamic_sampling(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        advantages = generation_batch["advantages"]
        keep_mask = self._get_dynamic_sampling_keep_mask(advantages)
        kept_indices = keep_mask.nonzero(as_tuple=True)[0].tolist()

        return {
            key: self._index_generation_batch_value(value, kept_indices) for key, value in generation_batch.items()
        }

    def _append_dynamic_sampling_batch(self, generation_batch: dict[str, torch.Tensor | Any]) -> None:
        filtered_generation_batch = self._apply_dynamic_sampling(generation_batch)
        gathered_batches = self._gather_dynamic_sampling_batches(filtered_generation_batch)
        if not gathered_batches:
            return
        gathered_generation_batch = self._concat_generation_batches(gathered_batches)

        if self._dynamic_sampling_pending_buffer is None:
            self._dynamic_sampling_pending_buffer = gathered_generation_batch
        else:
            self._dynamic_sampling_pending_buffer = self._concat_generation_batches(
                [self._dynamic_sampling_pending_buffer, gathered_generation_batch]
            )

    def _gather_dynamic_sampling_batches(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> list[dict[str, torch.Tensor | Any]]:
        if self._get_generation_batch_size(generation_batch) == 0:
            local_batches = []
        elif self.accelerator.num_processes == 1:
            local_batches = [generation_batch]
        else:
            local_batches = [self._move_generation_batch_to_device(generation_batch, "cpu")]

        if self.accelerator.num_processes == 1:
            return local_batches

        gathered_batches = gather_object(local_batches)
        return [self._move_generation_batch_to_device(batch, self.accelerator.device) for batch in gathered_batches]

    def _shuffle_generation_batch(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        batch_size = self._get_generation_batch_size(generation_batch)
        if batch_size <= 1:
            return generation_batch

        generator = torch.Generator()
        generator.manual_seed(self.args.seed + self._dynamic_sampling_shuffle_count)
        self._dynamic_sampling_shuffle_count += 1
        permutation = torch.randperm(batch_size, generator=generator).tolist()
        return {key: self._index_generation_batch_value(value, permutation) for key, value in generation_batch.items()}

    def _get_dynamic_sampling_global_target_batch_size(self) -> int:
        return self.global_train_batch_size

    def _get_dynamic_sampling_local_target_batch_size(self) -> int:
        return self._get_dynamic_sampling_global_target_batch_size() // self.accelerator.num_processes

    def _split_global_dynamic_sampling_batch(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        local_target_batch_size = self._get_dynamic_sampling_local_target_batch_size()
        process_index = self.accelerator.process_index
        start_idx = process_index * local_target_batch_size
        end_idx = start_idx + local_target_batch_size
        return self._slice_generation_batch(generation_batch, start_idx, end_idx)

    def _pop_dynamic_sampling_batch_from_pending(self) -> dict[str, torch.Tensor | Any] | None:
        global_target_batch_size = self._get_dynamic_sampling_global_target_batch_size()
        if self._dynamic_sampling_pending_buffer is None:
            return None
        pending_batch_size = self._get_generation_batch_size(self._dynamic_sampling_pending_buffer)
        if pending_batch_size < global_target_batch_size:
            return None

        self._dynamic_sampling_pending_buffer = self._shuffle_generation_batch(self._dynamic_sampling_pending_buffer)
        current_global_generation_batch = self._slice_generation_batch(
            self._dynamic_sampling_pending_buffer, 0, global_target_batch_size
        )
        self._dynamic_sampling_pending_buffer = None
        self._metrics["train"]["sampling/dynamic_sampling_rejected_examples"].append(
            pending_batch_size - global_target_batch_size
        )
        return self._split_global_dynamic_sampling_batch(current_global_generation_batch)

    def _get_dynamic_sampling_batch(
        self, initial_generation_batch: list[dict[str, torch.Tensor | Any]] | None = None
    ) -> dict[str, torch.Tensor | Any] | None:
        current_generation_batch = self._pop_dynamic_sampling_batch_from_pending()
        if current_generation_batch is not None:
            return current_generation_batch

        if initial_generation_batch is not None:
            generation_batch = self._generate_and_score_completions(initial_generation_batch, mode="train")
            generation_batch = split_pixel_values_by_grid(generation_batch)
            self._append_dynamic_sampling_batch(generation_batch)
            current_generation_batch = self._pop_dynamic_sampling_batch_from_pending()
            if current_generation_batch is not None:
                return current_generation_batch

        if self._active_train_batch_iterator is None:
            raise RuntimeError("Dynamic sampling requires an active train dataloader iterator.")

        while True:
            try:
                next_generation_batch = next(self._active_train_batch_iterator)
            except StopIteration:
                return None

            next_generation_batch = self._generate_and_score_completions(next_generation_batch, mode="train")
            next_generation_batch = split_pixel_values_by_grid(next_generation_batch)
            self._append_dynamic_sampling_batch(next_generation_batch)
            current_generation_batch = self._pop_dynamic_sampling_batch_from_pending()
            if current_generation_batch is not None:
                return current_generation_batch

    def floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]) -> int:
        if self.dynamic_sampling and self._is_dynamic_sampling_placeholder(inputs):
            return 0
        return super().floating_point_ops(inputs)

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            if self._needs_new_generation_batch():
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                if self.dynamic_sampling:
                    if self._ready_dynamic_sampling_batch is not None:
                        generation_batch = self._ready_dynamic_sampling_batch
                        self._ready_dynamic_sampling_batch = None
                    else:
                        raw_generation_batch = (
                            None if self._is_dynamic_sampling_placeholder(generation_batch) else generation_batch
                        )
                        generation_batch = self._get_dynamic_sampling_batch(raw_generation_batch)
                        if generation_batch is None:
                            raise RuntimeError(
                                "Dynamic sampling did not collect enough samples to build a full training batch."
                            )
                else:
                    generation_batch = self._generate_and_score_completions(generation_batch, mode="train")
                    generation_batch = split_pixel_values_by_grid(generation_batch)
                # When dynamic sampling is enabled, compute old log-probs only on the final kept local batch.
                generation_batch = self._materialize_dynamic_sampling_old_logps(generation_batch, mode)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                microbatch_size = self.current_gradient_accumulation_steps * self._train_batch_size

                if self._uses_branch_advantage() and generation_batch["advantages"].dim() > 1:
                    sequence_num_items = torch.sum(generation_batch["advantages"] != 0, dim=1)
                else:
                    sequence_num_items = torch.sum(generation_batch["completion_ids"] != self.pad_token_id, dim=1)

                if sequence_num_items.numel() % microbatch_size == 0:
                    num_items_batch = sequence_num_items.reshape(-1, microbatch_size).sum(dim=1)
                else:
                    num_items_batch = sequence_num_items.sum().unsqueeze(0)
                num_items_batch = self.accelerator.reduce(num_items_batch, reduction="sum")

                chunk_size = generation_batches[0]["completion_ids"].shape[0]
                if num_items_batch.numel() == 1 or microbatch_size % chunk_size != 0:
                    for batch in generation_batches:
                        batch["num_items_in_batch"] = num_items_batch[0]
                else:
                    chunks_per_optimizer_step = microbatch_size // chunk_size
                    for i, batch in enumerate(generation_batches):
                        index = min(i // chunks_per_optimizer_step, num_items_batch.numel() - 1)
                        batch["num_items_in_batch"] = num_items_batch[index]
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch, mode="eval")
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list, complete_answers):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state
        reward_kwargs["complete_answers"] = complete_answers
        if self.environments is not None:
            reward_kwargs["environments"] = self.environments

        async_funcs_info = []  # async custom functions for asyncio.gather

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                with profiling_context(self, reward_func_name):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            elif asyncio.iscoroutinefunction(reward_func):  # Separate async reward funcs to run them in parallel later
                async_funcs_info.append((i, reward_func, reward_func_name))
            else:
                # Run synchronous reward function
                with profiling_context(self, reward_func_name):
                    if reward_func is accuracy_reward:
                        
                        completions_to_process = [
                            [{
                                "role": "system",
                                "content": (
                                    completion[0]["content"].split("</think>", 1)[1]
                                    if "</think>" in completion[0]["content"]
                                    else ""
                                )
                            }]
                            for completion in completions
                            ]
                    else:
                        completions_to_process = completions
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions_to_process, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Execute async custom functions in parallel using asyncio.gather
        if async_funcs_info:

            async def _invoke_async(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    return index, output

            async def _run_async_funcs():
                coros = [_invoke_async(i, func, func_name) for (i, func, func_name) in async_funcs_info]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(_run_async_funcs(), self.async_loop).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _score_generated_completion(self, completion_ids, complete_answer, question, gt_answer):
        completion_text = self.processing_class.decode(completion_ids)
        return accuracy_reward(
            [[{"content": completion_text, "role": "assistant"}]],
            **{"complete_answers": [complete_answer], "question": [question], "solution": [gt_answer]},
        )[0]

    def _decode_expansion_prefix(self, token_ids):
        if not token_ids:
            return ""
        decode_kwargs = {"skip_special_tokens": False}
        try:
            decode_kwargs["clean_up_tokenization_spaces"] = False
            return self.processing_class.decode(token_ids, **decode_kwargs)
        except TypeError:
            return self.processing_class.decode(token_ids, skip_special_tokens=False)

    def _get_tokenizer_name_or_path(self) -> str:
        processing_class = self.processing_class
        tokenizer = processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class
        return str(getattr(tokenizer, "name_or_path", "") or "").lower()

    def _get_think_token_ids(self) -> tuple[int, int]:
        """Cache the thinking delimiter IDs required by the Qwen3 experiment."""
        if not getattr(self, "_think_token_checked", False):
            processing_class = self.processing_class
            tokenizer = processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class
            vocab = tokenizer.get_vocab()
            self._think_token_id = vocab["<think>"]
            self._end_think_token_id = vocab["</think>"]
            self._think_token_checked = True
        return self._think_token_id, self._end_think_token_id

    def _build_replay_prompt(self, starting_prompt, content: str):
        """Continue the existing assistant prefill, or append an assistant turn, without model-specific formatting."""
        prompt = copy.deepcopy(starting_prompt)
        if prompt:
            last_message = prompt[-1]
            if (
                isinstance(last_message, dict)
                and last_message.get("role") == "assistant"
                and isinstance(last_message.get("content"), str)
            ):
                last_message["content"] += content
                return prompt

        prompt.append({"role": "assistant", "content": content})
        return prompt

    @staticmethod
    def _prompt_has_only_text_content(prompt) -> bool:
        if not isinstance(prompt, list):
            return True

        for message in prompt:
            if not isinstance(message, dict):
                return False
            content = message.get("content")
            if isinstance(content, str) or content is None:
                continue
            if not isinstance(content, list):
                return False
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    return False
        return True

    def _should_use_vllm_prompt_token_ids(self) -> bool:
        """Whether BFS replay prompts can be sent to vLLM as exact token IDs."""
        return self.use_vllm and getattr(self.vllm_generation, "mode", None) in {"colocate", "server"}

    @staticmethod
    def _make_prompt_token_ids_prompt(
        prompt_ids: list[int],
        completion_prefix_ids: list[int] | None = None,
    ) -> dict[str, list[int]]:
        prompt_token_ids = list(prompt_ids)
        if completion_prefix_ids is not None:
            prompt_token_ids.extend(int(token_id) for token_id in completion_prefix_ids)
        return {"prompt_token_ids": prompt_token_ids}

    def _generate_single_turn(
        self, prompts: list, gt_answers: list, complete_answers: list, questions: list, mode: str | None = None
    ):
        device = self.accelerator.device
        if mode is None:
            mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        bfs_po_iterations = self.bfs_po_k + 1 if mode == "train" else self._eval_sampling_multiplier()
        num_samples = int(len(prompts) / num_generations)
        answers_info = {i: [] for i in range(num_samples)}
        dict_expansion = None
        samples_map = [[(i * num_generations) + j for j in range(num_generations)] for i in range(num_samples)]
        starting_prompts = prompts

        processing_class = self.processing_class
        _, end_think_token_id = self._get_think_token_ids()

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
        
            with profiling_context(self, "sync_weights"):
                self.vllm_generation.sync_weights()

            use_vllm_prompt_token_ids = self._should_use_vllm_prompt_token_ids()
            replay_prompt_token_ids = None
            for k in range(bfs_po_iterations):
                max_tokens_by_prompt = None
                if k > 0:
                    max_completion_length = (
                        getattr(self, "max_completion_length_eval", None)
                        if mode == "eval"
                        else getattr(self, "max_completion_length", None)
                    )
                    if max_completion_length is not None:
                        max_tokens_by_prompt = [
                            max(0, max_completion_length - dict_expansion[i][-1][1])
                            for i in answers_info
                            for _ in range(num_generations)
                        ]
                # Generate using vLLM
                sleep_after_generation = k == bfs_po_iterations - 1 and mode == "train"

                generation_outputs = self.vllm_generation.generate(
                    prompts=replay_prompt_token_ids if replay_prompt_token_ids is not None else prompts,
                    num_generations=num_generations,
                    profiler=profiling_context(self, "vLLM.generate"),
                    dict_expansion=dict_expansion,
                    validation=mode == "eval",
                    max_tokens_by_prompt=max_tokens_by_prompt,
                    sleep_after_generation=sleep_after_generation,
                )
                prompt_ids, completion_ids, logprobs, entropies, extra_fields = generation_outputs
                if max_tokens_by_prompt is not None:
                    completion_ids = [
                        completion[:max_tokens]
                        for completion, max_tokens in zip(completion_ids, max_tokens_by_prompt, strict=True)
                    ]
                    logprobs = [
                        logprobs_for_completion[:max_tokens]
                        for logprobs_for_completion, max_tokens in zip(logprobs, max_tokens_by_prompt, strict=True)
                    ]
                    entropies = [
                        entropies_for_completion[:max_tokens]
                        for entropies_for_completion, max_tokens in zip(entropies, max_tokens_by_prompt, strict=True)
                    ]

                if k == 0:
                    dict_expansion = {i: [] for i in range(num_samples)}
                    true_prompt_ids = prompt_ids
                # Update info
                for i in range(len(entropies)):
                    where_in_samples = [i in samples_map[j] for j in range(len(samples_map))].index(True)
                    entropy_to_append = entropies[i]

                    entropy_to_append[0] = 0
                    logprob_to_append = logprobs[i]
                    completion_ids_to_append = completion_ids[i]

                    if k > 0:
                        expanded_from_answer = dict_expansion[where_in_samples][-1][0]
                        expanded_from_token = dict_expansion[where_in_samples][-1][1]
                        entropy_to_append = answers_info[where_in_samples][expanded_from_answer]["entropy"][:expanded_from_token] + entropy_to_append
                        logprob_to_append = answers_info[where_in_samples][expanded_from_answer]["logprobs"][:expanded_from_token] + logprob_to_append
                        completion_ids_to_append = answers_info[where_in_samples][expanded_from_answer]["completion_ids"][:expanded_from_token] + completion_ids_to_append

                    sample_offset = where_in_samples * num_generations

                    try:
                        answer_start = completion_ids_to_append.index(end_think_token_id) + 1
                    except ValueError:
                        model_answer = ""
                    else:
                        model_answer = (
                            processing_class.decode(completion_ids_to_append[answer_start:])
                            if answer_start < len(completion_ids_to_append)
                            else ""
                        )
                    acc = accuracy_reward(
                        [[{"content": model_answer, "role": "assistant"}]],
                        **{"complete_answers": [complete_answers[sample_offset]],
                        "question": [questions[sample_offset]],
                        "solution": [gt_answers[sample_offset]]}
                    )[0]

                    answers_info[where_in_samples].append({"prompt_ids": true_prompt_ids[sample_offset],
                                                        "completion_ids": completion_ids_to_append,
                                                        "entropy": entropy_to_append,
                                                        "logprobs": logprob_to_append,
                                                            "accuracy": acc,
                                                            "answer_len": len(completion_ids_to_append),
                                                            "new_token_count": len(completion_ids[i])})
                prompts = []
                next_replay_prompt_token_ids = [] if use_vllm_prompt_token_ids else None

                for i in answers_info.keys():
                    sample_answers = answers_info[i]
                    which_expand = [0, 0]
                    all_correct = all(answer_info["accuracy"] for answer_info in sample_answers)
                    if not all_correct:
                        min_len = math.inf
                        for j, answer_info in enumerate(sample_answers):
                            if (
                                answer_info["accuracy"] == 1
                                and 32 < answer_info["answer_len"] < min_len
                            ):
                                max_index = self._select_bfs_po_expansion_token_index(
                                    answer_info, end_think_token_id
                                )
                                which_expand = [j, max_index]
                                min_len = answer_info["answer_len"]
                    sample_answers[which_expand[0]]["entropy"][which_expand[1]] = 0

                    if k > 0:
                        change_expand = False
                        for j in range(len(dict_expansion[i])):
                            if answers_info[i][dict_expansion[i][j][0]]["completion_ids"][:which_expand[1]] == answers_info[i][which_expand[0]]["completion_ids"][:which_expand[1]] and not change_expand:
                                change_expand = True
                                which_expand = [dict_expansion[i][j][0], which_expand[1]]

                    dict_expansion[i].append(which_expand)

                    expansion_prefix_ids = answers_info[i][which_expand[0]]["completion_ids"][:which_expand[1]]
                    content = self._decode_expansion_prefix(expansion_prefix_ids)
                    answers_info[i][which_expand[0]]["entropy"][which_expand[1]] = 0

                    replay_prompt = self._build_replay_prompt(
                        starting_prompts[int(i * num_generations)],
                        content,
                    )
                    for _ in range(num_generations):
                        prompts.append(copy.deepcopy(replay_prompt))
                        if next_replay_prompt_token_ids is not None:
                            next_replay_prompt_token_ids.append(
                                self._make_prompt_token_ids_prompt(
                                    answers_info[i][which_expand[0]]["prompt_ids"],
                                    expansion_prefix_ids,
                                )
                            )

                replay_prompt_token_ids = next_replay_prompt_token_ids

            prompt_ids = [answers_info[i][j]["prompt_ids"] for i in answers_info.keys() for j in range(len(answers_info[i]))]
            completion_ids = [answers_info[i][j]["completion_ids"] for i in answers_info.keys() for j in range(len(answers_info[i]))]
            logprobs = [answers_info[i][j]["logprobs"] for i in answers_info.keys() for j in range(len(answers_info[i]))]
            extra_fields = {
                "new_token_counts": [
                    answers_info[i][j]["new_token_count"] for i in answers_info.keys() for j in range(len(answers_info[i]))
                ]
            }
            if bfs_po_iterations > 1:
                extra_fields["expansion"] = [
                    dict_expansion[i][:-1] for i in answers_info.keys() for _ in range(int(num_generations * bfs_po_iterations))
                ]
            accuracies = [answers_info[i][j]["accuracy"] for i in range(len(answers_info)) for j in range(len(answers_info[i]))]

        elif self.use_transformers_paged:
            if is_conversational({"prompt": prompts[0]}):
                processor_outputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                processor_outputs = self.processing_class(text=prompts)
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"], generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None  # not used in this case
            extra_fields = {"new_token_counts": [len(ids) for ids in completion_ids]}
        else:
            # Regular generation path
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(
                    text=prompts, padding=True, padding_side="left", return_tensors="pt"
                )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                generation_config = self.generation_config
                max_completion_length = self._max_completion_length(mode)
                if max_completion_length != self.max_completion_length:
                    generation_config = copy.deepcopy(self.generation_config)
                    generation_config.max_new_tokens = max_completion_length
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
            logprobs = None  # not used in this case
            extra_fields = {"new_token_counts": [len(ids) for ids in completion_ids]}

        if "accuracies" not in locals():
            accuracies = [
                self._score_generated_completion(completion, answer, question, gt_answer)
                for completion, answer, question, gt_answer in zip(
                    completion_ids, complete_answers, questions, gt_answers, strict=True
                )
            ]
        
        return prompt_ids, completion_ids, logprobs, extra_fields, accuracies

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs, mode: str):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        max_completion_length = self._max_completion_length(mode)
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls

            # Call the tools, and build the new prompt for generation
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append((name, sync_tool_dict[name](**function["arguments"])))
                            elif name in async_tool_dict:
                                async_coros.append((name, async_tool_dict[name](**function["arguments"])))
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:
                    tool_message = {"role": "tool", "name": name, "content": str(result)}
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # Tokenize and filter samples whose length exceeds max allowed length. This is important, because both
            # vLLM and transformers will error out if the input is longer than the model's max length.
            pct_ids = self.processing_class.apply_chat_template(
                prompt_completion_tools,
                tools=self.tools,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.chat_template_kwargs,
            )
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.llm.llm_engine.model_config.max_model_len
            elif not self.use_vllm:
                max_model_len = self.model.config.max_position_embeddings
            else:
                raise NotImplementedError(
                    f"Unsupported mode detected: use_vllm={self.use_vllm}, vllm_mode={self.vllm_mode}"
                )
            overlong = [len(pct) >= max_model_len for pct in pct_ids]
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if overlong[idx]:
                    prompt_length = len(prompt_ids[idx_with_tool])
                    ct = pct_ids[idx][prompt_length : prompt_length + max_completion_length]
                    completion_ids[idx_with_tool] = ct
                    tool_mask[idx_with_tool] += [1] * (len(ct) - len(tool_mask[idx_with_tool]))
                    if logprobs is not None:
                        logprobs[idx_with_tool] += [0.0] * (len(ct) - len(logprobs[idx_with_tool]))
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tools = [pct for pct, o in zip(prompt_completion_tools, overlong, strict=True) if not o]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Generate new completions after tool execution
            prompt_completion_tool_ids, post_tool_ids, post_tool_logprobs, _ = self._generate_single_turn(
                prompt_completion_tools, mode=mode
            )

            # Sanity check: from experience, this is useful to catch bugs in the chat template
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                if prompt_ids[idx_with_tool] != pct[: len(prompt_ids[idx_with_tool])]:
                    raise ValueError(
                        "The chat template is not prefix-preserving. Please update it to use a prefix-preserving "
                        "format."
                    )

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_len = len(prompt_ids[idx_with_tool])
                completion_tool_ids = prompt_completion_tool_ids[idx][prompt_len:]
                excess_length = len(completion_tool_ids) + len(post_tool_ids[idx]) - max_completion_length
                if excess_length > 0:
                    # If exceeding max length, truncate post_tool_ids
                    post_tool_ids[idx] = post_tool_ids[idx][:-excess_length]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:-excess_length]
                    excess_length = len(completion_tool_ids) + len(post_tool_ids[idx]) - max_completion_length
                    if excess_length > 0:
                        # If still exceeding max length, truncate completion_tool_ids as well
                        prompt_completion_tool_ids[idx] = prompt_completion_tool_ids[idx][:-excess_length]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions
            post_tool_completions = [
                parse_response(self.processing_class, ids) if ids else {} for ids in post_tool_ids
            ]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1
        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _generate(self, prompts: list, gt_answers: list, complete_answers: list, mode: str | None = None):
        device = self.accelerator.device
        if mode is None:
            mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        # Native chat templates may receive only a user message, without an explicit system message.
        questions = [_extract_prompt_question(prompt) for prompt in prompts]
        prompt_ids, completion_ids, logprobs, extra_fields, accuracies = self._generate_single_turn(
            prompts, gt_answers, complete_answers, questions, mode=mode
        )

        # Decode completions. It's important to use `parse_response` when possible, because it handles tool calls.
        if is_conversational({"prompt": prompts[0]}):
            if (
                Version(transformers.__version__) >= Version("5.0.0")  # parse_response added in v5
                and isinstance(self.processing_class, PreTrainedTokenizerBase)  # doesn't work with processors
                and hasattr(self.processing_class, "response_schema")  # attribute not set by default for now
                and self.processing_class.response_schema is not None  # only works if the tokenizer has a schema
            ):
                completions = [[parse_response(self.processing_class, ids)] for ids in completion_ids]
            else:
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Extract tool calls from the completions and (possibly) execute them
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                tool_call_count,
                tool_failure_count,
            ) = self._tool_call_loop(prompts, prompt_ids, completion_ids, completions, logprobs, mode=mode)
        else:
            # Support custom env_mask from rollout_func (e.g., for environment feedback masking)
            # Internally treated as tool_mask - marks model tokens (1) vs external tokens (0)
            tool_mask = extra_fields.pop("env_mask", None)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:  # count only model-generated tokens (tool_mask=1)
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()

        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
            accuracies
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]], mode: str | None = None
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        if mode is None:
            mode = "train" if self.model.training else "eval"
        sampling_multiplier = self._generation_sampling_multiplier(mode)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            
        prompts = [x["prompt"] for x in inputs]
        gt_answers = [x["solution"] for x in inputs]
        complete_answers = [x["answer"] for x in inputs]
        questions = [inputs[i]["question"] for i in range(len(inputs))]
        if self.environments:
            for prompt, environment, reset_kwargs in zip(prompts, self.environments, inputs, strict=True):
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                prompt[-1]["content"] += observation

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        if mode == "eval":
            return self._generate_and_record_eval_metrics(inputs, prompts, gt_answers, complete_answers)

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
            accuracies,
        ) = self._generate(prompts, gt_answers, complete_answers, mode=mode)
    
        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [
                torch.tensor(logps, device=device, dtype=torch.float32) for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            # With dynamic sampling, this computation is deferred until the final kept local batch is assembled,
            # except when we need the vLLM IS ratio early to exclude fully masked sequences before recomputing group
            # advantages.
            defer_old_per_token_logps = (
                mode == "train"
                and self.dynamic_sampling
            )
            if self._should_compute_old_per_token_logps() and (
                not defer_old_per_token_logps
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None
            stored_old_per_token_logps = None if defer_old_per_token_logps else old_per_token_logps
            sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
            should_delay_branch_specific_vllm_ratio = self._should_compute_importance_sampling_specific_branch(
                extra_fields
            )

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction and old_per_token_logps is not None:
                if should_delay_branch_specific_vllm_ratio:
                    vllm_importance_sampling_ratio = None
                else:
                    vllm_importance_sampling_ratio, sequence_level_is = self._compute_vllm_importance_sampling_ratio(
                        old_per_token_logps,
                        sampling_per_token_logps,
                        completion_mask,
                        tool_mask,
                    )
            else:
                vllm_importance_sampling_ratio = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        # Expand each prompt to match the extra completions produced by BFS sampling.
        prompts = [p.copy() for p in prompts for _ in range(sampling_multiplier)]
        inputs = [p.copy() for p in inputs for _ in range(sampling_multiplier)]
        questions = [p for p in questions for _ in range(sampling_multiplier)]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        complete_answers = [inputs[i]["answer"] for i in range(len(inputs))]
        
        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if key == "new_token_counts":
                        continue
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list, complete_answers)
        if self.negative_rewards != 0:
            for i, reward_func in enumerate(self.reward_funcs):
                if reward_func is accuracy_reward:
                    rewards_per_func[:, i] = torch.where(
                        rewards_per_func[:, i] == 0,
                        torch.full_like(rewards_per_func[:, i], self.negative_rewards),
                        rewards_per_func[:, i],
                    )
        num_generations = self.num_generations * sampling_multiplier if mode == "train" else self.num_generations_eval
        accuracy_dtype = rewards_per_func.dtype
        accuracies = self.accelerator.gather(torch.as_tensor(accuracies, device=device, dtype=accuracy_dtype))


        if self._uses_branch_advantage(extra_fields):
            if self.multi_objective_aggregation == "sum_then_normalize":
                rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
                if num_generations > 1:
                    grouped_std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    grouped_std_rewards = grouped_std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    grouped_std_rewards = torch.zeros_like(rewards)
                if self.scale_rewards in ["group", "none"]:
                    std_rewards = grouped_std_rewards
                elif self.scale_rewards == "batch":
                    std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                    )
                is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            elif self.multi_objective_aggregation == "normalize_then_sum":
                grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
                mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
                std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
                reward_k = (grouped - mean_k) / (std_k + 1e-4)
                reward_k = reward_k.view(-1, len(self.reward_funcs))
                rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
                std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
                is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            else:
                raise ValueError(
                    f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                    "'sum_then_normalize' or 'normalize_then_sum'."
                )
        if self._uses_branch_advantage(extra_fields):
            expansions = [
                extra_fields["expansion"][i] for i in range(0, len(extra_fields["expansion"]), num_generations)
            ]
            expansions = gather_object(expansions)
            completion_ids_list = gather_object(completion_ids_list)
            advantages_shape = completion_ids.shape[1]

            advantages = compute_branch_advantages(
                rewards,
                expansions,
                completion_ids_list,
                advantages_shape,
                self.bfs_po_k,
                num_generations,
                accuracies,
            )
        else:
            advantages, is_std_zero = self._compute_sequence_level_advantages(
                rewards_per_func,
                accuracies,
                num_generations,
            )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]
        importance_sampling_mask = self._get_importance_sampling_specific_branch_mask(
            advantages,
            extra_fields,
        )
        if (
            importance_sampling_mask is not None
            and self.use_vllm
            and self.vllm_importance_sampling_correction
            and old_per_token_logps is not None
        ):
            vllm_importance_sampling_ratio, sequence_level_is = self._compute_vllm_importance_sampling_ratio(
                old_per_token_logps,
                sampling_per_token_logps,
                completion_mask,
                tool_mask,
                importance_sampling_mask=importance_sampling_mask,
            )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction and old_per_token_logps is not None:
            self._log_vllm_importance_sampling_metrics(
                mode,
                old_per_token_logps,
                sampling_per_token_logps,
                completion_mask,
                tool_mask,
                vllm_importance_sampling_ratio,
                sequence_level_is,
                importance_sampling_mask=importance_sampling_mask,
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if stored_old_per_token_logps is not None:
            output["old_per_token_logps"] = stored_old_per_token_logps
        if vllm_importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # Apply tool_mask (from env_mask) for loss computation in multi-turn training scenarios
        loss_mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]
        # Compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            # The attention_mask parameter in liger loss is actually used as a loss mask (not model attention)
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
            vllm_is_ratio=inputs.get("importance_sampling_ratio"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        return loss / normalizer

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_kernel:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    @staticmethod
    def get_off_policy_mask(
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
        sampling_per_token_logps: torch.Tensor,
        mask: torch.Tensor,
        off_policy_threshold: float,
    ) -> torch.Tensor:
        """
        Computes the Off-Policy Sequence Mask from DeepSeek-V3.2 paper. Returns a (B, 1) tensor where 1.0 indicates
        "Keep" and 0.0 indicates "Drop".
        """
        # forward KL div: log(pi_old) - log(pi_theta)
        kl_div = sampling_per_token_logps - per_token_logps.detach()
        # Sequence-level Mean KL (ignoring prompt+padding)
        seq_kl_sum = (kl_div * mask).sum(dim=1, keepdim=True)
        avg_seq_kl = seq_kl_sum / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Keep if (Advantage >= 0) OR (KL <= delta)
        is_pos_adv = advantages >= 0
        is_low_kl = avg_seq_kl <= off_policy_threshold
        return (is_pos_adv | is_low_kl).to(dtype=mask.dtype)  # (B, 1)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]
        mode = "train" if self.model.training else "eval"

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]

        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            # Unless importance_sampling_level="token" (not recommended here), per_token_loss is expected to be (B, 1)
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        metric_mask = mask
        if self._uses_branch_advantage():
            metric_mask = metric_mask * (advantages != 0).to(mask.dtype)
        completion_token_count = metric_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * metric_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        if model.training:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        return None, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # already include the evaluate() metric prefix (e.g. `eval` or `eval_math`). Recover that prefix so custom
        # metrics are grouped under the same namespace for both single-dataset and dict-of-datasets evaluation.
        if mode == "eval":
            eval_prefix = "eval"
            known_suffixes = (
                "_loss",
                "_runtime",
                "_samples_per_second",
                "_steps_per_second",
                "_jit_compilation_time",
                "_model_preparation_time",
            )
            for key in logs:
                if not key.startswith("eval"):
                    continue
                matched_suffix = next((suffix for suffix in known_suffixes if key.endswith(suffix)), None)
                if matched_suffix is not None:
                    eval_prefix = key[: -len(matched_suffix)]
                    break
            metrics = {f"{eval_prefix}_{key}": val for key, val in metrics.items()}
            metrics = self._add_eval_aes_metrics(metrics, eval_prefix)

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()
        if mode == "eval":
            self._eval_metrics_totals = None

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            table = {
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
                **self._logs["rewards"],
                "advantage": self._logs["advantages"],
            }

            df_base = pd.DataFrame(table)
            df_base.to_parquet(
                os.path.join(
                    self.args.output_dir,
                    "completions",
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            images_raw = self._logs["images"] or []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in self._logs["images"]:
                        images.append([logging_backend.Image(image) for image in image_list])
                    df = pd.concat(
                        [df_base, pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
                    df = df_base

                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])

                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
