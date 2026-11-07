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

import heapq
from typing import Any

import torch
from accelerate.utils import gather_object
from ...rewards import accuracy_reward
from trl.rewards.branch_advantages import (
    compute_branch_advantages,
)

from ...data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from ...models.utils import disable_gradient_checkpointing
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import nanmax, nanmin, nanstd, pad
from .grpo_with_replay_buffer_config import GRPOWithReplayBufferConfig


class ReplayBuffer:
    """
    A simple replay buffer to store and sample previously seen rollouts.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Min-heap of (score, insertion_idx, data) tuples
        self._insertion_idx = 0

    def add(self, scores: list[float], data: list[dict]):
        for score, datum in zip(scores, data, strict=True):
            heap_item = (score, self._insertion_idx, datum)
            self._insertion_idx += 1
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, heap_item)
            else:
                # Only add if score is better than worst (minimum) item
                # if score > self.heap[0][0]:
                    # heapq.heapreplace(self.heap, heap_item)
                
                for i in range(len(self.heap)):
                    self.heap[i] = (self.heap[i][0], self.heap[i][1] + 1, self.heap[i][2])
                ind_max = max([self.heap[i][1] for i in range(len(self.heap))])
                ind = [self.heap[i][1] == ind_max for i in range(len(self.heap))].index(True)
                for i in range(len(self.heap)):
                    if self.heap[i][1] == ind:
                        self.heap[ind] = (score, 0, datum)
                
    def sample(self, num_samples: int) -> list[dict[str, torch.Tensor]]:
        if not self.heap:
            return None

        # Sample by normalized scores
        scores = torch.tensor([item[0] for item in self.heap], dtype=torch.float32)
        probabilities = scores / scores.sum()
        replacement = False
        if num_samples > len(self.heap):
            replacement = True
        chosen_indices = torch.multinomial(probabilities, num_samples, replacement=replacement).tolist()
        return [self.heap[i][2] for i in chosen_indices]


class GRPOWithReplayBufferTrainer(GRPOTrainer):
    def __init__(self, args: GRPOWithReplayBufferConfig | None = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.replay_buffer = ReplayBuffer(args.replay_buffer_size) if args.replay_buffer_size > 0 else None
        self._eval_metrics_totals = None

    def _record_eval_metrics_only(
        self,
        inputs: list[dict[str, torch.Tensor | Any]],
        completion_ids_list: list[list[int]],
        tool_mask_list: list[list[int]] | None,
    ) -> None:
        """Keep eval bookkeeping limited to accuracy and mean completion length."""
        num_generations = self.num_generations_eval
        if num_generations <= 0:
            raise ValueError("`num_generations_eval` must be strictly positive during evaluation.")
        if len(inputs) % num_generations != 0:
            raise ValueError(
                "Eval batches must contain a whole number of prompt groups. "
                f"Received batch size {len(inputs)} with num_generations_eval={num_generations}."
            )
        if len(completion_ids_list) != len(inputs):
            raise ValueError(
                "Each eval input should produce exactly one completion after prompt repetition. "
                f"Got {len(completion_ids_list)} completions for {len(inputs)} inputs."
            )

        local_num_prompts = len(inputs) // num_generations
        device = self.accelerator.device

        if tool_mask_list is not None:
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask_list], dtype=torch.float32, device=device)
        else:
            think_token_id, end_think_token_id = self._get_think_token_ids()
            lengths = []
            for ids in completion_ids_list:
                try:
                    start = ids.index(think_token_id) + 1
                except ValueError:
                    start = 0
                try:
                    end = ids.index(end_think_token_id, start)
                except ValueError:
                    end = len(ids)
                length = end - start
                lengths.append(length if length > 0 else 0)
            completion_lengths = torch.tensor(lengths, dtype=torch.float32, device=device)
        completion_lengths = completion_lengths.view(local_num_prompts, num_generations)
        gathered_completion_lengths = self.accelerator.gather_for_metrics(completion_lengths)
        completion_length_sum = gathered_completion_lengths.sum().item()
        completion_length_count = gathered_completion_lengths.numel()

        completion_texts = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
        accuracies = accuracy_reward(
            [[{"role": "assistant", "content": text}] for text in completion_texts],
            complete_answers=[example["answer"] for example in inputs],
            question=[example["question"] for example in inputs],
            solution=[example["solution"] for example in inputs],
        )
        accuracies = torch.tensor(accuracies, dtype=torch.float32, device=device).view(local_num_prompts, num_generations)
        gathered_accuracies = self.accelerator.gather_for_metrics(accuracies)
        accuracy_sum = gathered_accuracies.sum().item()
        accuracy_count = gathered_accuracies.numel()

        if self._eval_metrics_totals is None:
            self._eval_metrics_totals = {
                "accuracy_sum": 0.0,
                "accuracy_count": 0,
                "completion_length_sum": 0.0,
                "completion_length_count": 0,
            }

        self._eval_metrics_totals["accuracy_sum"] += accuracy_sum
        self._eval_metrics_totals["accuracy_count"] += accuracy_count
        self._eval_metrics_totals["completion_length_sum"] += completion_length_sum
        self._eval_metrics_totals["completion_length_count"] += completion_length_count

        self._metrics["eval"].clear()
        self._metrics["eval"]["accuracy"] = [
            self._eval_metrics_totals["accuracy_sum"] / self._eval_metrics_totals["accuracy_count"]
        ]
        self._metrics["eval"]["completions/mean_length"] = [
            self._eval_metrics_totals["completion_length_sum"] / self._eval_metrics_totals["completion_length_count"]
        ]

    def _generate_and_record_eval_metrics(
        self,
        inputs: list[dict[str, torch.Tensor | Any]],
        prompts: list[Any],
        gt_answers: list[str],
        complete_answers: list[str],
    ) -> dict[str, torch.Tensor | Any]:
        (
            _prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            _completions,
            _num_items_in_batch,
            _sampling_per_token_logps_list,
            _extra_fields,
            _accuracies,
        ) = self._generate(prompts, gt_answers, complete_answers)

        self._record_eval_metrics_only(inputs, completion_ids_list, tool_mask_list)
        return {}

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        bfs_po_iterations = self.bfs_po_k + 1 if mode == "train" else 1

        prompts = [x["prompt"] for x in inputs]
        gt_answers = [x["solution"] for x in inputs]
        complete_answers = [x["answer"] for x in inputs]

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
            accuracies
        ) = self._generate(prompts, gt_answers, complete_answers)

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
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
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
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
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

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

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
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
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
        prompts = [prompt.copy() for prompt in prompts for _ in range(bfs_po_iterations)]
        inputs = [inp.copy() for inp in inputs for _ in range(bfs_po_iterations)]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        complete_answers = [inp["answer"] for inp in inputs]

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list, complete_answers)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        num_generations = self.num_generations * bfs_po_iterations if mode == "train" else self.num_generations_eval

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        grouped_std_rewards = rewards.view(-1, num_generations).std(dim=1)
        grouped_std_rewards = grouped_std_rewards.repeat_interleave(num_generations, dim=0)

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = grouped_std_rewards.clone()
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        accuracies = self.accelerator.gather(torch.as_tensor(accuracies, device=device, dtype=rewards.dtype))
        if self._uses_branch_advantage(extra_fields):
            expansions = [extra_fields["expansion"][i] for i in range(0, len(extra_fields["expansion"]), num_generations)]
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
            grouped_accuracies = accuracies.view(-1, num_generations)
            zero_advantage_groups = grouped_accuracies.eq(1).all(dim=1) | grouped_accuracies.eq(0).all(dim=1)
            if zero_advantage_groups.any():
                grouped_advantages = advantages.view(-1, num_generations)
                grouped_advantages[zero_advantage_groups] = 0
                advantages = grouped_advantages.view(-1)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]
        grouped_std_rewards = grouped_std_rewards[process_slice]
        group_advantages = advantages.view(-1, num_generations) if not self._uses_branch_advantage() else advantages
        group_std_rewards = grouped_std_rewards.view(-1, num_generations)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[mask]
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
        outputs_after_sampling_buffer = self.update_with_replay_buffer(
            group_advantages,
            group_std_rewards,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            forward_kwargs,
            num_items_in_batch,
            old_per_token_logps,
            ref_per_token_logps,
            importance_sampling_ratio if self.use_vllm and self.vllm_importance_sampling_correction else None,
            sampling_per_token_logps,
            tool_mask,
            group_size=num_generations,
        )
        if outputs_after_sampling_buffer is not None:
            return outputs_after_sampling_buffer
        else:
            output = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": advantages,
                "num_items_in_batch": num_items_in_batch,
            }
            if old_per_token_logps is not None:
                output["old_per_token_logps"] = old_per_token_logps
            if self.use_vllm and self.vllm_importance_sampling_correction:
                output["importance_sampling_ratio"] = importance_sampling_ratio
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
            if self.tools:
                output["tool_mask"] = tool_mask
            return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        if model.training:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        self._prepare_inputs(inputs)
        return None, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        super().log(logs, start_time)
        if not self.model.training:
            self._eval_metrics_totals = None

    def slice_group_data(
        self, data: torch.Tensor, mask: torch.Tensor, group_idx: int, group_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slices the input data and mask tensors for a specific group index. Also trims the sequence length to the
        maximum length in the group based on the mask.

        Args:
            data: Tensor of shape (num_groups * group_size, seq_length)
            mask: Tensor of shape (num_groups * group_size, seq_length)
            group_idx: Index of the group to slice
            group_size: Number of generations contained in each group
        Returns:
            Tuple of (sliced_data, sliced_mask) for the specified group, with sequence length trimmed to the maximum
            length in the group.
        """
        group_size = group_size or self.num_generations
        start_idx = group_idx * group_size
        end_idx = (group_idx + 1) * group_size
        group_data = data[start_idx:end_idx]
        group_mask = mask[start_idx:end_idx]
        group_max_len = group_mask.sum(dim=1).max().item()
        return group_data[:, :group_max_len], group_mask[:, :group_max_len]

    def update_replay_buffer(
        self,
        groups_with_variance: torch.Tensor,
        group_advantages: torch.Tensor,
        group_std_rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        forward_kwargs: dict,
        optional_vision_fields: list[str] = None,
        old_per_token_logps: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        importance_sampling_ratio: torch.Tensor | None = None,
        sampling_per_token_logps: torch.Tensor | None = None,
        tool_mask: torch.Tensor | None = None,
        group_size: int | None = None,
    ) -> None:
        """
        Update the replay buffer with groups that have reward variance (std > 0).

        Args:
            groups_with_variance: Boolean tensor indicating which groups have reward variance
            group_advantages: Tensor of shape (num_groups, group_size) containing advantage values
            std_rewards: Tensor of shape (num_groups, group_size) containing std of rewards per group
            prompt_ids: Tensor containing prompt token IDs
            prompt_mask: Tensor containing prompt attention masks
            completion_ids: Tensor containing completion token IDs
            completion_mask: Tensor containing completion attention masks
            forward_kwargs: Dictionary containing additional prompt inputs (vision data, etc.)
            optional_vision_fields: List of optional vision-related fields to include if present in forward_kwargs
            old_per_token_logps: Optional tensor of old per-token log probabilities
            ref_per_token_logps: Optional tensor of reference per-token log probabilities
            importance_sampling_ratio: Optional importance sampling correction ratio
            sampling_per_token_logps: Optional tensor of sampling-time per-token log probabilities
            tool_mask: Optional tensor masking model-generated tokens in the completion
            group_size: Number of generations contained in each group
        """
        group_size = group_size or group_advantages.size(1)
        row_has_variance = groups_with_variance
        if len(groups_with_variance.shape) > 1:
            row_has_variance = groups_with_variance.any(dim=1)
        # Prepare buffered outputs for groups with variance
        buffered_outputs = []

        if self._uses_branch_advantage():
            n_samples = group_advantages.shape[0] // group_size
            group_advantages = group_advantages.reshape(n_samples, group_size, -1)

        for _, group_idx in enumerate(row_has_variance.nonzero(as_tuple=True)[0].unique().tolist()):

            group_prompt_ids, group_prompt_mask = self.slice_group_data(prompt_ids, prompt_mask, group_idx, group_size)
            group_completion_ids, group_completion_mask = self.slice_group_data(
                completion_ids, completion_mask, group_idx, group_size
            )

            # Store unpadded data in the buffer
            buffered_output = {
                "prompt_ids": group_prompt_ids,
                "completion_ids": group_completion_ids,
                "advantages": group_advantages[group_idx].clone(),
                "prompt_mask": group_prompt_mask,
                "completion_mask": group_completion_mask,
            }

            # Add optional fields if they exist
            optional_fields = {
                "old_per_token_logps": old_per_token_logps if old_per_token_logps is not None else None,
                "ref_per_token_logps": ref_per_token_logps if ref_per_token_logps is not None else None,
                "importance_sampling_ratio": importance_sampling_ratio if importance_sampling_ratio is not None else None,
                "sampling_per_token_logps": sampling_per_token_logps if sampling_per_token_logps is not None else None,
                "tool_mask": tool_mask if tool_mask is not None else None,
            }
               
            for field_name, field_data in optional_fields.items():
                if field_data is not None:
                    buffered_output[field_name] = self.slice_group_data(
                        field_data, completion_mask, group_idx, group_size
                    )[0]

            if optional_vision_fields:
                # Add vision-related fields if they exist
                for field_name in optional_vision_fields:
                    if field_name in forward_kwargs:
                        buffered_output[field_name] = self.slice_group_data(
                            forward_kwargs[field_name], prompt_mask, group_idx, group_size
                        )[0]

            buffered_outputs.append(buffered_output)

        if row_has_variance.any():
            # Calculate replay buffer scores for groups with variance
            # replay_buffer_scores = (group_advantages.abs() * group_std_rewards).sum(dim=-1)[groups_with_variance]
            # Add all groups to replay buffer at once (batch operation)
            replay_buffer_scores = torch.ones(group_advantages.shape[0]).to(row_has_variance.device)[row_has_variance]
            self.replay_buffer.add(replay_buffer_scores.tolist(), buffered_outputs)


    def sample_from_replay_buffer(
        self, num_samples: int, optional_vision_fields: list[str] = None, optional_tensor_fields: list[str] = None
    ) -> dict[str, list[Any]] | None:
        """
        Sample groups from the replay buffer.

        Args:
            num_samples: Number of samples to draw from the replay buffer
            optional_vision_fields: List of optional vision-related fields to include if present in sampled data
            optional_tensor_fields: List of optional tensor fields to include if present in sampled data
        Returns:
            List of sampled data dictionaries from the replay buffer
        """
        sampled = self.replay_buffer.sample(num_samples=num_samples)
        if not sampled:
            return None

        # Extract and concatenate sampled data
        sampled_data = {
            "prompt_ids": [],
            "prompt_mask": [],
            "completion_ids": [],
            "completion_mask": [],
            "advantages": [],
        }

        all_optional_fields = (optional_tensor_fields or []) + (optional_vision_fields or [])
        # Initialize containers for optional fields if they exist in sampled data
        for field in all_optional_fields:
            if sampled and field in sampled[0]:
                sampled_data[field] = []

        # Extract data from each sampled item
        for item in sampled:
            # Handle core fields
            for key in ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask"]:
                sampled_data[key].append(item[key])

            # Handle per-group advantages
            sampled_data["advantages"].append(item["advantages"])

            # Handle optional fields
            for field in all_optional_fields:
                if field in item:
                    sampled_data[field].append(item[field])

        return sampled_data

    def update_with_replay_buffer(
        self,
        group_advantages: torch.Tensor,
        group_std_rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        forward_kwargs: dict,
        num_items_in_batch: int,
        old_per_token_logps: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        importance_sampling_ratio: torch.Tensor | None = None,
        sampling_per_token_logps: torch.Tensor | None = None,
        tool_mask: torch.Tensor | None = None,
        group_size: int | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """
        Update current batch data with samples from replay buffer.

        Groups with reward variance (std > 0) are added to the replay buffer and then replaced with samples from the
        buffer to improve training stability.

        Args:
            group_advantages: Tensor of shape (num_groups, group_size) containing advantage values
            std_rewards: Tensor of shape (num_groups, group_size) containing std of rewards per group
            prompt_ids: Tensor containing prompt token IDs
            prompt_mask: Tensor containing prompt attention masks
            completion_ids: Tensor containing completion token IDs
            completion_mask: Tensor containing completion attention masks
            forward_kwargs: Dictionary containing additional prompt inputs (vision data, etc.)
            num_items_in_batch: Number of items in the current batch
            old_per_token_logps: Optional tensor of old per-token log probabilities
            ref_per_token_logps: Optional tensor of reference per-token log probabilities
            importance_sampling_ratio: Optional importance sampling correction ratio
            sampling_per_token_logps: Optional tensor of sampling-time per-token log probabilities
            tool_mask: Optional tensor masking model-generated tokens in the completion
            group_size: Number of generations contained in each group
        """
        if self.replay_buffer is None or self.replay_buffer.max_size <= 0:
            return
        group_size = group_size or group_advantages.size(1)
        # TODO: verify case if token-level advantages are disabled.
        group_size = group_size if not self._uses_branch_advantage() else group_size // (self.bfs_po_k + 1)
        # Groups to consider for adding to the replay buffer
        groups_with_variance = torch.abs(group_advantages).max(dim=1).values > 0 if not self._uses_branch_advantage() else \
            (torch.sum(torch.abs(group_advantages), dim=1) != 0).reshape(group_advantages.shape[0] // group_size, -1)
        # Groups to replace from the replay buffer
        row_has_variance = groups_with_variance
        if groups_with_variance.dim() > 1:
            row_has_variance = groups_with_variance.any(dim=1)

        # Track which optional fields are present in sampled data
        optional_tensor_fields = ["old_per_token_logps", "ref_per_token_logps", "sampling_per_token_logps", "tool_mask"]
        if self.use_vllm and self.vllm_importance_sampling_correction:
            optional_tensor_fields.append("importance_sampling_ratio")
        vision_fields = ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"]

        self.update_replay_buffer(
            groups_with_variance,
            group_advantages,
            group_std_rewards,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            forward_kwargs,
            vision_fields,
            old_per_token_logps,
            ref_per_token_logps,
            importance_sampling_ratio,
            sampling_per_token_logps,
            tool_mask,
            group_size,
        )

        # Sample from replay buffer to replace groups without variance
        row_without_variance = ~row_has_variance
        num_groups_to_replace = row_without_variance.sum().item()
        if not num_groups_to_replace:
            return

        sampled_data = self.sample_from_replay_buffer(
            num_samples=num_groups_to_replace,
            optional_vision_fields=vision_fields,
            optional_tensor_fields=optional_tensor_fields,
        )
        if sampled_data is None:
            return

        # Pad sampled data if they are shorter than the current batch sequences
        # Or pad the current batch if sampled are longer
        current_batch_prompt_seq_len = prompt_ids.size(1)
        current_batch_completion_seq_len = completion_ids.size(1)

        groups_to_replace_idxs = row_without_variance.nonzero(as_tuple=True)[0].unique().tolist()

        # Determine target (max) sequence lengths once
        sampled_prompt_lengths = [t.size(1) for t in sampled_data["prompt_ids"]]
        sampled_completion_lengths = [t.size(1) for t in sampled_data["completion_ids"]]
        target_prompt_len = max([current_batch_prompt_seq_len] + sampled_prompt_lengths)
        target_completion_len = max([current_batch_completion_seq_len] + sampled_completion_lengths)

        # If any sampled prompt is longer, pad the whole batch prompt tensors once (left padding)
        if target_prompt_len > current_batch_prompt_seq_len:
            prompt_ids = pad(
                list(prompt_ids.unbind(0)),
                padding_value=self.pad_token_id,
                pad_to_multiple_of=target_prompt_len,
                padding_side="left",
            )
            prompt_mask = pad(
                list(prompt_mask.unbind(0)), padding_value=0, pad_to_multiple_of=target_prompt_len, padding_side="left"
            )
        # If any sampled completion is longer, pad the whole batch completion tensors once (right padding)
        if target_completion_len > current_batch_completion_seq_len:
            completion_ids = pad(
                list(completion_ids.unbind(0)),
                padding_value=self.pad_token_id,
                pad_to_multiple_of=target_completion_len,
                padding_side="right",
            )
            completion_mask = pad(
                list(completion_mask.unbind(0)),
                padding_value=0,
                pad_to_multiple_of=target_completion_len,
                padding_side="right",
            )
            if old_per_token_logps is not None:
                old_per_token_logps = pad(
                    list(old_per_token_logps.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
            if ref_per_token_logps is not None:
                ref_per_token_logps = pad(
                    list(ref_per_token_logps.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
            if importance_sampling_ratio is not None:
                importance_sampling_ratio = pad(
                    list(importance_sampling_ratio.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
            if sampling_per_token_logps is not None:
                sampling_per_token_logps = pad(
                    list(sampling_per_token_logps.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
            if tool_mask is not None:
                tool_mask = pad(
                    list(tool_mask.unbind(0)),
                    padding_value=1,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
            if self._uses_branch_advantage():
                group_advantages = pad(
                    list(group_advantages.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )

        # Replace per-group data, padding only sampled groups that are shorter than the target
        for i, group_idx in enumerate(groups_to_replace_idxs):
            start_idx = group_idx * group_size
            end_idx = (group_idx + 1) * group_size
            idx_range = slice(start_idx, end_idx)

            # Pad sampled prompt to target length if needed
            if sampled_data["prompt_ids"][i].size(1) < target_prompt_len:
                sampled_data["prompt_ids"][i] = pad(
                    sampled_data["prompt_ids"][i],
                    padding_value=self.pad_token_id,
                    pad_to_multiple_of=target_prompt_len,
                    padding_side="left",
                )
                sampled_data["prompt_mask"][i] = pad(
                    sampled_data["prompt_mask"][i],
                    padding_value=0,
                    pad_to_multiple_of=target_prompt_len,
                    padding_side="left",
                )

            # Pad sampled completion to target length if needed
            if sampled_data["completion_ids"][i].size(1) < target_completion_len:
                sampled_data["completion_ids"][i] = pad(
                    sampled_data["completion_ids"][i],
                    padding_value=self.pad_token_id,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
                sampled_data["completion_mask"][i] = pad(
                    sampled_data["completion_mask"][i],
                    padding_value=0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
                if "old_per_token_logps" in sampled_data:
                    sampled_data["old_per_token_logps"][i] = pad(
                        sampled_data["old_per_token_logps"][i],
                        padding_value=0.0,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )
                if "ref_per_token_logps" in sampled_data:
                    sampled_data["ref_per_token_logps"][i] = pad(
                        sampled_data["ref_per_token_logps"][i],
                        padding_value=0.0,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )
                if "importance_sampling_ratio" in sampled_data:
                    sampled_data["importance_sampling_ratio"][i] = pad(
                        sampled_data["importance_sampling_ratio"][i],
                        padding_value=0.0,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )
                if "sampling_per_token_logps" in sampled_data:
                    sampled_data["sampling_per_token_logps"][i] = pad(
                        sampled_data["sampling_per_token_logps"][i],
                        padding_value=0.0,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )
                if "tool_mask" in sampled_data:
                    sampled_data["tool_mask"][i] = pad(
                        sampled_data["tool_mask"][i],
                        padding_value=1,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )

            # Assign (replace) group slice
            prompt_ids[idx_range] = sampled_data["prompt_ids"][i]
            prompt_mask[idx_range] = sampled_data["prompt_mask"][i]
            completion_ids[idx_range] = sampled_data["completion_ids"][i]
            completion_mask[idx_range] = sampled_data["completion_mask"][i]

            sampled_advantages = sampled_data["advantages"][i]
            if self._uses_branch_advantage():
                sampled_advantages_len = sampled_advantages.size(1)
                if sampled_advantages_len < target_completion_len:
                    sampled_advantages = torch.nn.functional.pad(
                        sampled_advantages,
                        (0, target_completion_len - sampled_advantages_len),
                        value=0.0,
                    )
                elif sampled_advantages_len > target_completion_len:
                    sampled_advantages = sampled_advantages.narrow(1, 0, target_completion_len)

            group_advantages[idx_range] = sampled_advantages
            sampled_data["advantages"][i] = None

            if "old_per_token_logps" in sampled_data:
                old_per_token_logps[idx_range] = sampled_data["old_per_token_logps"][i]
            if "ref_per_token_logps" in sampled_data:
                ref_per_token_logps[idx_range] = sampled_data["ref_per_token_logps"][i]
            if "importance_sampling_ratio" in sampled_data and importance_sampling_ratio is not None:
                importance_sampling_ratio[idx_range] = sampled_data["importance_sampling_ratio"][i]
            if "sampling_per_token_logps" in sampled_data and sampling_per_token_logps is not None:
                sampling_per_token_logps[idx_range] = sampled_data["sampling_per_token_logps"][i]
            if "tool_mask" in sampled_data and tool_mask is not None:
                tool_mask[idx_range] = sampled_data["tool_mask"][i]

            for field in vision_fields:
                if field in sampled_data and field in forward_kwargs:
                    forward_kwargs[field][idx_range] = sampled_data[field][i]

        if "token_type_ids" in forward_kwargs:
            forward_kwargs["token_type_ids"] = torch.cat(
                [
                    torch.zeros_like(prompt_ids, dtype=forward_kwargs["token_type_ids"].dtype),
                    torch.zeros_like(completion_ids, dtype=forward_kwargs["token_type_ids"].dtype),
                ],
                dim=1,
            )

        # Prepare final outputs after sampling and replacement
        # TODO: before: "advantages": group_advantages.reshape(-1) if not self.replay_buffer else group_advantages
        outputs_after_sampling_buffer = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": group_advantages.reshape(-1) if not self._uses_branch_advantage() else group_advantages,
        }

        # Replace optional tensor fields if they exist
        for field in optional_tensor_fields:
            if field in sampled_data:
                outputs_after_sampling_buffer[field] = (
                    old_per_token_logps
                    if field == "old_per_token_logps"
                    else ref_per_token_logps
                    if field == "ref_per_token_logps"
                    else importance_sampling_ratio
                    if field == "importance_sampling_ratio"
                    else sampling_per_token_logps
                    if field == "sampling_per_token_logps"
                    else tool_mask
                )

        # Replace vision fields if they exist
        for field in vision_fields:
            if field in sampled_data and field in forward_kwargs:
                outputs_after_sampling_buffer[field] = forward_kwargs[field]
        

        # TODO: I have changed here
        outputs_after_sampling_buffer["num_items_in_batch"] = (
            torch.sum(outputs_after_sampling_buffer["advantages"] != 0)
            if self._uses_branch_advantage()
            else num_items_in_batch
        )
        # outputs_after_sampling_buffer["num_items_in_batch"] = num_items_in_batch
        if self.use_vllm and self.vllm_importance_sampling_correction:
            outputs_after_sampling_buffer["importance_sampling_ratio"] = importance_sampling_ratio

        return outputs_after_sampling_buffer
