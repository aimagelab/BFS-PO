from bisect import bisect_right  # We use binary search to find the next split token after a given position.
from functools import lru_cache  # We memoize subtree computations because the same branch can be queried many times.

import torch  # We stack scalar branch values into tensors without leaving the current device.


def compute_branch_advantages(
    rewards,
    expansions,
    completion_ids_list,
    advantages_shape,
    bfs_po_k,
    num_generations,
    accuracies,
):
    """Compute per-token subtree advantages normalized by all leaf rewards, inheriting parent prefixes.

    `bfs_po_k` counts backtracking rounds after the initial generation.
    """
    advantages = rewards.new_zeros((len(completion_ids_list), advantages_shape))
    seq_lengths = [min(len(ids), advantages_shape) for ids in completion_ids_list]
    bfs_po_iterations = bfs_po_k + 1
    branch_per_steps = num_generations // bfs_po_iterations
    num_prompt_groups = len(completion_ids_list) // num_generations
    grouped_accuracies = torch.as_tensor(accuracies, device=rewards.device).view(-1, num_generations)
    # Check every completion in the tree, including descendants of non-root expansions.
    uniform_accuracy_groups = (
        grouped_accuracies.eq(0).all(dim=1) | grouped_accuracies.eq(1).all(dim=1)
    ).tolist()

    for prompt_group_idx in range(num_prompt_groups):
        if uniform_accuracy_groups[prompt_group_idx]:
            continue
        group_start = prompt_group_idx * num_generations
        group_end = group_start + num_generations
        group_expansions = list(expansions[prompt_group_idx]) if prompt_group_idx < len(expansions) else []
        if len(group_expansions) == bfs_po_iterations - 2:
            group_expansions = [[0, 0], *group_expansions]

        root_children = list(range(group_start, min(group_start + branch_per_steps, group_end)))
        children_by_node = {}
        child_tokens_by_completion = {}
        prefix_copies = []
        for step in range(1, bfs_po_iterations):
            block_start = group_start + step * branch_per_steps
            if block_start >= group_end:
                break
            child_indices = list(range(block_start, min(block_start + branch_per_steps, group_end)))
            parent_local_idx, token_idx = group_expansions[step - 1] if step - 1 < len(group_expansions) else (0, 0)
            parent_local_idx, token_idx = int(parent_local_idx), int(token_idx)
            parent_idx = group_start + parent_local_idx
            if (parent_local_idx == 0 and token_idx == 0) or not group_start <= parent_idx < group_end:
                root_children.extend(child_indices)
                continue
            children_by_node.setdefault((parent_idx, token_idx), []).extend(child_indices)
            child_tokens_by_completion.setdefault(parent_idx, set()).add(token_idx)
            prefix_copies.extend((child_idx, parent_idx, token_idx) for child_idx in child_indices)

        child_tokens_by_completion = {
            completion_idx: tuple(sorted(token_indices))
            for completion_idx, token_indices in child_tokens_by_completion.items()
        }

        @lru_cache(maxsize=None)
        def subtree_stats(completion_idx, start_token):
            reward_sum = rewards[completion_idx]
            value_count = 1
            for token_idx in child_tokens_by_completion.get(completion_idx, ()):
                if token_idx <= start_token:
                    continue
                for child_idx in children_by_node[(completion_idx, token_idx)]:
                    child_reward_sum, child_count = subtree_stats(child_idx, token_idx)
                    reward_sum = reward_sum + child_reward_sum
                    value_count += child_count
            return reward_sum, value_count

        branch_values = []
        segments = []

        def add_branch(completion_idx, start_token):
            reward_sum, value_count = subtree_stats(completion_idx, start_token)
            branch_values.append(reward_sum / value_count)
            split_tokens = child_tokens_by_completion.get(completion_idx, ())
            next_idx = bisect_right(split_tokens, start_token)
            end_token = seq_lengths[completion_idx]
            if next_idx < len(split_tokens):
                end_token = min(split_tokens[next_idx], end_token)
            segments.append((completion_idx, start_token, end_token))

        for completion_idx in root_children:
            add_branch(completion_idx, 0)
        for (parent_idx, token_idx), child_indices in sorted(children_by_node.items()):
            add_branch(parent_idx, token_idx)
            for child_idx in child_indices:
                add_branch(child_idx, token_idx)

        if branch_values:
            branch_values = torch.stack(branch_values)
            # Normalize every branch using all leaf rewards in this prompt's tree, each counted once.
            leaf_rewards = rewards[group_start:group_end]
            leaf_std = leaf_rewards.std() if leaf_rewards.numel() > 1 else leaf_rewards.new_zeros(())
            branch_advantages = (branch_values - leaf_rewards.mean()) / (leaf_std + 1e-4)
            for (completion_idx, start_token, end_token), branch_advantage in zip(
                segments, branch_advantages, strict=True
            ):
                if end_token > start_token:
                    advantages[completion_idx, start_token:end_token] = branch_advantage

        for child_idx, parent_idx, token_idx in prefix_copies:
            prefix_end = min(token_idx, seq_lengths[child_idx], seq_lengths[parent_idx])
            if prefix_end > 0:
                advantages[child_idx, :prefix_end] = advantages[parent_idx, :prefix_end]

    return advantages
