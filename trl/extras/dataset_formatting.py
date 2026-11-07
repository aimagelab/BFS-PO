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

import functools
import re

import datasets
from datasets import Value
from packaging.version import Version

SYSTEM_PROMPT = "You are a helpful assistant."
REASONING_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
MULTIPLE_CHOICE_REASONING_INSTRUCTION = (
    "Please reason step by step, choose the correct answer from options A, B, C, and D, "
    "and put only the answer letter within \\boxed{}."
)
_MULTIPLE_CHOICE_OPTION_LABELS = ("A", "B", "C", "D")


if Version(datasets.__version__) >= Version("4.0.0"):
    from datasets import List

    FORMAT_MAPPING = {
        "chatml": List({"content": Value(dtype="string", id=None), "role": Value(dtype="string", id=None)}),
        "instruction": {"completion": Value(dtype="string", id=None), "prompt": Value(dtype="string", id=None)},
    }
else:
    FORMAT_MAPPING = {
        "chatml": [{"content": Value(dtype="string", id=None), "role": Value(dtype="string", id=None)}],
        "instruction": {"completion": Value(dtype="string", id=None), "prompt": Value(dtype="string", id=None)},
    }


def preprocess_gsm8k(example, tokenizer=None, has_think_token=None):
    answer = example["answer"].split("####")[-1].strip()

    return {
        "question": str(example["question"]),
        "prompt": _build_reasoning_prompt(
            example["question"],
            tokenizer=tokenizer,
            has_think_token=has_think_token,
        ),
        "answer": _normalize_full_answer(example["answer"]),
        "solution": answer,
    }


def _unwrap_tokenizer(tokenizer):
    if tokenizer is None:
        return None
    return getattr(tokenizer, "tokenizer", tokenizer)


def _get_tokenizer_name_or_path(tokenizer):
    tokenizer = _unwrap_tokenizer(tokenizer)
    if tokenizer is None:
        return ""
    return str(getattr(tokenizer, "name_or_path", "") or "").lower()


def is_qwen3_1_7b_tokenizer(tokenizer):
    """Select the custom prompt for Qwen3-1.7B, including Base variants and named local checkpoints."""
    name_or_path = _get_tokenizer_name_or_path(tokenizer)
    return bool(re.search(r"(?:^|[/\\])qwen3[-_]1[._]7b(?:$|[/\\_-])", name_or_path))


def tokenizer_uses_think_tokens(tokenizer):
    """Assume the Qwen3 experiment always uses thinking delimiters."""
    return True


def _has_think_token(tokenizer):
    return tokenizer_uses_think_tokens(tokenizer)


def _build_reasoning_prompt(
    question,
    tokenizer=None,
    has_think_token=None,
    instruction=REASONING_INSTRUCTION,
):
    user_content = str(question) if not instruction else f"{instruction}\n{question}"
    # All other models receive one user message; their tokenizer owns the chat formatting.
    if not is_qwen3_1_7b_tokenizer(tokenizer):
        return [{"content": user_content, "role": "user"}]

    # Preserve the existing Qwen3-1.7B system message and assistant reasoning prefill.
    prompt = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": user_content, "role": "user"},
    ]
    if has_think_token is None:
        has_think_token = _has_think_token(tokenizer)
    if has_think_token:
        prompt.append({"content": "<think>\n", "role": "assistant"})
    return prompt


def _strip_reasoning_instruction(text):
    if not isinstance(text, str):
        return text

    stripped = text.strip()
    for instruction in (REASONING_INSTRUCTION, MULTIPLE_CHOICE_REASONING_INSTRUCTION):
        prefix = instruction.strip()
        while stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].lstrip()
    return stripped


def _build_trl_example(
    question,
    solution,
    answer="",
    tokenizer=None,
    has_think_token=None,
    instruction=REASONING_INSTRUCTION,
):
    return {
        "question": str(question),
        "prompt": _build_reasoning_prompt(
            str(question),
            tokenizer=tokenizer,
            has_think_token=has_think_token,
            instruction=instruction,
        ),
        "answer": _normalize_full_answer(answer),
        "solution": solution,
    }


def _extract_prompt_question(prompt):
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, str):
                return _strip_reasoning_instruction(content)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        text_parts.append(str(item["text"]))
                if text_parts:
                    return _strip_reasoning_instruction("\n".join(text_parts))

        if prompt and isinstance(prompt[0], dict):
            content = prompt[0].get("content")
            if isinstance(content, str):
                return _strip_reasoning_instruction(content)

    raise ValueError("Unable to extract a user question from the dataset prompt.")


def _get_first_present(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    available_keys = tuple(example.keys())
    raise KeyError(
        f"None of the expected keys {keys} were found in dataset example. "
        f"Available keys: {available_keys}"
    )


def _get_first_present_or_default(example, keys, default=""):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return default


def _extract_last_boxed_answer(text):
    for marker in ("\\boxed{", "\\fbox{"):
        start = text.rfind(marker)
        if start == -1:
            continue

        depth = 1
        i = start + len(marker)
        extracted = []
        while i < len(text):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(extracted).strip()
            if depth > 0:
                extracted.append(char)
            i += 1
    return None


def _normalize_solution(solution):
    if isinstance(solution, list):
        if len(solution) == 1:
            solution = solution[0]
        else:
            solution = " ".join(str(item) for item in solution)

    if solution is None:
        raise ValueError("Dataset solution cannot be null after preprocessing.")

    if not isinstance(solution, str):
        return str(solution)

    stripped = solution.strip()
    if not stripped:
        return stripped

    boxed = _extract_last_boxed_answer(stripped)
    if boxed is not None:
        return boxed

    final_answer_match = re.search(r"Final answer:\s*(.+)", stripped, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    return stripped


def _normalize_full_answer(answer):
    if isinstance(answer, list):
        if len(answer) == 1:
            answer = answer[0]
        else:
            answer = " ".join(str(item) for item in answer)

    if answer is None:
        return ""

    if not isinstance(answer, str):
        return str(answer)

    return answer.strip()


def _ensure_boxed_answer(answer):
    answer = _normalize_full_answer(answer)
    if not answer:
        return answer

    if _extract_last_boxed_answer(answer) is not None:
        return answer

    return f"\\boxed{{{answer}}}"


def _normalize_multiple_choice_answer(answer, choices):
    num_choices = len(choices)
    if num_choices != len(_MULTIPLE_CHOICE_OPTION_LABELS):
        raise ValueError(
            "MMLU-STEM preprocessing expects exactly 4 choices to map them to options A, B, C, and D."
        )

    normalized_choices = [str(choice).strip() for choice in choices]

    if isinstance(answer, str):
        stripped = answer.strip()
        upper = stripped.upper()
        if upper in _MULTIPLE_CHOICE_OPTION_LABELS:
            return upper
        for idx, choice in enumerate(normalized_choices):
            if stripped == choice:
                return _MULTIPLE_CHOICE_OPTION_LABELS[idx]
        if stripped.isdigit():
            answer = int(stripped)
        else:
            raise ValueError(
                "Unsupported MMLU-STEM answer format. Expected a choice index, option letter, or exact option text."
            )

    if isinstance(answer, float):
        if not answer.is_integer():
            raise ValueError(
                "Unsupported non-integer MMLU-STEM answer format. Expected an integer index in [0, 3]."
            )
        answer = int(answer)

    if isinstance(answer, int):
        if 0 <= answer < num_choices:
            return _MULTIPLE_CHOICE_OPTION_LABELS[answer]
        if 1 <= answer <= num_choices:
            return _MULTIPLE_CHOICE_OPTION_LABELS[answer - 1]

    raise ValueError("Unsupported MMLU-STEM answer format. Expected an integer index in [0, 3] or a letter in A-D.")


def _format_multiple_choice_question(question, choices, subject=None):
    if len(choices) != len(_MULTIPLE_CHOICE_OPTION_LABELS):
        raise ValueError(
            "MMLU-STEM preprocessing expects exactly 4 choices to render the multiple-choice prompt."
        )

    parts = []
    if subject:
        normalized_subject = str(subject).replace("_", " ").strip()
        if normalized_subject:
            parts.append(f"The following is a multiple choice question about {normalized_subject}.")
            parts.append("")

    parts.append(str(question).strip())
    parts.append("")
    parts.extend(
        f"{label}. {choice}" for label, choice in zip(_MULTIPLE_CHOICE_OPTION_LABELS, choices, strict=True)
    )
    return "\n".join(parts)


_METAMATH_FINAL_ANSWER_RE = re.compile(r"the answer is:\s*", re.IGNORECASE)


def _last_metamath_final_answer_match(response):
    last_match = None
    for match in _METAMATH_FINAL_ANSWER_RE.finditer(response):
        last_match = match
    return last_match


def _extract_metamath_solution(response):
    response = _normalize_full_answer(response)
    if not response:
        return response

    answer_match = _last_metamath_final_answer_match(response)
    if answer_match is not None:
        final_answer = response[answer_match.end() :].strip()
        if final_answer:
            boxed = _extract_last_boxed_answer(final_answer)
            return boxed if boxed is not None else final_answer

    boxed = _extract_last_boxed_answer(response)
    if boxed is not None:
        return boxed

    if "####" in response:
        tail = response.split("####")[-1].strip()
        tail = re.split(r"(?i)the answer is:", tail)[-1].strip()
        return tail or response.strip()

    answer_match = re.search(r"(?:Answer is|Answer:)\s*([^\n]*)", response, re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).strip()
        if candidate:
            return candidate

    return _normalize_solution(response)


def _box_metamath_final_answer(response):
    response = _normalize_full_answer(response)
    match = _last_metamath_final_answer_match(response)
    if match is None:
        return response

    final_answer_with_trailing = response[match.end() :]
    final_answer = final_answer_with_trailing.rstrip()
    if not final_answer or "\\boxed{" in final_answer or "\\fbox{" in final_answer:
        return response

    trailing = final_answer_with_trailing[len(final_answer) :]
    return f"{response[:match.end()]}\\boxed{{{final_answer}}}{trailing}"


def preprocess_math500(example, tokenizer=None, has_think_token=None):
    question = _get_first_present(example, ("problem", "question"))
    answer = _get_first_present(example, ("answer", "solution"))
    full_answer = _get_first_present_or_default(example, ("solution",))
    return _build_trl_example(
        question,
        _normalize_solution(answer),
        full_answer,
        tokenizer=tokenizer,
        has_think_token=has_think_token,
    )


def preprocess_amc23(example, tokenizer=None, has_think_token=None):
    question = _get_first_present(example, ("question", "problem"))
    answer = _get_first_present(example, ("answer", "solution"))
    full_answer = _get_first_present_or_default(example, ("solution",), default="")
    if not full_answer:
        full_answer = _ensure_boxed_answer(answer)
    return _build_trl_example(
        question,
        _normalize_solution(answer),
        full_answer,
        tokenizer=tokenizer,
        has_think_token=has_think_token,
    )


def preprocess_mmlu_stem(example, tokenizer=None, has_think_token=None):
    question = _get_first_present(example, ("question", "problem"))
    choices = _get_first_present(example, ("choices", "options"))
    subject = _get_first_present_or_default(example, ("subject",), default="")
    solution = _normalize_multiple_choice_answer(
        _get_first_present(example, ("answer", "label", "target", "solution")),
        choices,
    )
    formatted_question = _format_multiple_choice_question(question, choices, subject=subject)
    return _build_trl_example(
        formatted_question,
        solution,
        _ensure_boxed_answer(solution),
        tokenizer=tokenizer,
        has_think_token=has_think_token,
        instruction=MULTIPLE_CHOICE_REASONING_INSTRUCTION,
    )


def preprocess_metamathqa(example, tokenizer=None, has_think_token=None):
    question = _get_first_present_or_default(example, ("query", "original_question", "question", "problem"))
    if isinstance(question, str) and not question.strip():
        question = _get_first_present(example, ("original_question", "question", "problem"))

    response = _box_metamath_final_answer(
        _get_first_present(example, ("response", "answer", "solution", "completion"))
    )
    solution = _extract_metamath_solution(response)
    return _build_trl_example(
        question,
        solution,
        response,
        tokenizer=tokenizer,
        has_think_token=has_think_token,
    )


def apply_reasoning_template(
    example, tokenizer=None, has_think_token=None
):
    question = _extract_prompt_question(example["prompt"])
    return {
        **example,
        "prompt": _build_reasoning_prompt(
            question,
            tokenizer=tokenizer,
            has_think_token=has_think_token,
        ),
    }


DATASET_ALIASES = {
    "gsm8k": {
        "dataset_name": "gsm8k",
        "dataset_config": "main",
        "dataset_split": "train",
    },
    "math-500": {
        "dataset_name": "math-ai/math500",
        "dataset_config": None,
        "dataset_split": "test",
    },
    "amc2023": {
        "dataset_name": "math-ai/amc23",
        "dataset_config": None,
        "dataset_split": "test",
    },
    "amc23": {
        "dataset_name": "math-ai/amc23",
        "dataset_config": None,
        "dataset_split": "test",
    },
    "mmlu-stem": {
        "dataset_name": "TIGER-Lab/MMLU-STEM",
        "dataset_config": None,
        "dataset_split": "test",
    },
    "mmlustem": {
        "dataset_name": "TIGER-Lab/MMLU-STEM",
        "dataset_config": None,
        "dataset_split": "test",
    },
    "metamathqa": {
        "dataset_name": "meta-math/MetaMathQA",
        "dataset_config": None,
        "dataset_split": "train",
    },
}


DATASET_PREPROCESSORS = {
    "gsm8k": preprocess_gsm8k,
    "math-ai/math500": preprocess_math500,
    "math-ai/amc23": preprocess_amc23,
    "TIGER-Lab/MMLU-STEM": preprocess_mmlu_stem,
    "meta-math/MetaMathQA": preprocess_metamathqa,
    "FiorenzoParascandolo/MetaMathQA-60k": preprocess_metamathqa,
}


def _wrap_preprocessing_fn(preprocessing_fn, tokenizer):
    if preprocessing_fn is None or tokenizer is None:
        return preprocessing_fn
    has_think_token = _has_think_token(tokenizer)
    return functools.partial(
        preprocessing_fn,
        # Preserve model identity so dataset loading and chat-template selection use the same policy.
        tokenizer=tokenizer,
        has_think_token=has_think_token,
    )


def resolve_reasoning_dataset(dataset_name, dataset_config=None, dataset_split=None, tokenizer=None):
    alias = DATASET_ALIASES.get(dataset_name)
    if alias is None and isinstance(dataset_name, str):
        alias = DATASET_ALIASES.get(dataset_name.lower())
    resolved_name = alias["dataset_name"] if alias else dataset_name
    resolved_config = dataset_config if dataset_config is not None else alias["dataset_config"] if alias else None
    resolved_split = dataset_split if dataset_split is not None else alias["dataset_split"] if alias else None
    preprocessing_fn = DATASET_PREPROCESSORS.get(resolved_name, DATASET_PREPROCESSORS.get(dataset_name))
    if preprocessing_fn is None and isinstance(dataset_name, str):
        preprocessing_fn = DATASET_PREPROCESSORS.get(dataset_name.lower())
    preprocessing_fn = _wrap_preprocessing_fn(preprocessing_fn, tokenizer)
    load_from_cache_file = True

    return {
        "dataset_name": resolved_name,
        "dataset_config": resolved_config,
        "dataset_split": resolved_split,
        "preprocessing_fn": preprocessing_fn,
        "load_from_cache_file": load_from_cache_file,
    }
