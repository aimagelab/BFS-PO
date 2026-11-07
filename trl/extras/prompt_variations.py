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

from ..chat_template_utils import qwen3_training_chat_template
from .dataset_formatting import is_qwen3_1_7b_tokenizer


QWEN3_CONTINUE_FINAL_ASSISTANT_CHAT_TEMPLATE = qwen3_training_chat_template.replace(
    """{%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}""",
    """{%- elif message.role == "assistant" %}
        {%- if loop.last and 'CONTINUE_FINAL_MESSAGE_TAG' in content %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- else %}
        {%- set reasoning_content = '' %}""",
    1,
).replace(
    """        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}""",
    """        {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- elif message.role == "tool" %}""",
    1,
)


def unwrap_tokenizer(processing_class):
    return getattr(processing_class, "tokenizer", processing_class)


def maybe_apply_custom_chat_template(processing_class):
    """Override Qwen3-1.7B only; preserve tokenizer-provided templates for every other model."""
    tokenizer = unwrap_tokenizer(processing_class)
    if not is_qwen3_1_7b_tokenizer(tokenizer):
        return None
    chat_template = QWEN3_CONTINUE_FINAL_ASSISTANT_CHAT_TEMPLATE

    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = chat_template
    if processing_class is not tokenizer and hasattr(processing_class, "chat_template"):
        processing_class.chat_template = chat_template

    return chat_template
