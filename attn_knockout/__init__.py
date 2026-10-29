"""Per-layer attention-knockout probe for vision-language models.

Blocks image-to-answer or question-to-answer attention one decoder layer at a time
and measures how much information about the answer flows through each pathway.
Used for the Gemma-3-4B vs MedGemma-4B study on VQA-RAD.
"""

from .attention_blocking import BLOCK_TYPES, AttentionBlocker, build_block_mask
from .vlm import SmallVLM, DEFAULT_MODEL_ID
from .tracing import trace_sample
from .metrics import normalized_information_flow, group_tokens_into_words

__all__ = [
    "BLOCK_TYPES",
    "AttentionBlocker",
    "build_block_mask",
    "SmallVLM",
    "DEFAULT_MODEL_ID",
    "trace_sample",
    "normalized_information_flow",
    "group_tokens_into_words",
]

__version__ = "1.0.0"
