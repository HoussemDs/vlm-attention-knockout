"""Attention masking ("blocking") engine - the heart of the probe.

The framework isolates the contribution of a modality (or of the generated
prefix) by *knocking out* the attention edges that flow from a **source** token
group to a **target** token group, then measuring how the probability of the
free-form answer changes.

Mechanically this is implemented with `forward_pre_hook`s installed on the
selected text-decoder layers. Each hook rewrites the 4-D additive causal mask
that the decoder already passes down to its attention module, setting the
blocked (query, key) pairs to the dtype minimum so that softmax gives them
~0 weight.

Five block types are supported (identical to the reference implementation):

    full_attention     : no intervention (baseline)
    image_to_last      : queries of the generated tokens cannot see image tokens
    question_to_last   : queries of the generated tokens cannot see prompt text
    last_to_last       : queries of the generated tokens cannot see earlier
                         generated tokens (autoregressive context knockout)
    image_to_question  : question queries cannot see image tokens
"""

from __future__ import annotations

from typing import List, Sequence

import torch

BLOCK_TYPES = (
    "full_attention",
    "last_to_last",
    "question_to_last",
    "image_to_last",
    "image_to_question",
)


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    """Locate the self-attention sub-module of a decoder layer."""
    for name in ("self_attn", "attention", "attn"):
        mod = getattr(layer, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    raise RuntimeError(f"could not find an attention module inside {type(layer).__name__}")


def build_block_mask(
    block_type: str,
    seq_len: int,
    image_idx: Sequence[int],
    question_idx: Sequence[int],
    pred_positions: Sequence[int],
    gen_token_positions: Sequence[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a boolean (seq_len, seq_len) mask: True == blocked attention edge.

    Rows are query positions, columns are key positions.
    """
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    if block_type == "full_attention":
        return mask

    if block_type in ("image_to_last", "question_to_last"):
        src = image_idx if block_type == "image_to_last" else question_idx
        if len(src):
            cols = torch.as_tensor(list(src), dtype=torch.long, device=device)
            rows = torch.as_tensor(list(pred_positions), dtype=torch.long, device=device)
            mask[rows[:, None], cols[None, :]] = True

    elif block_type == "last_to_last":
        # Each generated query may only be prevented from seeing *earlier*
        # generated tokens (columns prompt_len .. q-1).
        for q in pred_positions:
            earlier = [s for s in gen_token_positions if s < q]
            if earlier:
                mask[q, torch.as_tensor(earlier, dtype=torch.long, device=device)] = True

    elif block_type == "image_to_question":
        if len(image_idx):
            cols = torch.as_tensor(list(image_idx), dtype=torch.long, device=device)
            rows = torch.as_tensor(list(question_idx), dtype=torch.long, device=device)
            if rows.numel():
                mask[rows[:, None], cols[None, :]] = True

    else:
        raise ValueError(f"unknown block_type {block_type!r}; expected one of {BLOCK_TYPES}")

    return mask


class AttentionBlocker:
    """Context manager that applies a blocking mask to a set of decoder layers.

    ``block_mask`` is a boolean (seq_len, seq_len) tensor broadcastable to the
    model's 4-D additive attention mask.
    """

    def __init__(self, layers: Sequence[torch.nn.Module], block_mask: torch.Tensor):
        self.layers = list(layers)
        self.block = block_mask.view(1, 1, *block_mask.shape)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- hook -----------------------------------------------------------------
    def _hook(self, module, args, kwargs):
        attn_mask = kwargs.get("attention_mask", None)
        seq_len = self.block.shape[-1]

        if attn_mask is None:
            # Fallback: materialise a plain causal mask ourselves (only happens
            # if the backend did not pre-build one, e.g. lazy SDPA paths).
            hidden = args[0] if args else kwargs.get("hidden_states")
            q_len = hidden.shape[1]
            q = torch.arange(q_len, device=hidden.device).view(-1, 1)
            k = torch.arange(seq_len, device=hidden.device).view(1, -1)
            dtype = hidden.dtype if torch.is_floating_point(hidden) else torch.float32
            attn_mask = torch.where(k <= q, torch.zeros_like(q * k, dtype=dtype),
                                    torch.full_like(q * k, torch.finfo(dtype).min, dtype=dtype))
            attn_mask = attn_mask.view(1, 1, q_len, seq_len)

        if not torch.is_floating_point(attn_mask):
            attn_mask = attn_mask.to(torch.float32)

        block = self.block[:, :, : attn_mask.shape[-2], : attn_mask.shape[-1]]
        kwargs["attention_mask"] = attn_mask.masked_fill(block, torch.finfo(attn_mask.dtype).min)
        return args, kwargs

    # -- context manager ------------------------------------------------------
    def __enter__(self):
        for layer in self.layers:
            handle = get_attention_module(layer).register_forward_pre_hook(self._hook, with_kwargs=True)
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
