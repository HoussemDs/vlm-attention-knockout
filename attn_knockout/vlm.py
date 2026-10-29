"""Thin wrapper around a vision-language model used for the probe tracing.

The reference the probe code targets LLaVA-1.5 / LLaMA-3.2-Vision / Qwen2.5-VL
(7B-13B). This wrapper is architecture-aware and supports:

* ``HuggingFaceTB/SmolVLM-256M-Instruct`` (Idefics3) - the smallest VLM, CPU friendly;
* ``Qwen/Qwen2.5-VL-3B-Instruct`` - the ~3B Qwen VLM used for the model comparison.

Two things must hold for the attention-blocking engine to work, and both are
enforced here:

1. ``attn_implementation="eager"`` so the additive causal attention mask is
   materialised (our hook rewrites it);
2. a deterministic map between sequence positions and image / text token groups,
   which requires the model's ``image_token_id`` and its text-decoder layers.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Qwen2.5-VL patches are 28x28; 256 patches keeps a chart at ~256 visual tokens,
# which is what makes a full layer sweep feasible on CPU.
QWEN_PATCH = 28
QWEN_DEFAULT_MAX_PIXELS = QWEN_PATCH * QWEN_PATCH * 256

_TEXT_LAYER_PATHS = (
    ("model", "text_model", "layers"),     # SmolVLM / Idefics3
    ("model", "language_model", "layers"),  # Qwen2.5-VL
    ("language_model", "layers"),
    ("model", "layers"),                    # plain Llama-style
    ("model", "model", "layers"),
    ("language_model", "model", "layers"),
)

# Weight-only quantization backends that work on CPU.
QUANTIZATION_CHOICES = ("none", "int8", "int4")


def _build_quantization_config(quantization: str | None):
    """Map a short name to a transformers quantization config (CPU friendly)."""
    if quantization in (None, "none"):
        return None
    q = quantization.lower()
    if q in ("int8", "q8", "8bit"):
        from transformers import QuantoConfig  # optimum-quanto

        return QuantoConfig(weights="int8")
    if q in ("int4", "q4", "4bit"):
        from transformers import HqqConfig  # hqq

        return HqqConfig(nbits=4, group_size=64)
    raise ValueError(f"unknown quantization {quantization!r}; choices: {', '.join(QUANTIZATION_CHOICES)}")


class SmallVLM:
    """Uniform interface over the supported VLMs (name kept for compatibility)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype="auto",
        image_longest_edge: int = 512,
        image_max_pixels: int | None = None,
        quantization: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.quantization = quantization

        quantization_config = _build_quantization_config(quantization)
        load_kwargs = {"attn_implementation": "eager"}
        if quantization_config is None:
            load_kwargs["dtype"] = dtype
        else:
            load_kwargs["quantization_config"] = quantization_config
            # quantized weights compute against float activations; fp32 is the
            # safest and fastest choice on CPU.
            load_kwargs["dtype"] = torch.float32 if dtype == "auto" else dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).eval()
        if quantization_config is None:
            self.model = self.model.to(self.device)

        self.model_type = getattr(self.model.config, "model_type", "")
        self._is_qwen = "qwen" in self.model_type
        self._is_gemma = "gemma" in self.model_type

        self._set_image_resolution(image_longest_edge, image_max_pixels)
        self.image_token_id = self._resolve_image_token_id()
        self._layers = self._resolve_text_layers()

    # -- setup helpers --------------------------------------------------------
    def _set_image_resolution(self, longest_edge: int, max_pixels: int | None) -> None:
        """Keep the number of visual tokens small so the layer sweep stays cheap."""
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            return
        if self._is_qwen:
            max_pixels = max_pixels or QWEN_DEFAULT_MAX_PIXELS
            image_processor.max_pixels = max_pixels
            image_processor.min_pixels = min(getattr(image_processor, "min_pixels", max_pixels), max_pixels)
            return
        if self._is_gemma:
            # Gemma3 / MedGemma encode one image to a fixed 256 tokens. Disable
            # pan-and-scan so a single crop is used (otherwise extra crops
            # multiply the visual-token count and blow up the CPU layer sweep).
            for obj in (image_processor, getattr(self.processor, "do_pan_and_scan", None) and self.processor):
                if obj is not None and hasattr(obj, "do_pan_and_scan"):
                    obj.do_pan_and_scan = False
            return
        # Idefics3 / SmolVLM: cap the resize so a single 512px patch is used.
        try:
            image_processor.size = {"longest_edge": longest_edge}
            image_processor.max_image_size = min(getattr(image_processor, "max_image_size", 512), longest_edge)
        except Exception:
            pass

    def _resolve_image_token_id(self) -> int:
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                return val
        for token in ("<image>", "<|image_pad|>"):
            tid = self.processor.tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                return tid
        raise RuntimeError("could not resolve the image token id")

    def _resolve_text_layers(self) -> List[torch.nn.Module]:
        for path in _TEXT_LAYER_PATHS:
            obj = self.model
            for part in path:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        raise RuntimeError("could not locate the text-decoder layers of the model")

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def text_layers(self) -> List[torch.nn.Module]:
        return self._layers

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def build_inputs(self, image, question: str) -> Dict:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, inputs: Dict, max_new_tokens: int = 64):
        """Greedy generation under full attention; returns (full_ids, prompt_len)."""
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        return out, prompt_len

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor, inputs: Dict) -> torch.Tensor:
        """Forward a full sequence (no KV cache) and return logits (1, S, vocab)."""
        extra = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "token_type_ids")
        }
        attention_mask = torch.ones_like(input_ids, device=self.device)
        # The processor's token_type_ids has the ORIGINAL prompt length, but the
        # traced sequence is longer (prompt + generated tokens). Passing the stale
        # tensor makes Gemma3/MedGemma index out of bounds, so rebuild it to match
        # input_ids (Gemma3 convention: 1 = image token, 0 = text).
        if inputs.get("token_type_ids") is not None:
            extra["token_type_ids"] = (input_ids == self.image_token_id).long()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **extra)
        return out.logits

    def decode(self, token_ids) -> str:
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

    # -- closed (yes/no) scoring ---------------------------------------------
    _YES_VARIANTS = ("yes", "Yes", "YES", " yes", " Yes", " YES")
    _NO_VARIANTS = ("no", "No", "NO", " no", " No", " NO")

    def _first_token_ids(self, words) -> List[int]:
        ids = set()
        for w in words:
            toks = self.processor.tokenizer.encode(w, add_special_tokens=False)
            if toks:
                ids.add(int(toks[0]))
        return sorted(ids)

    @torch.no_grad()
    def score_closed(self, logits_row: torch.Tensor) -> Dict:
        """Principled yes/no scoring from a single logit row (no generation).

        logits_row is the distribution over the FIRST answer token (the baseline
        logit row at prompt_len-1). Aggregates probability mass over casing/space
        variants of yes and no, so the verdict is robust to tokenization.
        Returns prediction, the two log-probs, and the decision margin (nats).
        """
        logp = torch.log_softmax(logits_row.float(), dim=-1)
        yes_ids = self._first_token_ids(self._YES_VARIANTS)
        no_ids = self._first_token_ids(self._NO_VARIANTS)
        lp_yes = torch.logsumexp(logp[yes_ids], dim=-1) if yes_ids else torch.tensor(float("-inf"))
        lp_no = torch.logsumexp(logp[no_ids], dim=-1) if no_ids else torch.tensor(float("-inf"))
        pred = "yes" if lp_yes >= lp_no else "no"
        margin = float(abs(lp_yes - lp_no))
        return {
            "closed_pred": pred,
            "closed_logp_yes": float(lp_yes),
            "closed_logp_no": float(lp_no),
            "closed_margin": margin,
        }
