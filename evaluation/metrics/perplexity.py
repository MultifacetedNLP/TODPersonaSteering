"""Dialog-level perplexity metric.

Loads a causal LM once and computes per-token cross-entropy loss over the
natural-language utterances in a set of Activation-Steered-Personas dialog JSON files.  API-call
lines, search-result blocks, and empty turns are skipped so the score
reflects only fluency of generated natural language.

Typical usage (from entrypoints/evaluation/perplexity_eval.py):

    from evaluation.metrics.perplexity import DialogPerplexityEvaluator
    evaluator = DialogPerplexityEvaluator(model_name="Qwen/Qwen2.5-7B-Instruct")
    result = evaluator.evaluate_run(run_dir)
    # result = {"perplexity": float, "avg_loss": float, "total_tokens": int, ...}
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

# Regexes that identify non-natural-language lines in generated_dialog
_API_CALL_RE = re.compile(r"^\s*API[Cc]all\(", re.IGNORECASE)
_SEARCH_START_RE = re.compile(r"^\s*Search Results", re.IGNORECASE)
_SEARCH_END_RE = re.compile(r"^\s*End Search Results", re.IGNORECASE)

# Strip role prefixes ("User: ", "System: ") and end-of-dialog markers
_ROLE_PREFIX_RE = re.compile(r"^(?:User|System):\s*", re.IGNORECASE)
_EOD_RE = re.compile(r"<\|endofdialog\|>", re.IGNORECASE)

# Min chars for a turn to be scored (skip trivially short fragments)
_MIN_CHARS = 5


def _is_nl_turn(turn: str) -> bool:
    """Return True if the turn contains natural-language text worth scoring."""
    t = turn.strip()
    if not t:
        return False
    if _SEARCH_START_RE.match(t) or _SEARCH_END_RE.match(t):
        return False
    # Pure search-result data blocks (start with "[{")
    if t.startswith("[{") or t.startswith("Search Results"):
        return False
    # Strip role prefix then check for API calls
    nl = _ROLE_PREFIX_RE.sub("", t)
    if _API_CALL_RE.match(nl):
        return False
    nl = _EOD_RE.sub("", nl).strip()
    return len(nl) >= _MIN_CHARS


def _extract_nl(turn: str) -> str:
    """Strip role prefix and EOD marker, return plain NL text."""
    t = _ROLE_PREFIX_RE.sub("", turn.strip())
    t = _EOD_RE.sub("", t).strip()
    return t


def _collect_nl_texts_from_run(run_dir: Path) -> list[str]:
    """Walk a run directory and collect NL utterances from all dialog JSONs."""
    texts: list[str] = []
    for json_path in sorted(run_dir.rglob("dialog_*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        turns = data.get("generated_dialog", [])
        for turn in turns:
            if _is_nl_turn(turn):
                texts.append(_extract_nl(turn))
    return texts


class DialogPerplexityEvaluator:
    """Compute perplexity of generated NL utterances using a local causal LM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        batch_size: int = 8,
        max_length: int = 256,
        device: str | None = None,
        hf_cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.hf_cache_dir = hf_cache_dir

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"[ppl] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[ppl] Loading model: {model_name}  (device={self.device})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

    def _batch_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[float, int]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[:, :-1, :].float().contiguous()
        labels = input_ids[:, 1:].contiguous()

        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        ).view(labels.size(0), labels.size(1))

        # Only count positions where both current and next token are real
        valid = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
        valid = valid.to(dtype=per_token_loss.dtype)
        total_loss = (per_token_loss * valid).sum().item()
        total_tokens = int(valid.sum().item())
        return total_loss, total_tokens

    def _score_texts(self, texts: list[str]) -> dict[str, Any]:
        total_loss = 0.0
        total_tokens = 0
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self._tokenize_batch(batch)
            bl, bt = self._batch_loss(enc["input_ids"], enc["attention_mask"])
            total_loss += bl
            total_tokens += bt

        if total_tokens == 0:
            return {"perplexity": None, "avg_loss": None, "total_tokens": 0, "total_utterances": len(texts)}

        avg_loss = total_loss / total_tokens
        return {
            "perplexity": round(math.exp(avg_loss), 4),
            "avg_loss": round(avg_loss, 6),
            "total_tokens": total_tokens,
            "total_utterances": len(texts),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_texts(self, texts: list[str]) -> dict[str, Any]:
        """Score an explicit list of NL strings."""
        return self._score_texts(texts)

    def evaluate_run(self, run_dir: str | Path) -> dict[str, Any]:
        """Walk run_dir for dialog JSONs, extract NL turns, return ppl stats."""
        run_dir = Path(run_dir)
        texts = _collect_nl_texts_from_run(run_dir)
        print(f"[ppl] {run_dir.name}: {len(texts)} NL utterances collected")
        result = self._score_texts(texts)
        result["model_name"] = self.model_name
        return result
