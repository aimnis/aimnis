"""Model registry — maps an upstream model id to the compliance flags that must
travel with anything derived from it into the pool.

These flags encode upstream-ToS obligations for each model's outputs:
  - output_trainable     — may the distilled answer enter the training feed?
  - attribution_required — must served answers carry an attribution line?
  - no_grounded_cache    — must this never be cached at all? (e.g. grounded search)

Matching is by substring on the model id (OpenRouter ids look like
`meta-llama/llama-3.3-70b-instruct:free`). Unknown models get a CONSERVATIVE
default: cacheable, but not fed to training and no attribution asserted.
"""

from __future__ import annotations

# Conservative default for anything not matched below.
_DEFAULT = {"output_trainable": False, "attribution_required": False, "no_grounded_cache": False}


def compliance_for(model: str | None) -> dict:
    m = (model or "").lower()
    if "llama" in m:
        # Llama community license: redistribution/derivatives OK WITH attribution.
        return {"output_trainable": True, "attribution_required": True, "no_grounded_cache": False}
    if "gemini" in m or "gemma" in m or m.startswith("google/"):
        # Keep Google-model output out of the training feed; plain (non-grounded)
        # chat completions are still cacheable.
        return {"output_trainable": False, "attribution_required": False, "no_grounded_cache": False}
    if "deepseek" in m or "qwen" in m or "mistral" in m or "gpt-oss" in m:
        # Permissive (MIT / Apache-2.0) — trainable, no attribution asserted.
        return {"output_trainable": True, "attribution_required": False, "no_grounded_cache": False}
    return dict(_DEFAULT)


def attribution_for(model: str | None) -> str | None:
    """The attribution line a served answer must carry, or None."""
    if compliance_for(model)["attribution_required"] and "llama" in (model or "").lower():
        return "Built with Llama"
    return None
