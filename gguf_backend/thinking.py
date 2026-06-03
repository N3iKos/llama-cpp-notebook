from __future__ import annotations

import json
import re
from typing import Any


REASONING_HEAVY_MARKERS = {
    "deepseek-r1": "DeepSeek-R1",
    "qwq": "QwQ",
    "phi-4-reasoning": "Phi-4 reasoning",
    "magistral": "Magistral",
}

SUPPORTED_FAMILIES = {
    "auto",
    "gemma4",
    "qwen3",
    "deepseek-v31",
    "glm",
    "hermes4",
    "gpt-oss",
    "reasoning-heavy",
    "none",
}


def _normalize_text(*parts: str) -> str:
    text = " ".join(str(p or "") for p in parts).lower()
    return text.replace("_", "-")


def _is_filled(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _mode(value: Any, default: str = "auto") -> str:
    value = str(value if value is not None else default).strip().lower()
    return value or default


def _clean_format(value: Any) -> str:
    value = str(value or "").strip()
    if value.lower() == "none":
        return "none"
    return value


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _parse_kwargs(value: Any, warnings: list[str], label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        warnings.append(f"{label} chat_template_kwargs is not valid JSON; it was not merged.")
        return {}
    if not isinstance(parsed, dict):
        warnings.append(f"{label} chat_template_kwargs must be a JSON object; it was not merged.")
        return {}
    return parsed


def detect_thinking_family(model_path: str, alias: str = "") -> str:
    """Detect the thinking-control family from a GGUF path/name and alias."""
    name = _normalize_text(model_path, alias)

    if re.search(r"gemma-?4", name):
        return "gemma4"
    if re.search(r"qwen-?3(?:\.\d+)?", name):
        return "qwen3"
    if re.search(r"deepseek-v?3\.1|deepseek-v31", name):
        return "deepseek-v31"
    if re.search(r"glm-?4|glm-?5|z\.ai|\bzai\b", name):
        return "glm"
    if re.search(r"hermes-?4", name):
        return "hermes4"
    if re.search(r"gpt-?oss", name):
        return "gpt-oss"
    if re.search(r"deepseek-r1|qwq|phi-?4-?reasoning|magistral", name):
        return "reasoning-heavy"
    return "none"


# Backward-compatible name used by older code/tests.
detect_family = detect_thinking_family


def merge_chat_template_kwargs(
    mapper_kwargs: dict[str, Any] | None,
    manual_kwargs: dict[str, Any] | str | None = None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Merge template kwargs with manual values taking precedence."""
    warnings = warnings if warnings is not None else []
    merged = dict(mapper_kwargs or {})
    manual = _parse_kwargs(manual_kwargs, warnings, "manual")
    merged.update(manual)
    return merged


def build_thinking_plan(thinking_config: dict | None, model_path: str, alias: str = "") -> dict[str, Any]:
    """Build a model-aware thinking plan without checking binary flag support."""
    cfg = dict(thinking_config or {})
    requested_family = str(cfg.get("family", "auto") or "auto").strip().lower()
    detected_family = detect_thinking_family(model_path, alias)
    family = detected_family if requested_family in {"", "auto"} else requested_family
    if family not in SUPPORTED_FAMILIES:
        family = "none" if requested_family == "off" else requested_family

    mode = _mode(cfg.get("mode", "auto"), "auto")
    budget = str(cfg.get("budget", "") or "").strip()
    fmt = _clean_format(cfg.get("format", ""))
    soft_prompt = str(cfg.get("soft_prompt", "") or "").strip()

    plan: dict[str, Any] = {
        "requested_family": requested_family,
        "detected_family": detected_family,
        "family": family,
        "mode": mode,
        "effective_mode": mode,
        "budget": budget,
        "format": fmt,
        "soft_prompt": soft_prompt,
        "jinja": False,
        "chat_template_kwargs": {},
        "reasoning": None,
        "reasoning_budget": None,
        "reasoning_format": None,
        "warnings": [],
    }

    def set_reasoning(value: str | None, budget_value: str | None = None, fmt_value: str | None = None):
        plan["reasoning"] = value
        if budget_value is not None and str(budget_value).strip() != "":
            plan["reasoning_budget"] = str(budget_value).strip()
        if fmt_value is not None and str(fmt_value).strip() != "":
            plan["reasoning_format"] = str(fmt_value).strip()

    if family == "none":
        plan["effective_mode"] = "disabled"
        return plan

    if family in {"gemma4", "qwen3"}:
        plan["jinja"] = True
        if mode == "on":
            plan["chat_template_kwargs"] = {"enable_thinking": True}
            set_reasoning("on", budget or ("-1" if family == "gemma4" else None), fmt or None)
        elif mode == "off":
            plan["chat_template_kwargs"] = {"enable_thinking": False}
            set_reasoning("off", budget or ("0" if family == "gemma4" else None), fmt or None)
        elif mode == "auto":
            set_reasoning("auto", budget or None, fmt or None)
        else:
            plan["warnings"].append(f"Unsupported {family} thinking mode '{mode}'; using auto.")
            plan["effective_mode"] = "auto"
            set_reasoning("auto", budget or None, fmt or None)
        return plan

    if family == "deepseek-v31":
        plan["warnings"].append("DeepSeek-V3.1 thinking/non-thinking depends on the GGUF chat template.")
        if mode == "on":
            set_reasoning("on", budget or None, fmt or "deepseek")
        elif mode == "off":
            set_reasoning("off", budget or "0", fmt or None)
        elif mode == "auto":
            set_reasoning("auto", budget or None, fmt or "deepseek")
        else:
            plan["warnings"].append(f"Unsupported DeepSeek-V3.1 mode '{mode}'; using auto.")
            plan["effective_mode"] = "auto"
            set_reasoning("auto", budget or None, fmt or "deepseek")
        return plan

    if family == "glm":
        plan["warnings"].append("GLM thinking control is runtime/template-dependent for local GGUF builds.")
        if mode in {"on", "off"}:
            set_reasoning(mode, budget or None, fmt or None)
        elif mode == "auto":
            set_reasoning("auto", budget or None, fmt or None)
        else:
            plan["warnings"].append(f"Unsupported GLM mode '{mode}'; using auto.")
            plan["effective_mode"] = "auto"
            set_reasoning("auto", budget or None, fmt or None)
        return plan

    if family == "hermes4":
        plan["warnings"].append("Hermes 4 thinking behavior depends on the GGUF chat template exposing a thinking variable.")
        if mode == "on":
            plan["chat_template_kwargs"] = {"thinking": True}
            set_reasoning("on", budget or None, fmt or None)
        elif mode == "off":
            plan["chat_template_kwargs"] = {"thinking": False}
            set_reasoning("off", budget or None, fmt or None)
        elif mode == "auto":
            set_reasoning("auto", budget or None, fmt or None)
        else:
            plan["warnings"].append(f"Unsupported Hermes 4 mode '{mode}'; using auto.")
            plan["effective_mode"] = "auto"
            set_reasoning("auto", budget or None, fmt or None)
        return plan

    if family == "gpt-oss":
        if mode == "off":
            plan["effective_mode"] = "low"
            plan["chat_template_kwargs"] = {"reasoning_effort": "low"}
            plan["warnings"].append("gpt-oss has no true off switch; using low reasoning effort.")
        elif mode in {"low", "medium", "high"}:
            plan["effective_mode"] = mode
            plan["chat_template_kwargs"] = {"reasoning_effort": mode}
        elif mode in {"", "auto"}:
            plan["effective_mode"] = "default"
        else:
            plan["effective_mode"] = "default"
            plan["warnings"].append(f"Unsupported gpt-oss mode '{mode}'; leaving reasoning effort at template default.")
        return plan

    if family == "reasoning-heavy":
        plan["warnings"].append("This family is reasoning-heavy/reasoning-only; off mode is not reliable.")
        if mode in {"on", "off", "auto"}:
            set_reasoning(mode, budget or ("0" if mode == "off" else None), fmt or None)
        else:
            plan["warnings"].append(f"Unsupported reasoning-heavy mode '{mode}'; using auto.")
            plan["effective_mode"] = "auto"
            set_reasoning("auto", budget or None, fmt or None)
        return plan

    plan["warnings"].append("Unknown model family; applying only generic llama.cpp reasoning fields when available.")
    if mode in {"on", "off", "auto"}:
        set_reasoning(mode, budget or None, fmt or None)
    return plan


def _has_flag(help_text: str, flag: str) -> bool:
    return flag in (help_text or "")


def _apply_manual_overrides(plan: dict[str, Any], *, manual_reasoning=None, manual_reasoning_budget=None, manual_reasoning_format=None):
    if _is_filled(manual_reasoning):
        plan["reasoning"] = str(manual_reasoning).strip()
    if _is_filled(manual_reasoning_budget):
        plan["reasoning_budget"] = str(manual_reasoning_budget).strip()
    if _is_filled(manual_reasoning_format):
        plan["reasoning_format"] = str(manual_reasoning_format).strip()


def build_thinking_args(
    config: dict | None,
    *,
    help_text: str,
    model_path: str = "",
    alias: str = "",
    existing_chat_template_kwargs: dict[str, Any] | str | None = None,
    manual_reasoning: str | None = None,
    manual_reasoning_budget: str | None = None,
    manual_reasoning_format: str | None = None,
    manual_jinja: bool | None = None,
):
    """Return (args, summary_lines, warnings) for llama-server thinking config."""
    plan = build_thinking_plan(config, model_path, alias)
    warnings = list(plan.get("warnings") or [])
    merged_kwargs = merge_chat_template_kwargs(
        plan.get("chat_template_kwargs") or {},
        existing_chat_template_kwargs,
        warnings=warnings,
    )
    plan["chat_template_kwargs"] = merged_kwargs
    _apply_manual_overrides(
        plan,
        manual_reasoning=manual_reasoning,
        manual_reasoning_budget=manual_reasoning_budget,
        manual_reasoning_format=manual_reasoning_format,
    )
    if manual_jinja is True:
        plan["jinja"] = True

    args: list[str] = []
    supported = {
        "jinja": _has_flag(help_text, "--jinja"),
        "chat_template_kwargs": _has_flag(help_text, "--chat-template-kwargs"),
        "reasoning": _has_flag(help_text, "--reasoning"),
        "reasoning_budget": _has_flag(help_text, "--reasoning-budget"),
        "reasoning_format": _has_flag(help_text, "--reasoning-format"),
    }

    if plan.get("jinja"):
        if supported["jinja"]:
            args.append("--jinja")
        else:
            warnings.append("llama-server does not expose --jinja; Jinja template controls were skipped.")

    kwargs_json = ""
    if merged_kwargs:
        kwargs_json = _json_dumps(merged_kwargs)
        if supported["chat_template_kwargs"]:
            args += ["--chat-template-kwargs", kwargs_json]
        else:
            warnings.append("llama-server does not expose --chat-template-kwargs; template kwargs were skipped.")

    reasoning = plan.get("reasoning")
    if _is_filled(reasoning):
        if supported["reasoning"]:
            args += ["--reasoning", str(reasoning).strip()]
        else:
            warnings.append("llama-server does not expose --reasoning; reasoning mode was skipped.")

    budget = plan.get("reasoning_budget")
    if _is_filled(budget):
        if supported["reasoning_budget"]:
            args += ["--reasoning-budget", str(budget).strip()]
        else:
            warnings.append("llama-server does not expose --reasoning-budget; reasoning budget was skipped.")

    fmt = plan.get("reasoning_format")
    if _is_filled(fmt):
        if supported["reasoning_format"]:
            args += ["--reasoning-format", str(fmt).strip()]
        else:
            warnings.append("llama-server does not expose --reasoning-format; reasoning format was skipped.")

    summary = summarize_thinking_plan(plan, chat_template_kwargs_json=kwargs_json, warnings=warnings)
    return args, summary, warnings


def summarize_thinking_plan(plan: dict[str, Any], *, chat_template_kwargs_json: str = "", warnings: list[str] | None = None) -> list[str]:
    warnings = warnings if warnings is not None else list(plan.get("warnings") or [])
    kwargs_json = chat_template_kwargs_json
    if not kwargs_json and plan.get("chat_template_kwargs"):
        kwargs_json = _json_dumps(plan["chat_template_kwargs"])
    lines = [
        f"family: {plan.get('family', '-')}",
        f"detected family: {plan.get('detected_family', '-')}",
        f"mode: {plan.get('effective_mode') or plan.get('mode') or '-'}",
        f"chat_template_kwargs: {kwargs_json or '-'}",
        f"reasoning: {plan.get('reasoning') or '-'}",
        f"reasoning_budget: {plan.get('reasoning_budget') or '-'}",
        f"reasoning_format: {plan.get('reasoning_format') or '-'}",
    ]
    if plan.get("soft_prompt"):
        lines.append(f"soft_prompt: {plan['soft_prompt']}")
    if warnings:
        lines.append("warnings: " + " | ".join(dict.fromkeys(warnings)))
    else:
        lines.append("warnings: -")
    return lines


def summarize_thinking_config(config, *, model_path: str = "", alias: str = ""):
    plan = build_thinking_plan(config, model_path, alias)
    return summarize_thinking_plan(plan)
