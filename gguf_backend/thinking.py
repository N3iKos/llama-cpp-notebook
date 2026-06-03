from __future__ import annotations

import json
import re
from pathlib import Path


def has_flag(help_text: str, flag: str) -> bool:
    return flag in (help_text or "")


def _truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in {"1", "true", "yes", "y", "on", "enable", "enabled"}


def _clean_mode(mode):
    return str(mode or "auto").strip().lower()


def detect_family(model_path: str = "", alias: str = "") -> str:
    name = (str(model_path or "") + " " + str(alias or "")).lower()
    name = name.replace("_", "-")
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    if "qwen3" in name or "qwen-3" in name:
        return "qwen3"
    if "deepseek-v3.1" in name or "deepseek-v31" in name or "deepseek-3.1" in name:
        return "deepseek-v31"
    if "deepseek-r1" in name or "deepseek-r-1" in name:
        return "deepseek-r1"
    if "glm-4" in name or "glm4" in name or "glm-5" in name or "glm5" in name:
        return "glm"
    if "hermes-4" in name or "hermes4" in name:
        return "hermes4"
    if "gpt-oss" in name or "gptoss" in name:
        return "gpt-oss"
    if "qwq" in name:
        return "qwq"
    if "phi-4-reasoning" in name or "phi4-reasoning" in name:
        return "phi-reasoning"
    if "magistral" in name:
        return "magistral"
    return "unknown"


def _merge_kwargs(existing: str | None, updates: dict) -> str:
    data = {}
    if existing:
        try:
            data = json.loads(existing)
        except Exception:
            # If user passes a non-JSON template kwargs string, do not corrupt it.
            # Prefer the model-aware kwargs instead of concatenating invalid JSON.
            data = {}
    data.update(updates)
    return json.dumps(data, separators=(",", ":"))


def _add_reasoning_common(args, help_text, mode, budget="", fmt=""):
    if mode in {"on", "off", "auto"} and has_flag(help_text, "--reasoning"):
        args += ["--reasoning", mode]
    if str(budget).strip() != "" and has_flag(help_text, "--reasoning-budget"):
        args += ["--reasoning-budget", str(budget)]
    if str(fmt).strip() != "" and has_flag(help_text, "--reasoning-format"):
        args += ["--reasoning-format", str(fmt)]


def build_thinking_args(config, *, help_text: str, model_path: str = "", alias: str = "", existing_chat_template_kwargs=None):
    """Return (args, summary_lines, warnings) for model-aware thinking config.

    config fields:
      family: auto/gemma4/qwen3/deepseek-v31/glm/hermes4/gpt-oss/none
      mode: on/off/auto or low/medium/high for gpt-oss
      budget: -1/0/N or empty
      format: none/deepseek/deepseek-legacy or empty
      soft_prompt: optional user-facing hint, recorded in summary only
    """
    config = dict(config or {})
    requested_family = str(config.get("family", "auto") or "auto").strip().lower()
    detected = detect_family(model_path, alias)
    family = detected if requested_family in {"", "auto"} else requested_family
    mode = _clean_mode(config.get("mode", "auto"))
    budget = str(config.get("budget", "") or "").strip()
    fmt = str(config.get("format", "") or "").strip()
    soft_prompt = str(config.get("soft_prompt", "") or "").strip()

    args = []
    warnings = []
    kwargs = None

    def add_jinja():
        if has_flag(help_text, "--jinja") and "--jinja" not in args:
            args.append("--jinja")

    def add_kwargs(data):
        nonlocal kwargs
        kwargs = _merge_kwargs(existing_chat_template_kwargs, data)
        if has_flag(help_text, "--chat-template-kwargs"):
            args.extend(["--chat-template-kwargs", kwargs])
        else:
            warnings.append("llama-server does not expose --chat-template-kwargs; template thinking may not change.")

    if family in {"none", "off"}:
        summary = ["thinking family: none", "thinking mode: disabled by config"]
        return args, summary, warnings

    if family in {"gemma4", "qwen3"}:
        add_jinja()
        if mode == "on":
            add_kwargs({"enable_thinking": True})
            _add_reasoning_common(args, help_text, "on", budget or "-1", fmt)
        elif mode == "off":
            add_kwargs({"enable_thinking": False})
            _add_reasoning_common(args, help_text, "off", budget or "0", fmt)
        else:
            _add_reasoning_common(args, help_text, "auto", budget, fmt)
        summary = [
            f"thinking family: {family}",
            f"thinking mode: {mode}",
            f"detected family: {detected}",
            f"chat_template_kwargs: {kwargs or '-'}",
            f"reasoning budget: {budget or ('-1' if mode == 'on' else '0' if mode == 'off' else '-')}",
        ]

    elif family == "deepseek-v31":
        add_jinja()
        if mode == "on":
            _add_reasoning_common(args, help_text, "on", budget or "-1", fmt or "deepseek")
        elif mode == "off":
            _add_reasoning_common(args, help_text, "off", budget or "0", fmt or "deepseek")
            warnings.append("DeepSeek-V3.1 non-thinking is template-dependent; confirm the GGUF template supports it.")
        else:
            _add_reasoning_common(args, help_text, "auto", budget, fmt or "deepseek")
        summary = [f"thinking family: {family}", f"thinking mode: {mode}", f"detected family: {detected}", "template: DeepSeek-V3.1 hybrid behavior depends on GGUF template"]

    elif family == "glm":
        add_jinja()
        if mode in {"on", "off"}:
            add_kwargs({"thinking": {"type": "enabled" if mode == "on" else "disabled"}})
            _add_reasoning_common(args, help_text, "on" if mode == "on" else "off", budget or ("-1" if mode == "on" else "0"), fmt)
        else:
            _add_reasoning_common(args, help_text, "auto", budget, fmt)
        warnings.append("GLM thinking control is runtime/template-dependent in local GGUF.")
        summary = [f"thinking family: {family}", f"thinking mode: {mode}", f"detected family: {detected}", f"chat_template_kwargs: {kwargs or '-'}"]

    elif family == "hermes4":
        add_jinja()
        if mode == "on":
            add_kwargs({"thinking": True})
            _add_reasoning_common(args, help_text, "on", budget or "-1", fmt)
        elif mode == "off":
            add_kwargs({"thinking": False})
            _add_reasoning_common(args, help_text, "off", budget or "0", fmt)
        else:
            _add_reasoning_common(args, help_text, "auto", budget, fmt)
        warnings.append("Hermes 4 thinking depends on the GGUF chat template exposing a thinking variable.")
        summary = [f"thinking family: {family}", f"thinking mode: {mode}", f"detected family: {detected}", f"chat_template_kwargs: {kwargs or '-'}"]

    elif family == "gpt-oss":
        effort = mode if mode in {"low", "medium", "high"} else "low" if mode == "off" else "medium"
        if mode == "off":
            warnings.append("gpt-oss has no reliable off switch; using low reasoning effort.")
        if has_flag(help_text, "--reasoning-effort"):
            args += ["--reasoning-effort", effort]
        else:
            warnings.append("llama-server build does not expose --reasoning-effort; use prompt/system controls if needed.")
        summary = [f"thinking family: {family}", f"reasoning effort: {effort}", f"detected family: {detected}"]

    elif family in {"deepseek-r1", "qwq", "phi-reasoning", "magistral"}:
        warnings.append(f"{family} is treated as reasoning-heavy/reasoning-only; off toggle is not reliable.")
        if mode == "off":
            _add_reasoning_common(args, help_text, "off", budget or "0", fmt)
        elif mode == "on":
            _add_reasoning_common(args, help_text, "on", budget or "-1", fmt)
        else:
            _add_reasoning_common(args, help_text, "auto", budget, fmt)
        summary = [f"thinking family: {family}", f"thinking mode: {mode}", f"detected family: {detected}", "warning: no reliable hard on/off switch"]

    else:
        warnings.append("Unknown model family; applying generic llama.cpp reasoning flags only.")
        if mode in {"on", "off", "auto"}:
            _add_reasoning_common(args, help_text, mode, budget, fmt)
        summary = [f"thinking family: {family}", f"thinking mode: {mode}", f"detected family: {detected}", "mapper: generic"]

    if soft_prompt:
        summary.append(f"soft prompt hint: {soft_prompt}")
    if args:
        summary.append("thinking args: " + " ".join(args))
    else:
        summary.append("thinking args: -")
    if warnings:
        summary.extend(["warning: " + w for w in warnings])
    return args, summary, warnings


def summarize_thinking_config(config, *, model_path: str = "", alias: str = ""):
    cfg = dict(config or {})
    requested = str(cfg.get("family", "auto") or "auto").strip().lower()
    detected = detect_family(model_path, alias)
    family = detected if requested in {"", "auto"} else requested
    mode = _clean_mode(cfg.get("mode", "auto"))
    return [
        f"thinking family: {family}",
        f"thinking mode: {mode}",
        f"detected family: {detected}",
    ]
