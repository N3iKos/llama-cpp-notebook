"""Model-aware reasoning/thinking toggle mapper for llama.cpp.

This module translates a user-facing ``THINKING_CONFIG`` dict into the correct
``llama-server`` CLI flags, ``--chat-template-kwargs`` JSON, and ``--jinja``
toggles for each supported model family.

Public API
----------
detect_family(model_path)
    Return the family id string from a model file path.

resolve_thinking(thinking_config, model_path)
    Return a ``ThinkingResult`` with all resolved flags, kwargs, and warnings.

apply_thinking(cfg, help_text, cmd)
    Mutate *cmd* in-place to add thinking-related flags, respecting both
    ``--help`` availability and manual user overrides.  Returns the
    ``ThinkingResult`` for dashboard display.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ThinkingResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThinkingResult:
    """Output of the thinking mapper.

    Attributes
    ----------
    detected_family : str
        Resolved model family id (e.g. ``"gemma4"``, ``"qwen3"``).
    requested_family : str
        What the user wrote in ``THINKING_CONFIG["family"]``.
    mode : str
        Resolved mode after family-specific normalisation.
    extra_args : list[str]
        Raw CLI flag pairs to *potentially* append.  Each even index is a flag
        name (``"--reasoning"``), each odd index is its value (``"off"``).
        The caller must check ``--help`` before actually appending.
    chat_template_kwargs : str
        JSON string for ``--chat-template-kwargs``, or ``""`` if not needed.
    jinja : bool
        Whether ``--jinja`` should be enabled for this family.
    warnings : list[str]
        Human-readable warnings for the dashboard.
    summary : dict[str, str]
        Key→value lines for the dashboard thinking summary block.
    """

    detected_family: str = "unknown"
    requested_family: str = "auto"
    mode: str = "off"
    extra_args: list[str] = field(default_factory=list)
    chat_template_kwargs: str = ""
    jinja: bool = False
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, family_id).
# Order matters — first match wins.  Patterns operate on the lowercased,
# underscore-to-dash normalised basename of the model file.
_FAMILY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Reasoning-heavy / reasoning-only (must come before generic matches)
    (re.compile(r"deepseek-r1"),         "reasoning-heavy"),
    (re.compile(r"\bqwq\b"),             "reasoning-heavy"),
    (re.compile(r"phi-reasoning"),       "reasoning-heavy"),
    (re.compile(r"\bmagistral\b"),       "reasoning-heavy"),

    # Specific families
    (re.compile(r"gemma-?4"),            "gemma4"),
    (re.compile(r"qwen3(?:\.[0-9]+)?"),  "qwen3"),
    (re.compile(r"deepseek-v3\.?1"),     "deepseek-v31"),
    (re.compile(r"deepseek-v31"),        "deepseek-v31"),
    (re.compile(r"glm-?4(?:\.[0-9]+)?"), "glm"),
    (re.compile(r"\bglm\b"),             "glm"),
    (re.compile(r"hermes-?4"),           "hermes4"),
    (re.compile(r"\bgpt-oss\b"),         "gpt-oss"),
]

# Families that support Jinja-based ``enable_thinking`` kwarg.
_JINJA_THINKING_FAMILIES = {"gemma4", "qwen3"}

# Valid families for the ``family`` config field.
KNOWN_FAMILIES = frozenset({
    "auto", "none",
    "gemma4", "qwen3", "deepseek-v31", "glm", "hermes4", "gpt-oss",
    "reasoning-heavy", "unknown",
})


def detect_family(model_path: str) -> str:
    """Return the family id by matching the model filename against known patterns.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the GGUF model file.

    Returns
    -------
    str
        One of the ``KNOWN_FAMILIES`` strings, or ``"unknown"`` if no pattern
        matched.
    """
    basename = Path(model_path).name.lower().replace("_", "-")
    for pattern, family_id in _FAMILY_PATTERNS:
        if pattern.search(basename):
            return family_id
    return "unknown"


# ---------------------------------------------------------------------------
# Per-family mappers
# ---------------------------------------------------------------------------

def _map_gemma4(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """Gemma 4 thinking mapper.

    Gemma 4 uses Jinja templates with ``enable_thinking`` kwarg.
    Supports ``--reasoning on/off`` and ``--reasoning-budget``.
    """
    result = ThinkingResult(detected_family="gemma4", mode=mode, jinja=True)

    if mode == "on":
        result.chat_template_kwargs = json.dumps({"enable_thinking": True})
        result.extra_args = ["--reasoning", "on"]
        # Budget: user value or default -1 (unlimited)
        effective_budget = budget if budget != "" else "-1"
        result.extra_args += ["--reasoning-budget", str(effective_budget)]
        result.summary = {
            "chat_template_kwargs": result.chat_template_kwargs,
            "reasoning flag": "on",
            "reasoning budget": str(effective_budget),
            "jinja": "enabled",
        }

    elif mode == "off":
        result.chat_template_kwargs = json.dumps({"enable_thinking": False})
        result.extra_args = ["--reasoning", "off"]
        # Budget: user value or default 0
        effective_budget = budget if budget != "" else "0"
        result.extra_args += ["--reasoning-budget", str(effective_budget)]
        result.summary = {
            "chat_template_kwargs": result.chat_template_kwargs,
            "reasoning flag": "off",
            "reasoning budget": str(effective_budget),
            "jinja": "enabled",
        }

    elif mode == "auto":
        # Auto: enable Jinja but don't force enable_thinking.
        # Only add budget/format if user explicitly provided them.
        if budget != "":
            result.extra_args += ["--reasoning-budget", str(budget)]
        if fmt != "":
            result.extra_args += ["--reasoning-format", str(fmt)]
        result.summary = {
            "chat_template_kwargs": "(not forced)",
            "reasoning flag": "(not forced)",
            "reasoning budget": budget if budget != "" else "(default)",
            "jinja": "enabled",
        }

    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


def _map_qwen3(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """Qwen3 thinking mapper.

    Qwen3 uses Jinja templates with ``enable_thinking`` kwarg.
    Supports ``--reasoning on/off``.
    Budget support depends on llama.cpp version.
    """
    result = ThinkingResult(detected_family="qwen3", mode=mode, jinja=True)

    if mode == "on":
        result.chat_template_kwargs = json.dumps({"enable_thinking": True})
        result.extra_args = ["--reasoning", "on"]
        if budget != "":
            result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary = {
            "chat_template_kwargs": result.chat_template_kwargs,
            "reasoning flag": "on",
            "jinja": "enabled",
        }
        if budget != "":
            result.summary["reasoning budget"] = str(budget)

    elif mode == "off":
        result.chat_template_kwargs = json.dumps({"enable_thinking": False})
        result.extra_args = ["--reasoning", "off"]
        if budget != "":
            result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary = {
            "chat_template_kwargs": result.chat_template_kwargs,
            "reasoning flag": "off",
            "jinja": "enabled",
        }

    elif mode == "auto":
        if budget != "":
            result.extra_args += ["--reasoning-budget", str(budget)]
        if fmt != "":
            result.extra_args += ["--reasoning-format", str(fmt)]
        result.summary = {
            "chat_template_kwargs": "(not forced)",
            "reasoning flag": "(not forced)",
            "jinja": "enabled",
        }

    # Qwen3 soft_prompt hint: /think and /no_think are user-level prompt tokens.
    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'
    elif mode == "on":
        result.summary["soft_prompt hint"] = 'user can also put /think in prompt'
    elif mode == "off":
        result.summary["soft_prompt hint"] = 'user can also put /no_think in prompt'

    return result


def _map_deepseek_v31(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """DeepSeek-V3.1 thinking mapper.

    Hybrid thinking/non-thinking by template.  Local GGUF behavior depends
    heavily on chat template support — we do not overpromise.
    """
    result = ThinkingResult(detected_family="deepseek-v31", mode=mode)
    result.warnings.append(
        "DeepSeek-V3.1 local GGUF behavior depends heavily on chat template support. "
        "Thinking toggle may not work reliably without the correct template."
    )

    if mode == "on":
        result.extra_args = ["--reasoning", "on"]
        # Default format to "deepseek" if user didn't specify and mode is on.
        effective_fmt = fmt if fmt != "" else "deepseek"
        result.extra_args += ["--reasoning-format", effective_fmt]
        if budget != "":
            result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary = {
            "reasoning flag": "on",
            "reasoning format": effective_fmt,
        }
        if budget != "":
            result.summary["reasoning budget"] = str(budget)

    elif mode == "off":
        result.extra_args = ["--reasoning", "off"]
        result.summary = {"reasoning flag": "off"}

    elif mode == "auto":
        if fmt != "":
            result.extra_args += ["--reasoning-format", str(fmt)]
        if budget != "":
            result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary = {"reasoning flag": "(not forced)"}

    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


def _map_glm(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """GLM family thinking mapper.

    GLM thinking schema may not be fully controllable via llama.cpp GGUF
    unless template/runtime supports it.
    """
    result = ThinkingResult(detected_family="glm", mode=mode)
    result.warnings.append(
        "GLM thinking schema may not be fully controllable via llama.cpp GGUF "
        "unless template/runtime supports it."
    )

    if mode == "on":
        result.extra_args = ["--reasoning", "on"]
        result.summary = {"reasoning flag": "on"}
    elif mode == "off":
        result.extra_args = ["--reasoning", "off"]
        result.summary = {"reasoning flag": "off"}
    elif mode == "auto":
        result.summary = {"reasoning flag": "(not forced)"}

    if budget != "":
        result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary["reasoning budget"] = str(budget)
    if fmt != "":
        result.extra_args += ["--reasoning-format", str(fmt)]
        result.summary["reasoning format"] = str(fmt)
    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


def _map_hermes4(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """Hermes 4 thinking mapper.

    Hermes thinking depends on its template or deep-thinking prompt.
    """
    result = ThinkingResult(detected_family="hermes4", mode=mode)
    result.warnings.append(
        "Hermes 4 thinking depends on its template or deep-thinking prompt. "
        "Toggle may not work without the correct template."
    )

    if mode == "on":
        result.extra_args = ["--reasoning", "on"]
        result.summary = {"reasoning flag": "on"}
    elif mode == "off":
        result.extra_args = ["--reasoning", "off"]
        result.summary = {"reasoning flag": "off"}
    elif mode == "auto":
        result.summary = {"reasoning flag": "(not forced)"}

    if budget != "":
        result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary["reasoning budget"] = str(budget)
    if fmt != "":
        result.extra_args += ["--reasoning-format", str(fmt)]
        result.summary["reasoning format"] = str(fmt)
    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


def _map_gpt_oss(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """gpt-oss thinking mapper.

    gpt-oss has no true on/off switch — it uses effort levels (low/medium/high).
    This mapper does NOT invent unsupported flags.  The effort level is shown
    in the dashboard summary as informational only.
    """
    result = ThinkingResult(detected_family="gpt-oss")

    # Normalise mode: on→medium, off→low, auto→medium.
    valid_efforts = {"low", "medium", "high"}
    if mode == "off":
        effective = "low"
        result.warnings.append(
            'gpt-oss has no true off switch; using low effort. '
            'Use a non-reasoning instruct model if you need clean direct answers.'
        )
    elif mode == "on":
        effective = "medium"
    elif mode in valid_efforts:
        effective = mode
    elif mode == "auto":
        effective = "medium"
    else:
        effective = "medium"
        result.warnings.append(f'Unknown mode "{mode}" for gpt-oss; defaulting to medium.')

    result.mode = effective

    # Dashboard-only: no CLI flags emitted for effort level.
    result.summary = {
        "reasoning effort": effective,
        "note": "effort level is informational only; no llama-server flag emitted",
    }

    if budget != "":
        result.summary["reasoning budget (user)"] = str(budget)
    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


def _map_reasoning_heavy(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """Reasoning-heavy / reasoning-only models (DeepSeek-R1, QwQ, etc.).

    These models do not have a reliable off switch.
    """
    result = ThinkingResult(detected_family="reasoning-heavy", mode=mode)
    result.warnings.append(
        "This family is reasoning-heavy/reasoning-only. A reliable off switch is not known. "
        "Use a non-reasoning instruct model if you need clean direct answers."
    )

    # Still pass through any flags the user explicitly wants, but warn.
    if mode == "on":
        result.extra_args = ["--reasoning", "on"]
        result.summary = {"reasoning flag": "on"}
    elif mode == "off":
        result.extra_args = ["--reasoning", "off"]
        result.summary = {"reasoning flag": "off (unreliable for this family)"}
        result.warnings.append(
            'Setting reasoning=off for a reasoning-heavy model. This may not produce '
            'clean non-reasoning output.'
        )
    elif mode == "auto":
        result.summary = {"reasoning flag": "(not forced)"}

    if budget != "":
        result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary["reasoning budget"] = str(budget)
    if fmt != "":
        result.extra_args += ["--reasoning-format", str(fmt)]
        result.summary["reasoning format"] = str(fmt)
    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


def _map_unknown(mode: str, budget: str, fmt: str, soft_prompt: str) -> ThinkingResult:
    """Fallback mapper for unknown model families.

    Passes through raw flags without family-specific logic.
    """
    result = ThinkingResult(detected_family="unknown", mode=mode)

    if mode == "on":
        result.extra_args = ["--reasoning", "on"]
        result.summary = {"reasoning flag": "on"}
    elif mode == "off":
        result.extra_args = ["--reasoning", "off"]
        result.summary = {"reasoning flag": "off"}
    elif mode == "auto":
        result.summary = {"reasoning flag": "(not forced)"}

    if budget != "":
        result.extra_args += ["--reasoning-budget", str(budget)]
        result.summary["reasoning budget"] = str(budget)
    if fmt != "":
        result.extra_args += ["--reasoning-format", str(fmt)]
        result.summary["reasoning format"] = str(fmt)
    if soft_prompt:
        result.summary["soft_prompt hint"] = f'put "{soft_prompt}" in system/user prompt if needed'

    return result


# Family → mapper dispatch table.
_FAMILY_MAPPERS = {
    "gemma4":         _map_gemma4,
    "qwen3":          _map_qwen3,
    "deepseek-v31":   _map_deepseek_v31,
    "glm":            _map_glm,
    "hermes4":        _map_hermes4,
    "gpt-oss":        _map_gpt_oss,
    "reasoning-heavy": _map_reasoning_heavy,
    "unknown":        _map_unknown,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_thinking(thinking_config: dict[str, Any], model_path: str) -> ThinkingResult:
    """Resolve a ``THINKING_CONFIG`` dict into a ``ThinkingResult``.

    Parameters
    ----------
    thinking_config : dict
        User-facing config with keys: ``family``, ``mode``, ``budget``,
        ``format``, ``soft_prompt``.
    model_path : str
        Path to the GGUF model file (used for auto-detection).

    Returns
    -------
    ThinkingResult
    """
    cfg = dict(thinking_config)  # shallow copy to avoid mutating caller's dict
    requested_family = str(cfg.get("family", "auto")).strip().lower()
    mode = str(cfg.get("mode", "off")).strip().lower()
    budget = str(cfg.get("budget", "")).strip()
    fmt = str(cfg.get("format", "")).strip()
    soft_prompt = str(cfg.get("soft_prompt", "")).strip()

    # Resolve family.
    if requested_family in ("auto", ""):
        family = detect_family(model_path)
    elif requested_family == "none":
        # User explicitly disabled thinking mapper.
        result = ThinkingResult(
            detected_family="none",
            requested_family="none",
            mode=mode,
        )
        result.summary = {"thinking mapper": "disabled by user (family=none)"}
        return result
    elif requested_family in KNOWN_FAMILIES:
        family = requested_family
    else:
        family = "unknown"

    # Dispatch to per-family mapper.
    mapper = _FAMILY_MAPPERS.get(family, _map_unknown)
    result = mapper(mode, budget, fmt, soft_prompt)
    result.requested_family = requested_family
    result.detected_family = family

    # Populate top-level summary fields.
    result.summary = {
        "requested family": requested_family,
        "detected family": family,
        "mode": result.mode,
        **result.summary,
    }
    if result.warnings:
        result.summary["warnings"] = "; ".join(result.warnings)
    else:
        result.summary["warnings"] = "none"

    return result


def apply_thinking(cfg: Any, help_text: str, cmd: list[str]) -> ThinkingResult:
    """Apply the thinking mapper to a ServerConfig and command list.

    This function:

    1. Calls ``resolve_thinking`` to get the mapper result.
    2. For each flag pair in ``result.extra_args``, checks:
       - The flag exists in ``help_text`` (``llama-server --help`` output).
       - The user did NOT manually set the corresponding ``ServerConfig`` field.
    3. If ``result.chat_template_kwargs`` is non-empty and the user didn't
       manually set ``cfg.chat_template_kwargs``, applies it.
    4. If ``result.jinja`` is True and the user didn't manually set ``cfg.jinja``,
       adds ``--jinja`` (if the flag exists in help_text).

    Parameters
    ----------
    cfg : ServerConfig
        The server config dataclass instance.  ``cfg.thinking_config`` must be
        a non-None dict.  ``cfg.model_path`` is used for family detection.
    help_text : str
        Output of ``llama-server --help``.
    cmd : list[str]
        The command list being built.  Modified in-place.

    Returns
    -------
    ThinkingResult
        The resolved result (also stored as ``cfg._thinking_result``).
    """
    if not cfg.thinking_config:
        return ThinkingResult()

    result = resolve_thinking(cfg.thinking_config, cfg.model_path)

    # --- Map cfg field names to the CLI flags that extra_args might emit ---
    # This allows us to detect manual overrides.
    _FLAG_TO_FIELD = {
        "--reasoning": "reasoning",
        "--reasoning-budget": "reasoning_budget",
        "--reasoning-format": "reasoning_format",
    }

    # --- Apply extra_args (flag pairs) ---
    i = 0
    while i < len(result.extra_args) - 1:
        flag = result.extra_args[i]
        value = result.extra_args[i + 1]

        # Check if user manually set this field.
        field_name = _FLAG_TO_FIELD.get(flag)
        if field_name:
            manual_value = getattr(cfg, field_name, "")
            if isinstance(manual_value, str) and manual_value.strip() != "":
                # Manual override wins — skip this flag.
                i += 2
                continue

        # Check if flag exists in --help.
        if flag in help_text:
            # Avoid duplicating flags already in cmd.
            if flag not in cmd:
                cmd.extend([flag, value])

        i += 2

    # --- Apply chat_template_kwargs ---
    if result.chat_template_kwargs:
        manual_kwargs = getattr(cfg, "chat_template_kwargs", "")
        if isinstance(manual_kwargs, str) and manual_kwargs.strip() == "":
            if "--chat-template-kwargs" in help_text and "--chat-template-kwargs" not in cmd:
                cmd.extend(["--chat-template-kwargs", result.chat_template_kwargs])

    # --- Apply jinja ---
    if result.jinja:
        manual_jinja = getattr(cfg, "jinja", "")
        if isinstance(manual_jinja, str) and manual_jinja.strip() == "":
            if "--jinja" in help_text and "--jinja" not in cmd:
                cmd.append("--jinja")

    return result
