"""Advisor plugin — persistent second-model review for each agent turn.

Registers:
  - ``post_llm_call`` hook — fires at end of each turn, triggers review
  - ``on_session_finalize`` hook — bounded background-review shutdown grace
  - ``/advisor`` slash command — on/off/status
"""

import logging

from .runtime import AdvisorRuntime

logger = logging.getLogger(__name__)

_runtime: AdvisorRuntime | None = None


CONSULT_ADVISOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "consult_advisor",
        "description": (
            "Consult your senior coworker / Staff Architect (Claude Opus 5.5 / GPT-5.6 Sol) "
            "before starting substantive work, making architectural choices, or writing non-trivial code. "
            "Pass your goal and proposed approach to receive strategic guidance, cleaner patterns, "
            "and edge cases to watch for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The goal, feature, or bug you are working on.",
                },
                "proposed_approach": {
                    "type": "string",
                    "description": "Your initial plan or the architectural choices you are considering.",
                },
            },
            "required": ["task"],
        },
    },
}

VERIFY_TURN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verify_turn",
        "description": (
            "Verify your actions, findings, and proposed answer with the auditor before presenting your final reply to the user. "
            "Pass your draft answer or summary of actions. The auditor checks your work against the full session history, "
            "user constraints, and actual tool outputs before approving."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "draft": {
                    "type": "string",
                    "description": "Your proposed final response or summary of actions taken.",
                },
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional specific factual assertions to audit.",
                },
            },
        },
    },
}


def _handle_consult_advisor(args: dict | None = None, **kwargs) -> str:
    """Tool handler: runs on-demand advisor consultation."""
    runtime = _get_runtime()
    if runtime is None:
        return "Advisor plugin not initialized."
    args = args or {}
    task = args.get("task", "")
    proposed_approach = args.get("proposed_approach", "")
    session_id = kwargs.get("session_id", "")
    return runtime.consult_advisor(task=task, proposed_approach=proposed_approach, session_id=session_id)


def _handle_verify_turn(args: dict | None = None, **kwargs) -> str:
    """Tool handler: audits turn claims and draft against recorded tool outputs and session history."""
    import json
    runtime = _get_runtime()
    if runtime is None:
        return json.dumps({"status": "error", "reason": "Advisor plugin not initialized."})
    args = args or {}
    draft = args.get("draft", "")
    claims = args.get("claims", [])
    session_id = kwargs.get("session_id", "")
    res = runtime.verify_turn(draft=draft, claims=claims, session_id=session_id)
    return json.dumps(res) if isinstance(res, dict) else str(res)


def register(ctx):
    global _runtime

    _runtime = AdvisorRuntime(ctx)

    # Register the consult_advisor tool for in-turn architectural consultation
    ctx.register_tool(
        name="consult_advisor",
        toolset="advisor",
        schema=CONSULT_ADVISOR_SCHEMA,
        handler=_handle_consult_advisor,
        description="Consult senior coworker/architect for strategic design advice and cleaner patterns.",
        emoji="🧠",
    )

    # Register verify_turn tool for in-turn factual and evidence auditing
    ctx.register_tool(
        name="verify_turn",
        toolset="advisor",
        schema=VERIFY_TURN_SCHEMA,
        handler=_handle_verify_turn,
        description="Verify factual claims against tool outputs before answering.",
        emoji="✅",
    )

    # Register hooks
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_verify", _on_pre_verify)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("on_session_finalize", _on_session_finalize)

    # Register the /advisor slash command
    ctx.register_command(
        name="advisor",
        handler=_handle_advisor,
        description="Toggle/inspect advisor. Usage: /advisor [on|off|status]",
    )

    # Contribute the advisor badge to the CLI status bar, so "is it on, and
    # which provider/model is reviewing" is visible without running
    # /advisor status. Renders after the YOLO badge; ``field="advisor"`` lets
    # a user keep it while hiding other fields via display.status_bar.fields.
    #
    # Optional seam: ``register_status_segment`` is newer than the hooks this
    # plugin needs, so a build without it must still load — the badge is a
    # convenience, not a capability. (Without the guard, register() would raise
    # AttributeError and disable the whole plugin on that build.)
    register_segment = getattr(ctx, "register_status_segment", None)
    if callable(register_segment):
        register_segment("model", _status_segment, priority=100, field="advisor")

    logger.info("Advisor plugin registered (state=%s)",
                "enabled" if _runtime.state.enabled else "disabled")


def _on_pre_llm_call(**kwargs) -> dict:
    """Hook: injects verification directive into active turn context."""
    runtime = _get_runtime()
    if runtime is None:
        return {}
    return runtime.on_pre_llm_call(**kwargs)


def _on_post_tool_call(**kwargs):
    """Hook: buffers tool execution events for in-turn verification."""
    runtime = _get_runtime()
    if runtime is not None:
        runtime.record_tool_call(**kwargs)


def _on_pre_verify(**kwargs):
    """Hook: in-loop verification gate after file edits before completion."""
    runtime = _get_runtime()
    if runtime is None:
        return None
    return runtime.on_pre_verify(**kwargs)


def _on_transform_llm_output(**kwargs) -> str | None:
    """Hook: enforces verification on final response before delivery."""
    runtime = _get_runtime()
    if runtime is None:
        return None
    return runtime.on_transform_llm_output(**kwargs)


def _on_session_finalize(**kwargs):
    """Bound shutdown latency while allowing a fast review to finish."""
    runtime = _get_runtime()
    if runtime is not None:
        runtime.on_session_finalize(**kwargs)


def _handle_advisor(args: str) -> str | None:
    """Slash command handler for /advisor [on|off|status]."""
    runtime = _get_runtime()
    if runtime is None:
        return "Advisor plugin not initialized."
    return runtime.handle_command(args)


def _get_runtime() -> AdvisorRuntime | None:
    return _runtime


def _status_segment():
    """Status-bar badge: ``⚖ provider/model`` when on, ``⚖ off`` when not.

    Answers the three questions a glance at the footer should answer — is the
    advisor configured (the badge exists at all), is it on, and which
    provider/model is doing the review. Reads only in-memory state, so it is
    safe on a bar that repaints per keystroke.

    Returns a plain string, which ``ctx.register_status_segment`` accepts and
    renders in the host's default segment style. That keeps the plugin on
    documented plugin APIs only (catalog rule 9): importing ``StatusSegment`` /
    ``STYLE_DIM`` from ``hermes_cli.status_segments`` would bind the plugin to a
    core module that carries no compatibility promise.
    """
    runtime = _get_runtime()
    if runtime is None:
        return None
    state = runtime.state
    if not state.enabled:
        return "⚖ off"

    model = (state.model or "").strip()
    provider = (state.provider or "").strip()
    if model and provider:
        return f"⚖ {provider}/{model}"
    if model:
        return f"⚖ {model}"
    return "⚖ on"
