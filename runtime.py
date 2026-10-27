"""Advisor runtime — turn tracking, state, review triggers."""

import hashlib
import json
import logging
import os
import queue
import threading
from contextvars import copy_context
from pathlib import Path

from .advisor_prompt import ADVISOR_SYSTEM_PROMPT
from .models import AdvisorState, Advice, Severity, TurnDelta

logger = logging.getLogger(__name__)

# Env var to skip live reviews (keeps /advisor test path for manual testing)
ADVISOR_NO_REVIEW = "ADVISOR_NO_REVIEW"
WATCHDOG_FILENAME = "WATCHDOG.md"
REVIEW_TIMEOUT_SECONDS = 90
SHUTDOWN_GRACE_SECONDS = 1.0


def _find_guidelines(cwd: str) -> str:
    """Find WATCHDOG.md, AGENTS.md, or CLAUDE.md by walking up from cwd to git root."""
    try:
        base = Path(cwd).resolve()
        for d in [base, *base.parents]:
            for name in (WATCHDOG_FILENAME, "AGENTS.md", "CLAUDE.md"):
                p = d / name
                if p.is_file():
                    try:
                        return p.read_text(encoding="utf-8", errors="replace").strip()[:4000]
                    except OSError:
                        pass
            if (d / ".git").exists():
                break
    except Exception:
        pass
    return ""


def _is_substantive_turn(user_msg: str, asst_resp: str, history: list) -> bool:
    """True when this turn involved tool calls, code, or substantive technical explanation."""
    last_user = max((i for i, m in enumerate(history) if isinstance(m, dict) and m.get("role") == "user"), default=-1)
    if last_user >= 0:
        recent = history[last_user + 1:]
    else:
        # If history contains no user message, do not assume prior messages belong to this turn
        recent = []
    for msg in recent:
        if isinstance(msg, dict):
            if msg.get("role") == "tool" or msg.get("tool_calls"):
                return True
    text = (asst_resp or "").strip()
    if "```" in text or len(text) >= 200:
        return True
    return False


ADVISOR_INJECTION_PREFIX = "◆ Advisor review"


def _advisor_depth(messages: list) -> int:
    """Count consecutive advisor injections since the last genuine human user message."""
    depth = 0
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").lstrip()
        if content.startswith(ADVISOR_INJECTION_PREFIX):
            depth += 1
        else:
            break
    return depth


MAX_DIRECTIVES_CHARS = 8000
ANCHOR_CAP = 2000
SNIPPET_CAP = 600


def _get_git_diff(cwd: str, paths: list[str], max_chars: int = 12000) -> str | None:
    """Run git diff on changed paths safely, including new and untracked files."""
    try:
        import os
        import subprocess
        from pathlib import Path

        diff_chunks = []
        total_len = 0

        for p in paths:
            if not p or ".git" in str(p):
                continue
            full_path = Path(p) if os.path.isabs(p) else Path(cwd) / p
            if not full_path.exists():
                continue

            rel_p = os.path.relpath(full_path, cwd)

            # Check if tracked by git
            res_track = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", rel_p],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res_track.returncode == 0:
                # Tracked: diff against HEAD (covers staged and unstaged edits in one pass)
                cmd = ["git", "diff", "HEAD", "--", rel_p]
                res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5)
                chunk = res.stdout.strip() if res.returncode == 0 else ""
                if not chunk:
                    # Fallback to working-tree diff if HEAD is not yet committed (fresh repo)
                    cmd_c = ["git", "diff", "--", rel_p]
                    res_c = subprocess.run(cmd_c, cwd=cwd, capture_output=True, text=True, timeout=5)
                    chunk = res_c.stdout.strip() if res_c.returncode == 0 else ""
            else:
                # Untracked/new file: diff against /dev/null (exit 1 is normal for non-empty diff)
                cmd = ["git", "diff", "--no-index", "--", "/dev/null", rel_p]
                res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5)
                chunk = res.stdout.strip()

            if chunk:
                diff_chunks.append(chunk)
                total_len += len(chunk)
                if total_len > max_chars:
                    break

        if not diff_chunks:
            if paths:
                logger.warning("Advisor: changed_paths non-empty (%s) but diff was empty in %s", paths, cwd)
            return None

        full_diff = "\n\n".join(diff_chunks)
        if len(full_diff) > max_chars:
            full_diff = full_diff[:4000] + f"\n\n... [diff truncated - {len(full_diff)} chars total] ...\n\n" + full_diff[-6000:]
        return full_diff
    except Exception as exc:
        logger.debug("Advisor git diff failed for %s in %s: %s", paths, cwd, exc)
        return None


def _clip_directive(text: str, cap: int) -> str:
    """Clip a directive preserving both leading context and trailing instructions."""
    text = text.strip()
    if len(text) <= cap:
        return text
    head = cap * 2 // 5
    tail = cap - head - 20
    return f"{text[:head]} …[{len(text) - cap} chars]… {text[-tail:]}"


def _collect_directives(user_msgs: list[str]) -> list[str]:
    """Collect human user directives with an overall character budget.

    Preserves the initial task framing (first message, up to ANCHOR_CAP)
    and walks newest-to-oldest with balanced head-and-tail clipping so
    trailing instructions in long pastes survive.
    """
    if not user_msgs:
        return []

    cleaned = []
    for m in user_msgs:
        m_str = (m or "").strip()
        if not m_str or m_str in {"...", "…"} or m_str.startswith("◆ Advisor review"):
            continue
        cleaned.append(m_str)

    if not cleaned:
        return []

    first_snippet = _clip_directive(cleaned[0], ANCHOR_CAP)
    rest = cleaned[1:]
    kept_rev = []
    used = len(first_snippet)

    for m in reversed(rest):
        snippet = _clip_directive(m, SNIPPET_CAP)
        if used + len(snippet) > MAX_DIRECTIVES_CHARS:
            break
        kept_rev.append(snippet)
        used += len(snippet)

    return [first_snippet] + list(reversed(kept_rev))


class AdvisorRuntime:
    """Tracks turns and triggers reviews via the advisor model.

    Wired into Hermes plugin hooks:
      - ``post_llm_call`` — fires once per turn at end, carries full history.
        This is our ``turn_end`` equivalent.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        from hermes_constants import get_hermes_home

        self.state_file = get_hermes_home() / "advisor" / "state.json"
        self.sessions_dir = self.state_file.parent / "sessions"
        self.legacy_state_file = Path(__file__).parent / "state.json"
        self._state_lock = threading.RLock()
        self.state = self._load_state()
        self._session_states: dict[str, AdvisorState] = {}
        self._pre_verified_sessions: dict[str, tuple[str, ...]] = {}

        # Current turn tracking — populated from post_llm_call kwargs
        self.last_turn_key: tuple[str, str] | None = None
        self._review_queue: queue.Queue[TurnDelta | None] = queue.Queue(maxsize=1)
        self._queue_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    # ── state persistence ────────────────────────────────────────────────

    def _load_state(self) -> AdvisorState:
        for path in (self.state_file, self.legacy_state_file):
            try:
                state = AdvisorState.deserialize(json.loads(path.read_text()))
            except FileNotFoundError:
                continue
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning("Advisor: could not load state from %s: %s", path, exc)
                continue

            if path == self.legacy_state_file:
                self.state = state
                self._save_state()
            return state
        return AdvisorState(enabled=True)

    def _save_state(self):
        from utils import atomic_json_write

        with self._state_lock:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(
                self.state_file,
                {
                    **self.state.serialize(),
                    "held_notes": [],
                },
                indent=2,
                mode=0o600,
            )

    def _session_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return self.sessions_dir / f"{digest}.json"

    def _session_state(self, session_id: str) -> AdvisorState:
        key = session_id or "default"
        with self._state_lock:
            existing = self._session_states.get(key)
            if existing is not None:
                return existing
            try:
                data = json.loads(self._session_path(key).read_text())
                state = AdvisorState.deserialize(data)
            except FileNotFoundError:
                state = AdvisorState(enabled=True)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning("Advisor: could not load session state: %s", exc)
                state = AdvisorState(enabled=True)
            self._session_states[key] = state
            return state

    def _save_session_state(self, session_id: str, state: AdvisorState) -> None:
        from utils import atomic_json_write

        key = session_id or "default"
        with self._state_lock:
            path = self._session_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(
                path,
                {
                    "session_id": key,
                    "held_notes": state.held_notes,
                },
                indent=2,
                mode=0o600,
            )

    def _held_count(self) -> int:
        count = 0
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                count += len(data.get("held_notes") or [])
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                logger.warning("Advisor: could not count held notes in %s: %s", path, exc)
        return count

    def _clear_held_notes(self) -> None:
        from utils import atomic_json_write

        with self._state_lock:
            # Clear in-memory session states and track which disk files we cover
            cleared_digests: set[str] = set()
            for key, state in self._session_states.items():
                state.held_notes = []
                self._save_session_state(key, state)
                cleared_digests.add(self._session_path(key).name)

            # Catch orphaned session files not yet loaded into memory
            for path in self.sessions_dir.glob("*.json"):
                if path.name in cleared_digests:
                    continue  # already handled above
                try:
                    atomic_json_write(
                        path,
                        {"session_id": "(migrated)", "held_notes": []},
                        indent=2,
                        mode=0o600,
                    )
                except (OSError, TypeError) as exc:
                    logger.warning("Advisor: could not clear %s: %s", path, exc)

    # ── hook: end of each agent turn ─────────────────────────────────────

    def on_post_llm_call(
        self, *,
        session_id: str = "",
        turn_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: list | None = None,
        model: str = "",
        **kwargs,
    ):
        """Fired at end of each turn (tool-calling loop complete).

        This is the Hermes equivalent of pi's ``turn_end`` hook.
        """
        with self._state_lock:
            if not self.state.enabled:
                return
            # Check if this turn was already audited and cleared by in-loop pre_verify
            if (session_id or "default") in self._pre_verified_sessions:
                self._pre_verified_sessions.pop(session_id or "default", None)
                logger.debug("Advisor: skipping post_llm_call review, already audited by pre_verify")
                return
            turn_key = (session_id or "default", turn_id)
            if turn_id and turn_key == self.last_turn_key:
                return
            self.last_turn_key = turn_key
        if os.environ.get(ADVISOR_NO_REVIEW):
            return

        if not conversation_history:
            return

        # Multi-round depth guard: count consecutive advisor injections since last human message
        depth = _advisor_depth(conversation_history)
        max_rounds = getattr(self.state, "max_rounds", 2)
        if depth >= max_rounds:
            logger.debug(
                "Advisor: skipping review, reached max consecutive review depth (%d/%d)",
                depth, max_rounds,
            )
            return

        # Activity gate: skip review on trivial conversational turns with no tool calls
        if not _is_substantive_turn(user_message, assistant_response, conversation_history):
            return

        logger.debug(
            "Advisor: turn %s complete, user=%s, model=%s, msgs=%d",
            turn_id,
            (user_message or "")[:60],
            model or "?",
            len(conversation_history),
        )

        self._enqueue_review(
            TurnDelta(
                session_id=session_id or "default",
                turn_id=turn_id or "",
                user_message=user_message or "",
                assistant_response=assistant_response or "",
                conversation_history=list(conversation_history),
                model=model or "",
                context=copy_context(),
            )
        )

    def _enqueue_review(self, turn: TurnDelta) -> None:
        """Queue the newest completed turn without blocking hook dispatch."""
        self._ensure_worker()
        with self._queue_lock:
            self._idle.clear()
            try:
                self._review_queue.put_nowait(turn)
                return
            except queue.Full:
                pass

            try:
                dropped = self._review_queue.get_nowait()
                self._review_queue.task_done()
                if dropped is not None:
                    logger.info(
                        "Advisor: dropped stale queued turn %s in favor of %s",
                        dropped.turn_id,
                        turn.turn_id,
                    )
            except queue.Empty:
                pass
            self._review_queue.put_nowait(turn)

    def _ensure_worker(self) -> None:
        with self._state_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="hermes-advisor",
                daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            turn = self._review_queue.get()
            try:
                if turn is None:
                    return
                if getattr(turn, "context", None) is not None:
                    turn.context.run(self._review_turn, turn)
                else:
                    self._review_turn(turn)
            finally:
                with self._queue_lock:
                    self._review_queue.task_done()
                    if self._review_queue.unfinished_tasks == 0:
                        self._idle.set()

    def _review_turn(self, turn: TurnDelta) -> None:
        try:
            advice_list = self._run_review(
                user_message=turn.user_message,
                assistant_response=turn.assistant_response,
                conversation_history=turn.conversation_history,
                model=turn.model,
                turn_id=turn.turn_id,
                session_id=turn.session_id,
            )
        except Exception as exc:
            logger.warning("Advisor review failed for turn %s: %s", turn.turn_id, exc)
            return

        if not advice_list:
            logger.debug("Advisor: nothing to deliver for turn %s", turn.turn_id)
            return
        self._deliver_advice(advice_list, session_id=turn.session_id)

    def on_session_finalize(self, **_kwargs) -> None:
        """Give an in-flight review a small, fixed shutdown grace period."""
        if not self._idle.wait(SHUTDOWN_GRACE_SECONDS):
            logger.info(
                "Advisor: review still running after %.1fs shutdown grace; "
                "daemon worker will not delay exit",
                SHUTDOWN_GRACE_SECONDS,
            )

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Wait for queued work to finish. Intended for tests and diagnostics."""
        return self._idle.wait(timeout)

    # ── run the review ────────────────────────────────────────────────────

    def _run_review(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        model: str,
        turn_id: str,
        session_id: str,
    ) -> list[Advice]:
        """Build the prompt, call the advisor model, parse the result."""

        with self._state_lock:
            if not self.state.enabled:
                return []
            review_state = self._session_state(session_id)
            messages = self._build_review_prompt(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=conversation_history,
                cwd=os.getcwd(),
                review_state=review_state,
            )
            advisor_model = self.state.model
            advisor_provider = self.state.provider

        # Call the advisor model — use configured override or inherit primary
        kwargs = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if advisor_model:
            kwargs["model"] = advisor_model
        if advisor_provider:
            kwargs["provider"] = advisor_provider

        result = self.ctx.llm.complete(**kwargs)

        logger.debug(
            "Advisor: review complete, provider=%s model=%s tokens=%d",
            result.provider, result.model, result.usage.total_tokens if result.usage else 0,
        )

        with self._state_lock:
            if not self.state.enabled:
                return []
            advice = review_state.parse_response(result.text)
            self._save_session_state(session_id, review_state)
            return advice

    def _session_cwd(self, session_id: str = "") -> str:
        """Resolve session working directory from state.db or fallback to getcwd."""
        if not session_id:
            return os.getcwd()
        try:
            import sqlite3
            from hermes_constants import get_hermes_home
            db_path = get_hermes_home() / "state.db"
            if db_path.exists():
                with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT cwd FROM sessions WHERE id = ? LIMIT 1", (session_id,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        return row[0]
        except Exception:
            pass
        return os.getcwd()

    def consult_advisor(self, *, task: str, proposed_approach: str = "", session_id: str = "") -> str:
        """On-demand senior architect consultation during a turn."""
        cwd = self._session_cwd(session_id) or os.getcwd()
        guidelines = _find_guidelines(cwd)

        # Prefix caching optimization: static instructions & repo guidelines come FIRST
        prompt_parts = [
            "You are a Staff Software Architect paired with a coding agent. "
            "The agent is consulting you for architectural and design guidance before taking action.\n"
        ]
        if guidelines:
            prompt_parts.append(f"### Project Guidelines & Architecture (from repo):\n{guidelines}\n")

        prompt_parts.append(
            "### Instructions:\n"
            "Provide high-signal, concise (300-600 words) architectural direction:\n"
            "1. Recommended Pattern & Approach: The cleanest, most robust, idiomatic way to solve this.\n"
            "2. Potential Pitfalls & Edge Cases: Subtleties to watch out for.\n"
            "3. Existing Libraries / Primitives: Any standard or battle-tested libraries to prefer over custom code.\n"
            "Be direct, practical, and specific.\n"
        )
        prompt_parts.append(f"### Task / Goal:\n{task}\n")
        if proposed_approach:
            prompt_parts.append(f"### Proposed Approach:\n{proposed_approach}\n")

        prompt = "\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": "You are a senior Staff Software Architect advising a coding agent."},
            {"role": "user", "content": prompt},
        ]

        kwargs = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if self.state.model:
            kwargs["model"] = self.state.model
        if self.state.provider:
            kwargs["provider"] = self.state.provider

        result = self.ctx.llm.complete(**kwargs)
        return result.text.strip()

    def _session_user_directives(self, session_id: str = "") -> list[str]:
        """Pull user directives from state.db for the given session."""
        if not session_id:
            return []
        try:
            import sqlite3
            from hermes_constants import get_hermes_home
            db_path = get_hermes_home() / "state.db"
            if db_path.exists():
                with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT content FROM messages WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY timestamp ASC",
                        (session_id,)
                    )
                    msgs = [r[0] for r in cursor.fetchall() if r[0]]
                    return _collect_directives(msgs)
        except Exception:
            pass
        return []

    def on_pre_verify(
        self,
        *,
        session_id: str = "",
        platform: str = "",
        model: str = "",
        coding: bool = False,
        attempt: int = 0,
        final_response: str = "",
        changed_paths: list[str] | None = None,
        **kwargs,
    ) -> dict[str, str] | None:
        """In-loop stop gate: audits git diff before the agent finishes/delivers."""
        with self._state_lock:
            if not self.state.enabled:
                return None
        if os.environ.get(ADVISOR_NO_REVIEW):
            return None

        # Cap self-correction nudges to 2 attempts. When capped, turn finishes and
        # unresolved issues fall back to post_llm_call background review so findings aren't lost.
        if attempt >= 2:
            logger.debug(
                "Advisor pre_verify: reached max nudge attempt (%d >= 2), allowing turn to finish (will review in background)",
                attempt,
            )
            return None

        if not changed_paths:
            return None

        cwd = self._session_cwd(session_id) or os.getcwd()
        diff_text = _get_git_diff(cwd, changed_paths)
        if not diff_text:
            logger.debug("Advisor pre_verify: empty diff for %s, skipping", changed_paths)
            return None

        guidelines = _find_guidelines(cwd)
        directives = self._session_user_directives(session_id)

        prompt_parts = [
            "You are a senior Staff Software Engineer and Architect performing an in-loop verification gate.",
            "The coding agent has modified files and is preparing to report completion.",
            "Review the git diff below against the user's task and project guidelines before the agent is allowed to finish.\n"
        ]
        if guidelines:
            prompt_parts.append(f"### Project Guidelines & Architecture:\n{guidelines}\n")
        if directives:
            prompt_parts.append("### User Directives & Constraints (Session Context):\n" + "\n".join(f"- User: {d}" for d in directives) + "\n")

        prompt_parts.append(f"### Modified Files:\n" + "\n".join(f"- {p}" for p in changed_paths) + "\n")
        prompt_parts.append(f"### Git Diff:\n```diff\n{diff_text}\n```\n")
        if final_response:
            clipped_resp = final_response if len(final_response) <= 2000 else final_response[:800] + "\n... [truncated] ...\n" + final_response[-1200:]
            prompt_parts.append(f"### Agent's Proposed Completion Report:\n{clipped_resp}\n")

        prompt_parts.append(
            "### Audit Instructions:\n"
            "- Check: Does this diff introduce bugs, syntax errors, security flaws, broken contracts, or unverified claims?\n"
            "- Check: Did the agent claim something works without running real tests or checks?\n"
            "- Check: Does the diff violate any explicit user directive or project guideline?\n"
            "- If the changes are sound, tested, minimal, and ready to ship, reply EXACTLY: nothing to flag\n"
            "- If there is a real bug, regression, or missing check, reply with [BLOCKER] or [CONCERN] and the exact fix required.\n"
        )

        prompt = "\n".join(prompt_parts)
        messages = [
            {"role": "system", "content": "You are a Staff Software Engineer acting as a pre-completion verification gate."},
            {"role": "user", "content": prompt},
        ]

        kwargs_llm = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if self.state.model:
            kwargs_llm["model"] = self.state.model
        if self.state.provider:
            kwargs_llm["provider"] = self.state.provider

        try:
            result = self.ctx.llm.complete(**kwargs_llm)
            review_text = (result.text or "").strip()
        except Exception as exc:
            logger.warning("Advisor pre_verify LLM call failed: %s", exc)
            return None

        if not review_text or review_text.lower().startswith("nothing to flag"):
            logger.info("Advisor pre_verify: diff approved ('nothing to flag')")
            with self._state_lock:
                self._pre_verified_sessions[session_id or "default"] = tuple(sorted(changed_paths))
            return None

        if "[BLOCKER]" in review_text or "[CONCERN]" in review_text:
            logger.info("Advisor pre_verify: caught issue, nudging agent (attempt %d/2)", attempt + 1)
            return {
                "action": "continue",
                "message": (
                    f"Advisor caught issue before finish (attempt {attempt + 1}/2):\n"
                    f"{review_text}\n"
                    f"Fix this issue and verify before reporting completion."
                ),
            }

        with self._state_lock:
            self._pre_verified_sessions[session_id or "default"] = tuple(sorted(changed_paths))
        return None

    def _build_review_prompt(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        cwd: str,
        review_state: AdvisorState | None = None,
    ) -> list[dict]:
        """Build the message list for the advisor model."""

        # Base system prompt
        system_prompt = ADVISOR_SYSTEM_PROMPT

        # Append project guidelines if present (WATCHDOG.md, AGENTS.md, or CLAUDE.md)
        guidelines = _find_guidelines(cwd)
        if guidelines:
            system_prompt += (
                f"\n\nEspecially pay attention to project guidelines:\n"
                f"<guidelines>\n{guidelines}\n</guidelines>"
            )

        # Build the user content: reconfirm preamble + turn transcript
        user_content_parts = []

        # Reconfirm preamble for held notes
        preamble = (review_state or self.state).format_reconfirm_preamble()
        if preamble:
            user_content_parts.append(preamble)

        # Format the conversation history as a readable turn transcript
        transcript = self._format_history(
            user_message=user_message,
            response=assistant_response,
            history=conversation_history,
        )
        if transcript:
            user_content_parts.append(transcript)

        if not user_content_parts:
            return [{"role": "system", "content": system_prompt}]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

    @staticmethod
    def _format_history(
        *, user_message: str, response: str, history: list
    ) -> str:
        """Format conversation history into a markdown transcript for review.

        Shows the user prompt, tool calls/results made in THIS turn, and the assistant response.
        Brief context from recent prior turns is included without smearing old tool executions.
        """
        parts = []

        # Find where this turn started (last user message)
        last_user = max(
            (i for i, m in enumerate(history) if isinstance(m, dict) and m.get("role") == "user"),
            default=-1,
        )

        # Collect human user directives across the session, budget-bounded
        # User messages carry crucial decisions, authorizations, and constraints.
        user_msgs = []
        if last_user > 0:
            for msg in history[:last_user]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msgs.append(msg.get("content") or "")

        directives = _collect_directives(user_msgs)
        if directives:
            formatted_directives = [f"- User: {d}" for d in directives]
            parts.append("### User Directives & History (Session Context)\n" + "\n".join(formatted_directives))

        # Brief summary of recent assistant turns for continuity (last 3 assistant responses)
        recent_asst = []
        if last_user > 0:
            for msg in history[:last_user]:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    c = (msg.get("content") or "").strip()
                    if c:
                        recent_asst.append(f"- Assistant: {c[:250]}")
            if recent_asst:
                parts.append("### Recent Assistant Context\n" + "\n".join(recent_asst[-3:]))

        # Current turn user message
        parts.append(f"### Current Turn Being Reviewed\n#### User\n\n{(user_message or '').strip()}")

        # Build tool-call and result summary for the CURRENT TURN ONLY
        tool_calls: list[str] = []
        tool_results: list[str] = []

        current_turn_msgs = history[last_user + 1:] if last_user >= 0 else history
        for msg in current_turn_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    tc_name = function.get("name") or tool_call.get("name") or "?"
                    tc_args = function.get("arguments", tool_call.get("arguments", {}))
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(tc_args, (dict, list)):
                        tc_text = json.dumps(tc_args, ensure_ascii=False, indent=1)
                    else:
                        tc_text = str(tc_args)
                    tool_calls.append(f"→ tool `{tc_name}`: {tc_text[:800]}")

                # Retain compatibility with providers that use content blocks.
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") in {"toolCall", "tool_use"}:
                                tc_name = block.get("name", "?")
                                tc_args = block.get("arguments", block.get("input", {}))
                                tc_str = json.dumps(
                                    tc_args, ensure_ascii=False, indent=1
                                )[:800]
                                tool_calls.append(f"→ tool `{tc_name}`: {tc_str}")
            elif role == "tool":
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content, dict):
                    text = json.dumps(content, ensure_ascii=False)
                else:
                    text = str(content)
                if text.strip():
                    tool_name = msg.get("name") or "tool"
                    tool_results.append(
                        f"→ `{tool_name}` result: {text.strip()[:1200]}"
                    )

        if tool_calls:
            parts.append("#### Tool calls in this turn\n\n" + "\n".join(tool_calls))
        if tool_results:
            parts.append("#### Tool results in this turn\n\n" + "\n".join(tool_results))

        # Assistant response (clipped to 4000 chars, preserving head and tail)
        if response and response.strip():
            resp_text = response.strip()
            if len(resp_text) > 4000:
                resp_text = resp_text[:1500] + "\n... [truncated] ...\n" + resp_text[-2500:]
            parts.append(f"#### Assistant\n\n{resp_text}")

        return "\n\n".join(parts)

    # ── deliver advice back to the conversation ──────────────────────────

    def _deliver_advice(self, advice_list: list[Advice], session_id: str = ""):
        """Inject advice into the active conversation.

        Uses ctx.inject_message(). Delivers to CLI, Desktop/TUI, or Gateway.
        """
        if not advice_list:
            return

        lines = []
        for a in advice_list:
            lines.append(f"{a.tag()} {a.note}")

        advisory_text = "\n".join(lines)
        full_msg = (
            f"◆ Advisor review (automated reviewer, NOT the user)\n\n"
            f"{advisory_text}\n\n"
            f"[Directive: This message was generated by an automated background reviewer, NOT the user. "
            f"First answer the user's most recent actual message (the one before this review). "
            f"Apply any technical fixes above secondarily and briefly. Never attribute this review to the user.]"
        )

        ok = self.ctx.inject_message(full_msg, role="user", session_key=session_id or None)
        if ok:
            logger.info("Advisor: injected %d item(s) into conversation", len(advice_list))
        else:
            logger.info(
                "Advisor: %d item(s) not injected (delivery unavailable).",
                len(advice_list),
            )

    # ── interactive model selector ───────────────────────────────────────

    def _interactive_select(self) -> str | None:
        """Open Hermes' native provider/model modal for the advisor slot."""

        def apply_selection(result) -> str:
            if not result.success:
                return f"Advisor model selection failed: {result.error_message}"
            with self._state_lock:
                self.state.model = result.new_model
                self.state.provider = result.target_provider
                self._save_state()
            return (
                f"Advisor model set to: {result.new_model} "
                f"({result.provider_label or result.target_provider})"
            )

        # Optional seam: builds without ``request_model_selection`` keep the
        # text commands (``/advisor provider``, ``/advisor model <name>``) and
        # say so, instead of raising AttributeError on every picker attempt.
        request_selection = getattr(self.ctx, "request_model_selection", None)
        if not callable(request_selection):
            return (
                "Interactive advisor model selection is not available in this "
                "Hermes build.\nUse /advisor provider <name> and "
                "/advisor model <name> here."
            )

        with self._state_lock:
            current_provider = self.state.provider
            current_model = self.state.model
        opened = request_selection(
            apply_selection,
            current_provider=current_provider,
            current_model=current_model,
        )
        if opened:
            return None
        return (
            "Interactive advisor model selection is available in the Hermes CLI.\n"
            "Use /advisor model <name> and /advisor provider <name> here."
        )

    # ── slash command ─────────────────────────────────────────────────────

    def handle_command(self, args: str) -> str | None:
        """Handle /advisor [on|off|status|model|provider|providers|models|test].

        Keywords match case-insensitively; values (model names, provider
        slugs, test notes) keep the case the user typed.
        """
        raw = args.strip()
        tokens = raw.split(None, 1)
        head = tokens[0].lower() if tokens else ""
        value = tokens[1].strip() if len(tokens) > 1 else ""

        # ── status ──
        if head in ("", "status", "config") and not (head == "config" and value):
            return self._format_status()

        # ── on ──
        if head == "on":
            self.state.enabled = True
            self._save_state()
            return "Advisor on."

        # ── off ──
        if head == "off":
            self.state.enabled = False
            self._clear_held_notes()
            self._save_state()
            return "Advisor off."

        # ── model (no args) — open the native provider/model picker ──
        if head == "model" and not value:
            return self._interactive_select()

        # ── model <name> ──
        if head == "model":
            self.state.model = value
            self._save_state()
            return f"Advisor model set to: {value}"

        # Provider is selected as the first stage of /advisor model.
        if head == "provider" and not value:
            return "Provider selection is part of /advisor model."

        # ── provider <name> ──
        if head == "provider":
            self.state.provider = value
            self._save_state()
            return f"Advisor provider set to: {value}"

        # ── config <key> <value> ──
        if head == "config":
            sub_tokens = value.split(None, 1)
            sub = sub_tokens[0].lower() if sub_tokens else ""
            sub_value = sub_tokens[1].strip() if len(sub_tokens) > 1 else ""
            if sub == "model" and sub_value:
                self.state.model = sub_value
                self._save_state()
                return f"Advisor model set to: {sub_value}"
            if sub == "provider" and sub_value:
                self.state.provider = sub_value
                self._save_state()
                return f"Advisor provider set to: {sub_value}"
            return "Usage: /advisor config <model|provider> <value>"

        # ── providers — the picker's provider rows (no private catalog) ──
        if head in ("providers", "list-providers"):
            return self._list_providers()

        # ── models [provider] — the picker's model rows (no private catalog) ──
        if head in ("models", "list-models"):
            return self._list_models(value)

        # ── rounds <n> ──
        if head in ("rounds", "max-rounds"):
            try:
                n = int(value)
                if n < 1 or n > 20:
                    return "Rounds must be between 1 and 20."
                self.state.max_rounds = n
                self._save_state()
                return f"Advisor max rounds set to: {n}"
            except ValueError:
                return "Usage: /advisor rounds <1-20>"

        # ── test — inject a test advice message ──
        if head == "test":
            import re

            m = re.match(r"(nit|concern|blocker)\s+([\s\S]+)$", value, re.IGNORECASE)
            if m:
                sev = Severity(m.group(1).lower())
                note = m.group(2).strip()
                self._deliver_advice([Advice(note=note, severity=sev)])
                return f"Advisor: delivered test {sev.value}."
            return "Usage: /advisor test <nit|concern|blocker> <note>"

        return "Usage: /advisor [on|off|status|config|model|provider|providers|models|rounds|test]"

    def _format_status(self) -> str:
        state = "enabled" if self.state.enabled else "disabled"
        model = self.state.model or "(inherit primary)"
        provider = self.state.provider or "(inherit primary)"
        max_rounds = getattr(self.state, "max_rounds", 2)
        held = self._held_count()
        return (
            f"Advisor {state}.\n"
            f"  model:      {model}\n"
            f"  provider:   {provider}\n"
            f"  max rounds: {max_rounds}\n"
            f"  held:       {held}\n"
            f"Usage: /advisor [on|off|status|config|model|provider|rounds]"
        )

    def _provider_rows(self) -> list[dict]:
        """Provider rows for the text listings, from the documented plugin seam.

        ``providers`` (``list_providers`` / ``get_provider_profile``) is the
        documented extension point — the same provider registry the native
        ``/model`` picker is built from — whereas ``hermes_cli.inventory`` is
        core-internal and carries no compatibility promise (catalog rule 9).

        The interactive path is unaffected: ``/advisor model`` opens Hermes' own
        picker through ``ctx.request_model_selection``, so no catalog of the
        advisor's own is ever needed for selection. These rows only feed the
        text listings, for surfaces where a modal cannot open (gateway, plain
        output).
        """
        try:
            from providers import list_providers

            profiles = list_providers()
        except Exception as exc:
            logger.warning("Advisor: could not load the provider registry: %s", exc)
            return []

        with self._state_lock:
            current = (self.state.provider or "").strip().lower()
        rows = []
        for profile in profiles:
            slug = (getattr(profile, "name", "") or "").strip()
            if not slug:
                continue
            label = (getattr(profile, "display_name", "") or "").strip() or slug
            rows.append({
                "slug": slug,
                "name": label,
                "is_current": bool(current) and slug.lower() == current,
                "models": list(getattr(profile, "fallback_models", []) or []),
            })
        return rows

    def _list_providers(self) -> str:
        providers = self._provider_rows()
        if not providers:
            return (
                "No providers found in the plugin provider registry. "
                "Add one via hermes setup or model.provider in config.yaml."
            )
        lines = ["Providers available to /advisor model:"]
        for p in providers:
            slug = p.get("slug") or "?"
            label = p.get("name") or slug
            marker = "  (current)" if p.get("is_current") else ""
            lines.append(f"  {slug}  —  {label}{marker}")
        return "\n".join(lines)

    def _list_models(self, provider_arg: str) -> str:
        target = (provider_arg or "").strip()
        with self._state_lock:
            target = target or self.state.provider or ""
        if not target:
            return "Usage: /advisor models <provider>\n(run /advisor providers first)"
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(target)
        except Exception as exc:
            logger.warning("Advisor: provider lookup failed for %r: %s", target, exc)
            profile = None
        if profile is None:
            return (
                f"No provider '{target}' found. Run /advisor providers for valid names."
            )
        # A profile owns its catalog: fetch live ids (public contract, the profile
        # resolves its own credentials), else fall back to its curated list.
        models: list[str] = []
        try:
            models = [m for m in (profile.fetch_models() or []) if isinstance(m, str)]
        except Exception as exc:
            logger.debug("Advisor: catalog fetch failed for %s: %s", target, exc)
        if not models:
            models = [m for m in (getattr(profile, "fallback_models", []) or []) if isinstance(m, str)]
        total = None
        if not models:
            return (
                f"No models listed for '{target}'. Open /advisor model to pick "
                "interactively, or refresh with /model --refresh."
            )
        lines = [f"Models for {target}:"]
        shown = sorted(models)[:30]
        for m in shown:
            lines.append(f"  {m}")
        hidden = len(models) - len(shown)
        capped = (total - len(models)) if isinstance(total, int) else 0
        if hidden or capped > 0:
            lines.append(f"  ... and {hidden + max(capped, 0)} more")
        return "\n".join(lines)
