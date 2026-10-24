"""Advisor data models."""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Markers indicating leaked machine output rather than prose advice.
_TOOL_CALL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<function",
    "<parameter",
    "<tool_use",
    "<invoke",
)
MAX_NOTE_CHARS = 4000


def is_malformed_note(note: str) -> bool:
    """True when a note is leaked machine output rather than prose advice."""
    if not note or not note.strip():
        return True
    stripped = note.strip()
    if len(stripped) > MAX_NOTE_CHARS:
        return True
    lowered = stripped.casefold()
    if any(marker in lowered for marker in _TOOL_CALL_MARKERS):
        return True
    # Bare JSON payload — a dict/list serialised straight into the review.
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return True
        except (ValueError, TypeError):
            pass
    return False


class Severity(str, Enum):
    SUGGESTION = "suggestion"
    NIT = "nit"
    CONCERN = "concern"
    BLOCKER = "blocker"


SEVERITY_RANK = {
    Severity.SUGGESTION: 1,
    Severity.NIT: 1,
    Severity.CONCERN: 2,
    Severity.BLOCKER: 3,
}


@dataclass
class Advice:
    """A single piece of advice with severity."""

    note: str
    severity: Severity = Severity.SUGGESTION

    def tag(self) -> str:
        return f"[{self.severity.value.upper()}]"


@dataclass
class TurnDelta:
    """Data about one agent turn."""

    session_id: str
    turn_id: str
    user_message: str
    assistant_response: str
    conversation_history: list  # full message list
    model: str
    context: Any = None


@dataclass
class AdvisorState:
    """Persistent state for the advisor plugin."""

    enabled: bool = False
    held_notes: list[dict] = field(default_factory=list)
    # Per-advisor model override — empty means inherit primary model
    model: str = ""
    provider: str = ""

    def dedupe_key(self, note: str) -> str:
        return re.sub(r"\s+", " ", note.strip()).casefold()

    def has_held(self) -> bool:
        return len(self.held_notes) > 0

    def format_reconfirm_preamble(self) -> str:
        """Build a reconfirmation preamble from held notes."""
        if not self.held_notes:
            return ""
        items = "\n".join(
            f"- [{h['severity'].upper()}] {h['note']}"
            for h in self.held_notes
        )
        return (
            "### Prior observations\n\n"
            f"{items}\n\n---"
        )

    def parse_response(self, text: str) -> list[Advice]:
        """Parse coworker review and return actionable advice to deliver immediately."""
        if not text or not text.strip():
            self.held_notes = []
            return []

        text = text.strip()
        if "nothing to flag" in text.lower() and len(text) < 40:
            self.held_notes = []
            return []

        parsed: list[Advice] = []

        # Find all tagged blocks: [SUGGESTION], [CONCERN], [BLOCKER], [NIT]
        pattern = r"\[(SUGGESTION|CONCERN|BLOCKER|NIT)\]"
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if matches:
            for idx, m in enumerate(matches):
                tag_str = m.group(1).lower()
                sev = Severity(tag_str)
                start_pos = m.end()
                end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                note_content = text[start_pos:end_pos].strip().strip(":").strip()
                if note_content and not is_malformed_note(note_content):
                    parsed.append(Advice(note=note_content, severity=sev))
                elif note_content:
                    logger.warning(
                        "Advisor: dropping malformed %s note (%d chars)",
                        sev.value, len(note_content),
                    )
        else:
            # Unstructured response
            if len(text) > 20:
                if is_malformed_note(text):
                    logger.warning("Advisor: dropping malformed unstructured review (%d chars)", len(text))
                else:
                    parsed.append(Advice(note=text, severity=Severity.SUGGESTION))

        self.held_notes = []
        return parsed

    def serialize(self) -> dict:
        return {
            "enabled": self.enabled,
            "held_notes": self.held_notes,
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "AdvisorState":
        return cls(
            enabled=data.get("enabled", True),
            held_notes=data.get("held_notes", []),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )
