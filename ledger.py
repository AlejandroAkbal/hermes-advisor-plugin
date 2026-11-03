"""Deterministic Verification Ledger for Advisor Plugin.

Provides zero-latency, deterministic static verification of drafts before calling an LLM:
1. Placeholder / Empty Draft Detection (e.g. "Testing verify_turn", "...", stubs).
2. Unhandled Tool Error Detection: flags drafts claiming success when tool events failed.
3. Truncated Identifier Citation: flags IDs/hashes cited from truncated outputs.
4. Hard Assertion Corroboration: checks required diagnostic tools for git/test/build claims.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

# Drafts that are clearly placeholders or stubs
_PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*(testing verify_turn|test draft|placeholder|\.{2,}|todo)\s*$", re.IGNORECASE),
    re.compile(r"^\s*draft\s*:?\s*$", re.IGNORECASE),
]

# Action patterns requiring specific diagnostic tools in the turn
_CORROBORATION_RULES = [
    {
        "name": "test_suite_passed",
        "pattern": re.compile(r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )\b(all \d+ tests? pass(ed)?|test suite passes?|tests? pass(ed)?)\b"),
        "required_tool_patterns": [r"\bunittest\b", r"\bpytest\b", r"\b(pnpm|bun|yarn|npm) (run )?test\b", r"\bcargo test\b", r"\bgo test\b", r"\bmake test\b", r"\btox\b"],
        "suggested_command": "Run your test suite (e.g. pytest or unittest) before asserting tests pass.",
    },
    {
        "name": "clean_git_tree",
        "pattern": re.compile(r"(?i)(?<!make sure )(?<!ensure )(?<!verify )\b(working tree (is )?clean|clean working tree|git status clean|git tree (is )?clean|no uncommitted changes)\b"),
        "required_tool_patterns": [r"git status", r"git diff", r"git -C .* status"],
        "suggested_command": "Run `git status` to verify the working tree is clean.",
    },
    {
        "name": "git_pushed",
        "pattern": re.compile(
            r"(?i)"
            r"(?<!not )(?<!never )(?<!n't )(?<!yet )(?<!no )(?<!nothing )(?<!without )(?<!neither )"
            r"(?<!ready to be )(?<!going to be )(?<!yet to be )(?<!to be )"
            r"(?<!not been )(?<!never been )(?<!yet been )"
            r"(?<!were )(?<!was )(?<!been )(?<!be )(?<!changes )(?<!commits )(?<!branch )(?<!branches )"
            r"\b(?:"
            r"(?:pushed\s+(?:to\s+(?:the\s+)?(?:origin|remote|github)|(?:the\s+|all\s+)?(?:changes?|branch(?:es)?(?:\s+[\w\-\/\.]+)?|commits?|updates?)\s+to\s+(?:the\s+)?(?:origin|remote|github)))"
            r"|(?<!no )(?<!nothing )(?<!not )(?<!n't )(?<!never )(?<!without )(?<!neither )(?<!yet )"
            r"(?:changes?|branch(?:es)?(?:\s+[\w\-\/\.]+)?|commits?|it)\s+"
            r"(?:(?:have|has)\s+been\s+|were\s+|was\s+)?"
            r"pushed\s+to\s+(?:the\s+)?(?:origin|remote|github)"
            r"|(?:i|we)\s+(?:ran|did)\s+git\s+push"
            r")\b"
        ),
        "required_tool_patterns": [r"git push", r"push -u origin", r"push origin"],
        "suggested_command": "Run `git push` before asserting changes were pushed.",
    },
    {
        "name": "pr_created",
        "pattern": re.compile(
            r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )(?<!to )(?<!will )(?<!not )(?<!yet )(?<!n't )(?<!never )(?<!haven't )(?<!hasn't )(?<!hadn't )(?<!without )(?<!yet to )"
            r"(?<!cron )(?<!log )(?<!earlier )(?<!prior )"
            r"\b(?:(?:opened|created|submitted)\s+(?:a\s+|the\s+)?(?:new\s+|draft\s+)?pr(?: #?\d+)?"
            r"|(?:opened|created|submitted)\s+(?:a\s+|the\s+)?(?:new\s+|draft\s+)?pull request(?: #?\d+)?"
            r"|(?<!no )(?<!not )(?<!n't )(?<!never )(?<!yet )(?<!cron )(?<!log )pr #?\d+\s+(?:was |is )?(?:opened|created))\b"
        ),
        "required_tool_patterns": [r"\bgh\s+pr\s+create\b"],
        "suggested_command": "Run `gh pr create` before claiming a PR was created or opened.",
    },
    {
        "name": "pr_merged",
        "pattern": re.compile(
            r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )(?<!to )(?<!will )(?<!not )(?<!n't )(?<!never )(?<!without )"
            r"(?<!cron )(?<!log )(?<!earlier )(?<!prior )"
            r"\b(?:(?:merged|merging)\s+(?:the\s+)?(?:new\s+|draft\s+)?(?:pr|pull request)(?:\s+#?\d+)?"
            r"|(?<!cron )(?<!log )(?<!earlier )(?<!prior )(?:pr|pull request)\s+#?\d+\s+(?:was\s+|is\s+)?merged"
            r"|(?:automatically\s+)?merged\s+the\s+pr\b"
            r")\b"
        ),
        "required_tool_patterns": [r"\bgh\s+pr\s+merge\b", r"\bgh\s+pr\s+view\b"],
        "suggested_command": "Verify PR merge via `gh pr merge` or `gh pr view`.",
    },
    {
        "name": "git_committed",
        "pattern": re.compile(
            r"(?i)"
            r"(?<!make sure )(?<!ensure )(?<!verify )(?<!if )(?<!to )(?<!will )"
            r"(?<!not )(?<!n't )(?<!never )(?<!yet )(?<!no )(?<!without )(?<!neither )"
            r"(?<!ready to be )(?<!going to be )(?<!yet to be )"
            r"\b(?:"
            r"(?:committed\s+(?:the\s+|all\s+)?(?:changes?|diffs?|files?|updates?|fixes?|code)\b)"
            r"|(?:committed\s+(?:\w+\s+)?to\s+(?:the\s+)?branch\s+[a-zA-Z0-9_\-\.\/]+)"
            r"|(?:created\s+(?:a\s+|the\s+)?commit\b)"
            r"|(?:changes?\s+(?:have\s+been\s+|were\s+)?committed\b)"
            r")\b"
        ),
        "required_tool_patterns": [r"\bgit\s+(?:-c\s+\S+\s+)*commit\b", r"\bgit\s+-C\s+.*?\s+commit\b"],
        "suggested_command": "Run `git commit` before claiming changes were committed.",
    },
    {
        "name": "coderabbit_reviewed",
        "pattern": re.compile(
            r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )(?<!run )"
            r"\b(?:coderabbit\s+(?:review\s+)?(?:passed|clean|completed|finished|approved|reported)"
            r"|(?:ran|executed)\s+coderabbit\s+review)\b"
        ),
        "required_tool_patterns": [r"\bcoderabbit\s+review\b"],
        "suggested_command": "Run `coderabbit review` before asserting CodeRabbit review completed or passed.",
    },
]

# Keywords indicating an unhandled tool failure in raw tool results
_TOOL_FAILURE_PATTERNS = [
    re.compile(r"""(?i)["']?exit[_-]?code["']?\s*[:=]\s*([1-9]\d*)"""),
    re.compile(r"\bexit code\s+([1-9]\d*)\b"),
    re.compile(r"\b(Permission denied|command not found|SyntaxError)\b|Traceback \(most recent call last\)"),
]


def _result_indicates_failure(res_str: str) -> bool:
    """Check if tool output indicates a real command failure."""
    if not res_str:
        return False
    for err_pat in _TOOL_FAILURE_PATTERNS:
        if err_pat.search(res_str):
            return True
    for line in res_str.splitlines():
        line_clean = line.strip().lower()
        if (
            line_clean.startswith("error:")
            or line_clean.startswith("fatal:")
            or "pull request create failed" in line_clean
            or "command failed" in line_clean
            or "review failed" in line_clean
        ):
            return True
    return False


def _extract_created_pr_number(args_str: str, res_str: str) -> str | None:
    """Extract actual created PR number from gh pr create output or URL."""
    url_m = re.search(r"https?://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)", res_str)
    if url_m:
        return url_m.group(1)
    num_m = re.search(r"(?<!\w)#(\d+)\b", res_str)
    if num_m:
        return num_m.group(1)
    return None


def _pr_view_indicates_merged(res_str: str) -> bool:
    """Authoritatively check if gh pr view output indicates MERGED state."""
    res_clean = res_str.strip()
    if res_clean.startswith("{") and res_clean.endswith("}"):
        try:
            data = json.loads(res_clean)
            if isinstance(data, dict):
                return str(data.get("state", "")).upper() == "MERGED"
        except Exception:
            pass
    return bool(re.search(r"(?im)^\s*state:\s*merged\s*$", res_str))


def _extract_merged_pr_number(args_str: str, res_str: str) -> str | None:
    """Extract actual merged PR number from gh pr merge/view command arguments or URL."""
    res_clean = res_str.strip()
    if res_clean.startswith("{") and res_clean.endswith("}"):
        try:
            data = json.loads(res_clean)
            if isinstance(data, dict) and "number" in data:
                return str(data["number"])
        except Exception:
            pass
    if re.search(r"\bgh\s+pr\s+merge\b", args_str):
        url_m = re.search(r"https?://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)", res_str)
        if url_m:
            return url_m.group(1)
        res_m = re.search(r"(?i)\b(?:merged\s+)?(?:pull request|pr)\s+(?:[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)?#?(\d+)\b", res_str)
        if res_m:
            return res_m.group(1)
        num_m = re.search(r"(?<!\w)#(\d+)\b", res_str)
        if num_m:
            return num_m.group(1)

    try:
        tokens = shlex.split(args_str)
    except Exception:
        tokens = args_str.split()

    subcmd_idx = -1
    for i, tok in enumerate(tokens):
        if tok in ("merge", "view") and i > 0 and tokens[i - 1] == "pr":
            subcmd_idx = i
            break

    if subcmd_idx != -1:
        opts_with_val = {
            "-b", "--body", "-F", "--body-file", "-t", "--title", "-s", "--subject",
            "-R", "--repo", "-m", "--match-head-commit", "-q", "--jq", "--template"
        }
        idx = subcmd_idx + 1
        while idx < len(tokens):
            tok = tokens[idx]
            if tok in opts_with_val:
                idx += 2
                continue
            if tok.startswith("-"):
                idx += 1
                continue
            url_match = re.search(r"https?://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)", tok)
            if url_match:
                return url_match.group(1)
            num_match = re.search(r"^(?:[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)?#?(\d+)$", tok)
            if num_match:
                return num_match.group(1)
            idx += 1

    return None


class LedgerVerdict:
    def __init__(self, passed: bool, reason: str = "", missing_checks: list[str] | None = None, rule: str = ""):
        self.passed = passed
        self.reason = reason
        self.missing_checks = missing_checks or []
        self.rule = rule

    def to_dict(self) -> dict[str, Any]:
        if self.passed:
            return {"status": "approved", "reason": self.reason or "Deterministic ledger verified."}
        res: dict[str, Any] = {
            "status": "rejected",
            "reason": self.reason,
            "rule": self.rule,
        }
        if self.missing_checks:
            res["missing_checks"] = self.missing_checks
        return res


def audit_draft_ledger(
    draft: str,
    tool_events: list[dict],
    evidence: str = "",
    claims: list[str] | str | None = None,
) -> LedgerVerdict:
    """Deterministically audit draft against tool events before invoking LLM."""
    clean_draft = (draft or "").strip()
    if isinstance(claims, list):
        claims_text = " ".join(str(c) for c in claims).strip()
    else:
        claims_text = str(claims or "").strip()

    if not clean_draft:
        clean_draft = claims_text

    if not clean_draft:
        return LedgerVerdict(
            passed=False,
            reason="Draft is empty. A valid response draft or claim list is required for verification.",
            rule="empty_draft"
        )

    # Combined text for checking factual claims across draft and explicit claims
    scan_text = f"{clean_draft} {claims_text}".strip()

    # 1. Check for placeholder drafts
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.search(clean_draft):
            return LedgerVerdict(
                passed=False,
                reason=f"Draft '{clean_draft[:50]}' is a placeholder or stub, not a user-facing response.",
                rule="placeholder_draft"
            )

    combined_tool_text = " ".join(
        f"{ev.get('tool', '')} {ev.get('args', '')}" for ev in tool_events
    )
    valid_evidence_lines = []
    if evidence:
        skip_block = False
        for line in evidence.splitlines():
            if line.startswith("[Active Probe"):
                # Only keep confirmed probes as evidence; drop CONTRADICTED or INCONCLUSIVE blocks
                skip_block = not line.startswith("[Active Probe CONFIRMED]")
            if not skip_block:
                valid_evidence_lines.append(line)
    filtered_evidence = "\n".join(valid_evidence_lines)
    combined_evidence = f"{combined_tool_text} {filtered_evidence}".lower()

    # 2. Check corroboration rules
    missing_checks = []
    unbacked_reasons = []
    for rule in _CORROBORATION_RULES:
        rule_match = rule["pattern"].search(scan_text)
        if rule_match:
            # Check if any required pattern was executed in tool events
            matching_events = [
                ev for ev in tool_events
                if any(
                    re.search(req_pat, f"{ev.get('tool', '')} {ev.get('args', '')}", re.IGNORECASE)
                    for req_pat in rule["required_tool_patterns"]
                )
            ]
            if not matching_events:
                # Active probe notes or caller evidence can only corroborate state rules (e.g. clean_git_tree),
                # not execution-bound actions requiring tool commands
                if rule["name"] == "clean_git_tree":
                    matched_evidence = any(
                        re.search(req_pat, combined_evidence, re.IGNORECASE)
                        for req_pat in rule["required_tool_patterns"]
                    )
                    if not matched_evidence:
                        missing_checks.append(rule["suggested_command"])
                        unbacked_reasons.append(
                            f"Claim matching '{rule['name']}' has no supporting tool execution in this turn."
                        )
                        continue
                else:
                    missing_checks.append(rule["suggested_command"])
                    unbacked_reasons.append(
                        f"Claim matching '{rule['name']}' has no supporting tool execution in this turn."
                    )
                    continue

            # Deep verification for specific rules
            if rule["name"] == "coderabbit_reviewed" and matching_events:
                latest_rev = matching_events[-1]
                res_str = str(latest_rev.get("result", ""))
                if _result_indicates_failure(res_str):
                    missing_checks.append("Resolve CodeRabbit errors before claiming review passed.")
                    unbacked_reasons.append("CodeRabbit review command reported an error or failure.")

            elif rule["name"] == "pr_created" and matching_events:
                cited_matches = list(rule["pattern"].finditer(scan_text))
                for m in cited_matches:
                    pr_num_m = re.search(r"(?i)\b(?:pr|pull request)\s+#?(\d+)\b", m.group(0))
                    claimed_num = pr_num_m.group(1) if pr_num_m else None
                    matching_success = False
                    for ev in matching_events:
                        res_str = str(ev.get("result", ""))
                        args_str = str(ev.get("args", ""))
                        if not _result_indicates_failure(res_str):
                            if claimed_num is None:
                                matching_success = True
                                break
                            actual_num = _extract_created_pr_number(args_str, res_str)
                            if actual_num == claimed_num:
                                matching_success = True
                                break
                    if not matching_success:
                        if claimed_num:
                            missing_checks.append(f"Verify PR #{claimed_num} identity via `gh pr view {claimed_num}`.")
                            unbacked_reasons.append(f"No successful `gh pr create` event corroborated creating PR #{claimed_num}.")
                        else:
                            missing_checks.append("Verify `gh pr create` succeeded before claiming a PR was created.")
                            unbacked_reasons.append("Tool `gh pr create` failed to create a PR.")

            elif rule["name"] == "pr_merged" and matching_events:
                cited_matches = list(rule["pattern"].finditer(scan_text))
                for m in cited_matches:
                    match_str = m.group(0).lower()
                    is_action_claim = ("was merged" not in match_str and "is merged" not in match_str) and bool(
                        re.search(r"\b(?:merged|merging)\s+(?:the\s+)?(?:new\s+|draft\s+)?(?:pr|pull request|#?\d+)\b", match_str)
                    )
                    pr_num_m = re.search(r"(?i)\b(?:pr|pull request)\s+#?(\d+)\b", m.group(0))
                    claimed_num = pr_num_m.group(1) if pr_num_m else None
                    matching_success = False
                    for ev in matching_events:
                        res_str = str(ev.get("result", ""))
                        args_str = str(ev.get("args", ""))
                        if not _result_indicates_failure(res_str):
                            if re.search(r"\bgh\s+pr\s+view\b", args_str):
                                if is_action_claim:
                                    continue
                                if not _pr_view_indicates_merged(res_str):
                                    continue
                            if claimed_num is None:
                                matching_success = True
                                break
                            actual_num = _extract_merged_pr_number(args_str, res_str)
                            if actual_num == claimed_num:
                                matching_success = True
                                break
                    if not matching_success:
                        if claimed_num:
                            if is_action_claim:
                                missing_checks.append(f"Verify PR #{claimed_num} merge action via `gh pr merge {claimed_num}`.")
                                unbacked_reasons.append(f"No successful `gh pr merge` event corroborated merging PR #{claimed_num}.")
                            else:
                                missing_checks.append(f"Verify PR #{claimed_num} merge via `gh pr view {claimed_num}`.")
                                unbacked_reasons.append(f"No successful `gh pr merge` or `gh pr view` event corroborated merged status for PR #{claimed_num}.")
                        else:
                            missing_checks.append("Verify PR merge via `gh pr merge` or `gh pr view`.")
                            unbacked_reasons.append("Tool `gh pr merge` or `gh pr view` failed or did not confirm merge.")

            elif rule["name"] == "git_committed" and matching_events:
                latest_commit = matching_events[-1]
                res_str = str(latest_commit.get("result", ""))
                if _result_indicates_failure(res_str) or "nothing to commit" in res_str.lower():
                    missing_checks.append("Verify `git commit` created a commit before asserting changes were committed.")
                    unbacked_reasons.append("Tool `git commit` failed or reported nothing to commit.")

            elif rule["name"] == "git_pushed" and matching_events:
                latest_push = matching_events[-1]
                res_str = str(latest_push.get("result", ""))
                if _result_indicates_failure(res_str):
                    missing_checks.append("Verify `git push` succeeded before claiming changes were pushed.")
                    unbacked_reasons.append("Tool `git push` failed to push changes.")

            elif rule["name"] == "test_suite_passed" and matching_events:
                latest_test = matching_events[-1]
                res_str = str(latest_test.get("result", ""))
                has_positive_failure = bool(
                    _result_indicates_failure(res_str)
                    or re.search(r"\bFAILED\b", res_str)
                    or re.search(r"(?i)\b(?:failures|errors)\s*=\s*[1-9]\d*\b", res_str)
                    or re.search(r"(?i)\b[1-9]\d*\s+failed\b", res_str)
                    or re.search(r"(?i)\b[1-9]\d*\s+errors?\b", res_str)
                )
                if has_positive_failure:
                    missing_checks.append("Fix failing tests before claiming the test suite passed.")
                    unbacked_reasons.append("Test runner reported failures or errors.")

    if missing_checks:
        return LedgerVerdict(
            passed=False,
            reason=" ".join(unbacked_reasons),
            missing_checks=missing_checks,
            rule="unbacked_assertion"
        )

    # 3. Check for severe unhandled tool errors if draft claims unqualified success
    claims_clean_success = re.search(r"(?i)\b(all clean|succeeded|done successfully|everything works?|healthy)\b", scan_text)
    if claims_clean_success and tool_events:
        # Check only the final execution result per tool to avoid flagging retried/recovered commands
        latest_event_per_tool = {}
        for ev in tool_events:
            tool_name = ev.get("tool", "")
            if tool_name:
                latest_event_per_tool[tool_name] = ev

        for tool_name, ev in latest_event_per_tool.items():
            res = str(ev.get("result", ""))
            for err_pat in _TOOL_FAILURE_PATTERNS:
                m = err_pat.search(res)
                if m:
                    # An error occurred in the final tool call of this tool, but draft claims clean success
                    ack_words = ["failed", "error", "exit", "issue", "problem", "unable to", "warning"]
                    if not any(w in clean_draft.lower() for w in ack_words):
                        return LedgerVerdict(
                            passed=False,
                            reason=(
                                f"Tool '{tool_name}' reported a failure ({m.group(0)}), but the draft claims "
                                "clean success without acknowledging or explaining the failure."
                            ),
                            missing_checks=["Inspect tool error and explain or resolve it in the draft."],
                            rule="unacknowledged_tool_failure"
                        )

    return LedgerVerdict(passed=True, reason="Deterministic ledger checks passed.")
