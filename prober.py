"""Active Autonomous Probe Engine for Advisor Plugin.

Provides read-only, non-mutating active probes to verify common system and repo assertions
without requiring an extra agent turn:
- Repo clean state (git status --porcelain)
- Current git branch (git branch --show-current)
- Git commit verification (git log -1)
- File existence / non-empty checks (test -e)
- Service presence (launchctl list)
- Process PID presence (ps -p)
- TCP port listening state (lsof -iTCP -sTCP:LISTEN)

All probes are bounded by strict timeouts (<= 2.0s) and executed in a read-only manner.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

PROBE_TIMEOUT_SECONDS = 2.0


class ProbeResult:
    def __init__(self, executed: bool, success: bool, claim: str, command: str, output: str):
        self.executed = executed
        self.success = success
        self.claim = claim
        self.command = command
        self.output = output.strip()

    def format_evidence(self) -> str:
        if not self.executed:
            status_str = "INCONCLUSIVE"
        else:
            status_str = "CONFIRMED" if self.success else "CONTRADICTED"
        return f"[Active Probe {status_str}]: `{self.command}` -> {self.output[:200]}"


def _is_git_work_tree(cwd: str) -> bool:
    """Check safely whether cwd is inside a git working tree."""
    if os.path.isdir(os.path.join(cwd, ".git")):
        return True
    try:
        res = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        return res.returncode == 0 and res.stdout.strip() == "true"
    except Exception:
        return False


def probe_git_clean_status(cwd: str) -> ProbeResult | None:
    """Safely probe if working tree is clean."""
    if not _is_git_work_tree(cwd):
        return None

    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            is_clean = len(proc.stdout.strip()) == 0
            out = "working tree clean" if is_clean else f"uncommitted changes present: {proc.stdout.strip()[:100]}"
            return ProbeResult(
                executed=True,
                success=is_clean,
                claim="working tree clean",
                command=f"git -C {cwd} status --porcelain",
                output=out,
            )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim="working tree clean",
            command=f"git -C {cwd} status --porcelain",
            output=f"Probe inconclusive: {e}",
        )
    return None


def probe_git_branch(cwd: str) -> ProbeResult | None:
    """Safely probe current git branch."""
    if not _is_git_work_tree(cwd):
        return None

    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            branch = proc.stdout.strip()
            return ProbeResult(
                executed=True,
                success=True,
                claim="current git branch",
                command=f"git -C {cwd} branch --show-current",
                output=branch,
            )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim="current git branch",
            command=f"git -C {cwd} branch --show-current",
            output=f"Probe inconclusive: {e}",
        )
    return None


def probe_file_existence(path: str) -> ProbeResult:
    """Safely probe if a file exists on disk."""
    exists = os.path.exists(path)
    return ProbeResult(
        executed=True,
        success=exists,
        claim=f"file exists: {path}",
        command=f"test -e {path}",
        output=f"exists={exists}",
    )


def probe_service_running(label: str) -> ProbeResult | None:
    """Safely probe if a launchd service is alive on macOS."""
    try:
        proc = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.split()
                if parts and parts[-1] == label:
                    pid = parts[0]
                    is_alive = pid.isdigit() and int(pid) > 0
                    return ProbeResult(
                        executed=True,
                        success=is_alive,
                        claim=f"service {label} running",
                        command=f"launchctl list | grep {label}",
                        output=f"entry={line.strip()} (alive={is_alive})",
                    )
            # Not found in caller's launchctl domain (may exist in system domain or another session)
            return None
    except Exception:
        pass
    return None


def probe_port_listening(port: int) -> ProbeResult:
    """Safely probe if a TCP port is listening on the host."""
    if not (1 <= port <= 65535):
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"port {port} listening",
            command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
            output=f"Invalid port number: {port}",
        )
    try:
        proc = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-n", "-P"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            lines = [
                line.strip()
                for line in proc.stdout.splitlines()
                if "LISTEN" in line or ("TCP" in line and not line.startswith("COMMAND"))
            ]
            if lines:
                return ProbeResult(
                    executed=True,
                    success=True,
                    claim=f"port {port} listening",
                    command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
                    output=lines[0],
                )
        if proc.returncode == 1 and not proc.stderr.strip():
            return ProbeResult(
                executed=True,
                success=False,
                claim=f"port {port} listening",
                command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
                output=f"no process listening on port {port}",
            )
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"port {port} listening",
            command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
            output=f"Probe inconclusive: lsof exit code {proc.returncode} {proc.stderr.strip()}".strip(),
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"port {port} listening",
            command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
            output=f"Probe inconclusive: {e}",
        )


def probe_process_running(pid: int) -> ProbeResult:
    """Safely probe if a process with given PID is running on the host."""
    if pid <= 0 or pid > 99999:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"process PID {pid} running",
            command=f"ps -p {pid} -o pid=,user=,comm=",
            output=f"Invalid PID: {pid}",
        )
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,user=,comm="],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return ProbeResult(
                executed=True,
                success=True,
                claim=f"process PID {pid} running",
                command=f"ps -p {pid} -o pid=,user=,comm=",
                output=proc.stdout.strip(),
            )
        if proc.returncode != 0 and proc.stderr.strip():
            return ProbeResult(
                executed=False,
                success=False,
                claim=f"process PID {pid} running",
                command=f"ps -p {pid} -o pid=,user=,comm=",
                output=f"Probe inconclusive: {proc.stderr.strip()}",
            )
        return ProbeResult(
            executed=True,
            success=False,
            claim=f"process PID {pid} running",
            command=f"ps -p {pid} -o pid=,user=,comm=",
            output=f"PID {pid} not found",
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"process PID {pid} running",
            command=f"ps -p {pid} -o pid=,user=,comm=",
            output=f"Probe inconclusive: {e}",
        )


def probe_git_commit(commit_ref: str, cwd: str) -> ProbeResult | None:
    """Safely probe if a git commit hash exists in the repository."""
    if not _is_git_work_tree(cwd):
        return None

    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "log", "-1", "--format=%h %s", commit_ref],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return ProbeResult(
                executed=True,
                success=True,
                claim=f"git commit {commit_ref} exists",
                command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
                output=proc.stdout.strip(),
            )
        return ProbeResult(
            executed=True,
            success=False,
            claim=f"git commit {commit_ref} exists",
            command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
            output=f"commit {commit_ref} not found",
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git commit {commit_ref} exists",
            command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
            output=f"Probe inconclusive: {e}",
        )


def run_active_probes_for_draft(draft: str, cwd: str) -> list[ProbeResult]:
    """Inspect draft for verifiable assertions and run safe read-only probes."""
    probes: list[ProbeResult] = []

    # 1. Probe git clean status if asserted
    if re.search(r"(?i)\b(working tree (is )?clean|git status clean|clean working tree|git tree (is )?clean|no uncommitted changes)\b", draft):
        res = probe_git_clean_status(cwd)
        if res:
            probes.append(res)

    # 2. Probe service running if asserted
    service_labels: set[str] = set()
    if re.search(r"(?i)\b(ai\.hermes\.gateway|gateway is running|gateway restarted)\b", draft):
        service_labels.add("ai.hermes.gateway")

    # Match domain-style launchd service/daemon labels (e.g. com.alejandro.hermes-chrome, ai.hermes.gateway)
    found_labels = re.findall(
        r"(?i)\b(?:service|daemon|LaunchAgent|LaunchDaemon)\s+[`'\"]?([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_.-]+)+)[`'\"]?",
        draft,
    )
    found_labels += re.findall(
        r"(?i)\b([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_.-]+)+)(?:\s+(?:service|daemon))?\s+(?:is\s+)?(?:running|alive|active|restarted)\b",
        draft,
    )
    ignored_label_extensions = {
        "com", "org", "net", "io", "dev", "app", "js", "py", "json", "yml", "yaml", "md", "txt", "html", "sh", "ts", "log", "conf", "cfg"
    }
    for lbl in found_labels:
        lbl = lbl.rstrip(".")
        parts = lbl.lower().split(".")
        if len(parts) >= 2 and parts[-1] not in ignored_label_extensions:
            service_labels.add(lbl)

    for lbl in service_labels:
        res = probe_service_running(lbl)
        if res:
            probes.append(res)

    # 3. Probe git branch if claimed with explicit branch wording
    branch_m = re.search(
        r"(?i)\b(?:switched to branch|checked out branch|on feature branch|on branch)\s+(?:branch\s+)?([a-zA-Z0-9_\-\.\/]+)\b",
        draft,
    )
    if branch_m:
        claimed_branch = branch_m.group(1).strip()
        stop_words = {"the", "a", "an", "our", "this", "that", "clean", "dirty", "and", "or", "to", "is", "for", "with", "all", "tests", "parity", "flags"}
        if claimed_branch.lower() not in stop_words:
            if re.search(r"[/\-_0-9]", claimed_branch) or claimed_branch.lower() in {"main", "master", "dev", "develop"}:
                res = probe_git_branch(cwd)
                if res and res.executed:
                    res.success = (res.output == claimed_branch)
                    res.claim = f"on branch {claimed_branch}"
                    probes.append(res)

    # 4. Probe file existence if explicit file creation or existence is asserted with a path-like token
    ext_pattern = r"(?:py|json|ya?ml|toml|md|txt|ts|js|sh|html|css|rs|go|log|lock|conf|cfg|ini|sql)"
    post_verb_matches = re.findall(
        rf"(?i)(?<!not )(?<!never )(?<!n't )\b(?:created|written\s+to|written\s+into|saved|added)\s+(?:in|to|at)?\s*[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?",
        draft,
    )
    pre_verb_matches = re.findall(
        rf"(?i)(?<!not )(?<!never )(?<!n't )[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?\s+exists?\b",
        draft,
    )
    ignored_basenames = {"node.js", "react.js", "vue.js", "next.js", "deno.js"}
    for p in dict.fromkeys(post_verb_matches + pre_verb_matches):
        if os.path.basename(p).lower() in ignored_basenames:
            continue
        expanded_path = os.path.expanduser(p)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.abspath(os.path.join(cwd, expanded_path))
        probes.append(probe_file_existence(expanded_path))

    # 5. Probe TCP port listening if explicitly asserted
    port_matches = re.findall(
        r"(?i)\b(?:listening\s+(?:on|at)\s+(?:port\s+)?(?:(?:\d{1,3}(?:\.\d{1,3}){3}|localhost):)?(\d{2,5})(?!\.\d)|bound\s+to\s+(?:port\s+)?(?:(?:\d{1,3}(?:\.\d{1,3}){3}|localhost):)?(\d{2,5})(?!\.\d)|port\s+(\d{2,5})\s+(?:is\s+)?(?:listening|open))\b",
        draft,
    )
    seen_ports: set[int] = set()
    for match_tuple in port_matches:
        for p_str in match_tuple:
            if p_str and p_str.isdigit():
                port_num = int(p_str)
                if 1 <= port_num <= 65535 and port_num not in seen_ports:
                    seen_ports.add(port_num)
                    probes.append(probe_port_listening(port_num))

    # 6. Probe process PID if explicitly asserted
    pid_matches = re.findall(
        r"(?i)(?:\brunning\s+as\s+.*?\bPID\s+(\d{1,6})|\brunning(?:\s+\w+)?\s*\(\s*PID\s+(\d{1,6})\s*\)|\bPID\s+(\d{1,6})\s+(?:is\s+)?running|\bprocess\s+(\d{1,6})\s+(?:is\s+)?running)",
        draft,
    )
    seen_pids: set[int] = set()
    for match_tuple in pid_matches:
        for pid_str in match_tuple:
            if pid_str and pid_str.isdigit():
                pid_num = int(pid_str)
                if 1 <= pid_num <= 99999 and pid_num not in seen_pids:
                    seen_pids.add(pid_num)
                    probes.append(probe_process_running(pid_num))

    # 7. Probe git commit SHA if explicitly asserted
    commit_matches = re.findall(
        r"(?i)\b(?:committed\s+(?:in|as|with\s+(?:hash|sha|commit)?)\s+|commit\s+|commit\s+hash\s+|on\s+commit\s+)([0-9a-f]{7,40})\b",
        draft,
    )
    seen_commits: set[str] = set()
    for sha in commit_matches:
        sha_lower = sha.lower()
        if sha_lower not in seen_commits:
            seen_commits.add(sha_lower)
            res = probe_git_commit(sha_lower, cwd)
            if res:
                probes.append(res)

    return probes
