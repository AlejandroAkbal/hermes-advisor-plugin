"""Active Autonomous Probe Engine for Advisor Plugin.

Provides read-only, non-mutating active probes to verify common system and repo assertions
without requiring an extra agent turn:
- Repo clean state (git status --porcelain)
- Current git branch (git branch --show-current)
- Git commit verification (git log -1)
- Git tag verification (git tag -l)
- Git remote verification (git remote get-url)
- File existence / non-empty checks (test -e)
- Path / file absence / deletion checks (test ! -e)
- Directory existence (test -d)
- Environment variable presence (test -n "$VAR")
- Service presence (launchctl list)
- Process PID presence & zombie rejection (ps -p)
- TCP port listening state (lsof -iTCP -sTCP:LISTEN)
- Command / CLI availability in PATH (which / shutil.which)

All probes are bounded by strict timeouts (<= 2.0s) and executed in a read-only manner.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess

PROBE_TIMEOUT_SECONDS = 2.0
PID_LIMIT = 4_194_304  # Linux pid_max ceiling; pid_max itself is not allocated


class ProbeResult:
    """Encapsulates the outcome of a read-only active probe."""

    def __init__(self, executed: bool, success: bool, claim: str, command: str, output: str) -> None:
        """Initialize probe result with execution outcome and output details."""
        self.executed = executed
        self.success = success
        self.claim = claim
        self.command = command
        self.output = output.strip()

    def format_evidence(self) -> str:
        """Format probe outcome into an evidence line for the auditor."""
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
            check=False,
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
            check=False,
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
            check=False,
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


def probe_file_absence(path: str) -> ProbeResult:
    """Safely probe if a file or path has been deleted or does not exist."""
    clean_path = os.path.expanduser(path.strip())
    try:
        exists = os.path.exists(clean_path)
        is_absent = not exists
        out = "absent=True (path does not exist)" if is_absent else f"absent=False (path still exists: {clean_path})"
        return ProbeResult(
            executed=True,
            success=is_absent,
            claim=f"path absent: {clean_path}",
            command=f"test ! -e {clean_path}",
            output=out,
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"path absent: {clean_path}",
            command=f"test ! -e {clean_path}",
            output=f"Probe inconclusive: {e}",
        )


def probe_directory_existence(path: str) -> ProbeResult:
    """Safely probe if a directory exists on disk."""
    clean_path = os.path.expanduser(path.strip())
    try:
        is_dir = os.path.isdir(clean_path)
        return ProbeResult(
            executed=True,
            success=is_dir,
            claim=f"directory exists: {clean_path}",
            command=f"test -d {clean_path}",
            output=f"isdir={is_dir}",
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"directory exists: {clean_path}",
            command=f"test -d {clean_path}",
            output=f"Probe inconclusive: {e}",
        )


def probe_service_running(label: str) -> ProbeResult | None:
    """Safely probe if a launchd service is alive on macOS."""
    try:
        proc = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
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
            check=False,
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
            # lsof hides sockets of processes owned by other users; cross-check with a connect.
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return ProbeResult(
                        executed=True,
                        success=True,
                        claim=f"port {port} listening",
                        command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
                        output=f"127.0.0.1:{port} accepts connections (listener not visible to lsof)",
                    )
            except ConnectionRefusedError:
                return ProbeResult(
                    executed=False,
                    success=False,
                    claim=f"port {port} listening",
                    command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
                    output=f"Probe inconclusive: no visible listener and 127.0.0.1:{port} refused connection",
                )
            except OSError as e:
                return ProbeResult(
                    executed=False,
                    success=False,
                    claim=f"port {port} listening",
                    command=f"lsof -iTCP:{port} -sTCP:LISTEN -n -P",
                    output=f"Probe inconclusive: lsof found no visible listener; {e}",
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
    """Safely probe if a process with given PID is running and not a zombie on the host."""
    if pid <= 0 or pid >= PID_LIMIT:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"process PID {pid} running",
            command=f"ps -p {pid} -o pid=,stat=,user=,comm=",
            output=f"Invalid PID: {pid}",
        )
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,stat=,user=,comm="],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = proc.stdout.strip().splitlines()
            data_line = lines[0].strip()
            fields = data_line.split()
            # fields: [pid, stat, user, comm...]
            if len(fields) >= 2 and (fields[1].startswith("Z") or "<defunct>" in data_line):
                return ProbeResult(
                    executed=True,
                    success=False,
                    claim=f"process PID {pid} running",
                    command=f"ps -p {pid} -o pid=,stat=,user=,comm=",
                    output=f"PID {pid} is a zombie process: {data_line}",
                )
            return ProbeResult(
                executed=True,
                success=True,
                claim=f"process PID {pid} running",
                command=f"ps -p {pid} -o pid=,stat=,user=,comm=",
                output=data_line,
            )
        if proc.returncode != 0 and proc.stderr.strip():
            return ProbeResult(
                executed=False,
                success=False,
                claim=f"process PID {pid} running",
                command=f"ps -p {pid} -o pid=,stat=,user=,comm=",
                output=f"Probe inconclusive: {proc.stderr.strip()}",
            )
        return ProbeResult(
            executed=True,
            success=False,
            claim=f"process PID {pid} running",
            command=f"ps -p {pid} -o pid=,stat=,user=,comm=",
            output=f"PID {pid} not found",
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"process PID {pid} running",
            command=f"ps -p {pid} -o pid=,stat=,user=,comm=",
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
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return ProbeResult(
                executed=True,
                success=True,
                claim=f"git commit {commit_ref} exists",
                command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
                output=proc.stdout.strip(),
            )
        stderr = proc.stderr.strip().lower()
        if proc.returncode != 0 and ("unknown revision" in stderr or "bad object" in stderr or not stderr):
            return ProbeResult(
                executed=True,
                success=False,
                claim=f"git commit {commit_ref} exists",
                command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
                output=f"commit {commit_ref} not found",
            )
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git commit {commit_ref} exists",
            command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
            output=f"Probe inconclusive: git exit code {proc.returncode} {proc.stderr.strip()}".strip(),
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git commit {commit_ref} exists",
            command=f"git -C {cwd} log -1 --format='%h %s' {commit_ref}",
            output=f"Probe inconclusive: {e}",
        )


def probe_git_tag(tag_name: str, cwd: str) -> ProbeResult | None:
    """Safely probe if a git tag exists in the repository."""
    if not _is_git_work_tree(cwd):
        return None
    clean_tag = tag_name.strip()
    if (
        not re.fullmatch(r"[a-zA-Z0-9_\-\.\/\+]+", clean_tag)
        or len(clean_tag) > 100
        or clean_tag.startswith("-")
    ):
        return None
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "tag", "-l", clean_tag],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode == 0:
            found_tags = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            is_found = clean_tag in found_tags
            output_str = f"tag {clean_tag} exists" if is_found else f"tag {clean_tag} not found"
            return ProbeResult(
                executed=True,
                success=is_found,
                claim=f"git tag {clean_tag} exists",
                command=f"git -C {cwd} tag -l {clean_tag}",
                output=output_str,
            )
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git tag {clean_tag} exists",
            command=f"git -C {cwd} tag -l {clean_tag}",
            output=f"Probe inconclusive: git exit code {proc.returncode} {proc.stderr.strip()}".strip(),
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git tag {clean_tag} exists",
            command=f"git -C {cwd} tag -l {clean_tag}",
            output=f"Probe inconclusive: {e}",
        )


def _redact_url_credentials(url: str) -> str:
    """Redact embedded credentials from git URLs before exposing in evidence."""
    return re.sub(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s]*@", r"\1***@", url)


def probe_git_remote(remote_name: str, cwd: str) -> ProbeResult | None:
    """Safely probe if a git remote is configured and retrieve its URL."""
    if not _is_git_work_tree(cwd):
        return None
    clean_remote = remote_name.strip()
    if (
        not re.fullmatch(r"[a-zA-Z0-9_\-\.]+", clean_remote)
        or len(clean_remote) > 64
        or clean_remote.startswith("-")
    ):
        return None
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "remote", "get-url", clean_remote],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            url = _redact_url_credentials(proc.stdout.strip())
            return ProbeResult(
                executed=True,
                success=True,
                claim=f"git remote {clean_remote} exists",
                command=f"git -C {cwd} remote get-url {clean_remote}",
                output=url,
            )
        stderr = proc.stderr.strip().lower()
        if proc.returncode != 0 and ("no such remote" in stderr or not stderr):
            return ProbeResult(
                executed=True,
                success=False,
                claim=f"git remote {clean_remote} exists",
                command=f"git -C {cwd} remote get-url {clean_remote}",
                output=f"remote {clean_remote} not found",
            )
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git remote {clean_remote} exists",
            command=f"git -C {cwd} remote get-url {clean_remote}",
            output=f"Probe inconclusive: git exit code {proc.returncode} {proc.stderr.strip()}".strip(),
        )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"git remote {clean_remote} exists",
            command=f"git -C {cwd} remote get-url {clean_remote}",
            output=f"Probe inconclusive: {e}",
        )


def probe_command_available(command_name: str) -> ProbeResult:
    """Safely probe if an executable binary or CLI command is installed and available in PATH."""
    clean_name = command_name.strip()
    if not re.fullmatch(r"[a-zA-Z0-9_\-\.]+", clean_name) or len(clean_name) > 64:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"command {command_name} available",
            command=f"which {command_name}",
            output=f"Invalid command name: {command_name}",
        )
    resolved_path = shutil.which(clean_name)
    if resolved_path:
        return ProbeResult(
            executed=True,
            success=True,
            claim=f"command {clean_name} available",
            command=f"which {clean_name}",
            output=f"{clean_name} available at {resolved_path}",
        )
    return ProbeResult(
        executed=True,
        success=False,
        claim=f"command {clean_name} available",
        command=f"which {clean_name}",
        output=f"{clean_name} not found in PATH",
    )


def probe_env_var(var_name: str) -> ProbeResult:
    """Safely probe if an environment variable is set."""
    clean_name = var_name.strip().lstrip("$")
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", clean_name) or len(clean_name) > 128:
        return ProbeResult(
            executed=False,
            success=False,
            claim=f"environment variable {var_name} is set",
            command=f"test -n \"${var_name}\"",
            output=f"Invalid environment variable name: {var_name}",
        )
    is_set = clean_name in os.environ
    return ProbeResult(
        executed=True,
        success=is_set,
        claim=f"environment variable {clean_name} is set",
        command=f"test -n \"${clean_name}\"",
        output=f"{clean_name} is set" if is_set else f"{clean_name} is not set",
    )


def _is_negated_claim(text: str, match_start: int) -> bool:
    """Check if the text leading up to a match contains a negation."""
    window = text[max(0, match_start - 60):match_start]
    return bool(re.search(r"(?i)(?:\bnot|\bnever|n't|\bno)\s+(?:\w+\s+){0,4}$", window))


def _extract_command_names(draft: str) -> list[str]:
    """Safely extract claimed executable/CLI names from draft, filtering negations."""
    patterns = [
        re.compile(r"(?i)\b(?:CLI|binary|command|tool|executable)\s+[`'\"]?([a-zA-Z0-9_\-]+)[`'\"]?\s+(?:is\s+)?(?:installed|available|present)\b"),
        re.compile(r"(?i)\b[`'\"]?([a-zA-Z0-9_\-]+)[`'\"]?\s+(?:CLI|binary|tool)\s+(?:is\s+)?(?:installed|available|present)\b"),
        re.compile(r"(?i)\binstalled\s+[`'\"]?([a-zA-Z0-9_\-]+)[`'\"]?\s+(?:CLI|binary|tool)\b"),
    ]
    ignored_cmds = {
        "the", "a", "an", "this", "that", "new", "our", "clean", "all", "each", "and", "or",
        "not", "no", "to", "is", "for", "with", "any", "some", "it", "its", "which", "what"
    }
    cmds: list[str] = []
    for pat in patterns:
        for m in pat.finditer(draft):
            if _is_negated_claim(draft, m.start()):
                continue
            cmd = m.group(1).strip()
            if cmd.lower() not in ignored_cmds:
                cmds.append(cmd)
    return cmds


def _extract_tag_names(draft: str) -> list[str]:
    """Safely extract claimed git tag names from draft, handling options, combined flags, and message arguments."""
    pattern = re.compile(
        r'(?i)\b(?:git\s+tag(?:\s+(?:-(?:[a-zA-Z0-9]*[mF])\s+(?:"[^"]*"|\'[^\']*\'|\S+)|-[a-ce-ln-z0-9]+))*|tagged(?:\s+as)?)\s+[`\'"]?([a-zA-Z0-9_\-\.\/\+]+)[`\'"]?'
    )
    tags: list[str] = []
    for m in pattern.finditer(draft):
        if _is_negated_claim(draft, m.start()):
            continue
        # Exclude deletion invocations such as git tag -d v1.0 or git tag --delete v1.0
        match_prefix = draft[max(0, m.start() - 10):m.end()]
        if re.search(r"(?i)\bgit\s+tag\s+-(?:d|-delete)\b", match_prefix):
            continue
        candidate = m.group(1).strip().rstrip(".,;:!?)'\"")
        if candidate.startswith("-") or candidate.lower() in {"the", "a", "an", "new", "latest", "as"}:
            continue
        tags.append(candidate)
    return tags


FILLER_WORDS = {
    "was", "were", "has", "have", "been", "already", "now", "still", "also", "just", "successfully",
    "exists", "exist", "created", "present", "is", "are", "defined", "set",
}


def _extract_directory_paths(draft: str, cwd: str) -> list[str]:
    """Safely extract claimed directory/folder paths from draft, filtering negations."""
    patterns = [
        re.compile(r"(?i)\b(?:directory|folder)\s+[`'\"]?([/~a-zA-Z0-9_\-./]+)[`'\"]?\s+(?:exists?|created|is present|was created)\b"),
        re.compile(r"(?i)\b(?:created|made)\s+(?:directory|folder)\s+[`'\"]?([/~a-zA-Z0-9_\-./]+)[`'\"]?\b"),
        re.compile(r"(?i)\b(?:in|into|under)\s+(?:directory|folder)\s+[`'\"]?([/~a-zA-Z0-9_\-./]+)[`'\"]?\b"),
    ]
    ignored = {
        "the", "a", "an", "this", "that", "our", "new", "all", "each", "and", "or", "to", "is", "for", "with",
        "structure", "hierarchy", "tree", "layout", "permissions", "size", "contents", "path", "name",
    }
    paths: list[str] = []
    for pat in patterns:
        for m in pat.finditer(draft):
            if _is_negated_claim(draft, m.start()):
                continue
            cand = m.group(1).strip().rstrip(".,;:!?)'\"")
            cand_lower = cand.lower()
            if not cand or cand_lower in ignored or cand_lower in FILLER_WORDS:
                continue
            if "/" in cand or cand.startswith(("~", ".")) or re.fullmatch(r"[a-zA-Z0-9_\-]+", cand):
                expanded = os.path.expanduser(cand)
                if not os.path.isabs(expanded):
                    expanded = os.path.abspath(os.path.join(cwd, expanded))
                paths.append(expanded)
    return paths


def _extract_git_remotes(draft: str) -> list[str]:
    """Safely extract claimed git remote names from draft, filtering negations."""
    patterns = [
        re.compile(r"(?i)\b(?:git\s+remote|remote)\s+[`'\"]?([a-zA-Z0-9_\-\.]+)[`'\"]?\s+(?:exists?|is\s+set|configured|points\s+to)\b"),
        re.compile(r"(?i)\b(?:configured|set|added)\s+(?:git\s+)?remote\s+[`'\"]?([a-zA-Z0-9_\-\.]+)[`'\"]?\b"),
    ]
    ignored_remotes = {"the", "a", "an", "this", "that", "our", "new", "all", "each", "and", "or", "not", "no", "to", "is", "for", "with", "branch", "commit", "tag"}
    remotes: list[str] = []
    for pat in patterns:
        for m in pat.finditer(draft):
            if _is_negated_claim(draft, m.start()):
                continue
            cand = m.group(1).strip().rstrip(".,;:!?)'\"")
            cand_lower = cand.lower()
            if not cand or cand.startswith("-") or cand_lower in ignored_remotes or cand_lower in FILLER_WORDS:
                continue
            remotes.append(cand)
    return remotes


def _extract_env_vars(draft: str) -> list[str]:
    """Safely extract claimed environment variable names from draft, filtering negations."""
    patterns = [
        re.compile(r"(?i)\b(?:environment\s+variable|env\s+var)\s+[`'\"]?\$?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?\s+(?:is\s+)?(?:set|exported|present|defined)\b"),
        re.compile(r"(?i)\b[`'\"]?\$([A-Z_][A-Z0-9_]{1,64})[`'\"]?\s+(?:is\s+)?(?:set|exported|defined|configured)\b"),
    ]
    ignored_vars = {"THE", "A", "AN", "THIS", "THAT", "AND", "OR", "NOT", "NO", "TO", "IS", "FOR", "WITH", "TRUE", "FALSE", "NULL"}
    vars_list: list[str] = []
    for pat in patterns:
        for m in pat.finditer(draft):
            if _is_negated_claim(draft, m.start()):
                continue
            cand = m.group(1).strip()
            if not cand or cand.upper() in ignored_vars or cand.lower() in FILLER_WORDS or len(cand) > 128:
                continue
            vars_list.append(cand)
    return vars_list


def run_active_probes_for_draft(draft: str, cwd: str) -> list[ProbeResult]:
    """Inspect draft for verifiable assertions and run safe read-only probes."""
    # Cap scanned draft length to prevent pathological scanning overhead
    scan_draft = draft[:50000] if draft else ""
    probes: list[ProbeResult] = []

    # 1. Probe git clean status if asserted
    if re.search(r"(?i)\b(working tree (is )?clean|git status clean|clean working tree|git tree (is )?clean|no uncommitted changes)\b", scan_draft):
        res = probe_git_clean_status(cwd)
        if res:
            probes.append(res)

    # 2. Probe service running if asserted
    service_labels: set[str] = set()
    if re.search(r"(?i)\b(ai\.hermes\.gateway|gateway is running|gateway restarted)\b", scan_draft):
        service_labels.add("ai.hermes.gateway")

    # Match domain-style launchd service/daemon labels (e.g. com.alejandro.hermes-chrome, ai.hermes.gateway)
    found_labels = re.findall(
        r"(?i)\b(?:service|daemon|LaunchAgent|LaunchDaemon)\s+[`'\"]?([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)+)[`'\"]?",
        scan_draft,
    )
    found_labels += re.findall(
        r"(?i)\b([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)+\.?)(?:\s+(?:service|daemon))?\s+(?:is\s+)?(?:running|alive|active|restarted)\b",
        scan_draft,
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
        scan_draft,
    )
    if branch_m:
        claimed_branch = branch_m.group(1).strip()
        stop_words = {"the", "a", "an", "our", "this", "that", "clean", "dirty", "and", "or", "to", "is", "for", "with", "all", "tests", "parity", "flags"}
        if claimed_branch.lower() not in stop_words and (re.search(r"[/\-_0-9]", claimed_branch) or claimed_branch.lower() in {"main", "master", "dev", "develop"}):
            res = probe_git_branch(cwd)
            if res and res.executed:
                res.success = (res.output == claimed_branch)
                res.claim = f"on branch {claimed_branch}"
                probes.append(res)

    # 4. Probe file existence if explicit file creation or existence is asserted with a path-like token
    ext_pattern = r"(?:py|json|ya?ml|toml|md|txt|ts|js|sh|html|css|rs|go|log|lock|conf|cfg|ini|sql|bak|disabled|plist)"
    post_verb_matches = re.findall(
        rf"(?i)(?<!not )(?<!never )(?<!n't )\b(?:created|written\s+to|written\s+into|saved|added)\s+(?:in|to|at)?\s*[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?",
        scan_draft,
    )
    pre_verb_matches = re.findall(
        rf"(?i)(?<!not )(?<!never )(?<!n't )[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?\s+exists?\b",
        scan_draft,
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
        scan_draft,
    )
    seen_ports: set[int] = set()
    for match_tuple in port_matches:
        for p_str in match_tuple:
            if p_str and p_str.isdigit():
                port_num = int(p_str)
                if 1 <= port_num <= 65535 and port_num not in seen_ports:
                    seen_ports.add(port_num)
                    probes.append(probe_port_listening(port_num))

    # 6. Probe process PID if explicitly asserted (bounded character gap to prevent regex performance overhead)
    pid_matches = re.findall(
        r"(?i)(?:\brunning\s+as\s+.{1,80}?[\b\s]PID\s+(\d{1,7})|\brunning(?:\s+\w+)?\s*\(\s*PID\s+(\d{1,7})\s*\)|\bPID\s+(\d{1,7})\s+(?:is\s+)?running|\bprocess\s+(\d{1,7})\s+(?:is\s+)?running)",
        scan_draft,
    )
    seen_pids: set[int] = set()
    for match_tuple in pid_matches:
        for pid_str in match_tuple:
            if pid_str and pid_str.isdigit():
                pid_num = int(pid_str)
                if 1 <= pid_num < PID_LIMIT and pid_num not in seen_pids:
                    seen_pids.add(pid_num)
                    probes.append(probe_process_running(pid_num))

    # 7. Probe git commit SHA if explicitly asserted
    commit_matches = re.findall(
        r"(?i)\b(?:committed\s+(?:in|as|with\s+(?:hash|sha|commit)?)\s+|commit\s+|commit\s+hash\s+|on\s+commit\s+)([0-9a-f]{7,40})\b",
        scan_draft,
    )
    seen_commits: set[str] = set()
    for sha in commit_matches:
        sha_lower = sha.lower()
        if sha_lower not in seen_commits:
            seen_commits.add(sha_lower)
            res = probe_git_commit(sha_lower, cwd)
            if res:
                probes.append(res)

    # 8. Probe command/CLI availability if asserted (bounded by budget)
    MAX_CMD_PROBES = 5
    cmd_count = 0
    seen_cmds: set[str] = set()
    for c_raw in _extract_command_names(scan_draft):
        if cmd_count >= MAX_CMD_PROBES:
            break
        c_lower = c_raw.lower()
        if c_lower not in seen_cmds:
            seen_cmds.add(c_lower)
            probes.append(probe_command_available(c_raw))
            cmd_count += 1

    # 9. Probe git tag if asserted (bounded by budget)
    MAX_TAG_PROBES = 5
    tag_count = 0
    seen_tags: set[str] = set()
    for t_clean in _extract_tag_names(scan_draft):
        if tag_count >= MAX_TAG_PROBES:
            break
        if t_clean not in seen_tags:
            seen_tags.add(t_clean)
            res = probe_git_tag(t_clean, cwd)
            if res:
                probes.append(res)
                tag_count += 1

    # 10. Probe directory existence if asserted (bounded by budget)
    MAX_DIR_PROBES = 5
    dir_count = 0
    seen_dirs: set[str] = set()
    for d_path in _extract_directory_paths(scan_draft, cwd):
        if dir_count >= MAX_DIR_PROBES:
            break
        if d_path not in seen_dirs:
            seen_dirs.add(d_path)
            probes.append(probe_directory_existence(d_path))
            dir_count += 1

    # 11. Probe git remote if asserted (bounded by budget)
    MAX_REMOTE_PROBES = 5
    remote_count = 0
    seen_remotes: set[str] = set()
    for r_name in _extract_git_remotes(scan_draft):
        if remote_count >= MAX_REMOTE_PROBES:
            break
        r_clean = r_name.lower()
        if r_clean not in seen_remotes:
            seen_remotes.add(r_clean)
            res = probe_git_remote(r_name, cwd)
            if res:
                probes.append(res)
                remote_count += 1

    # 12. Probe environment variables if asserted (bounded by budget)
    MAX_ENV_PROBES = 5
    env_count = 0
    seen_envs: set[str] = set()
    for v_name in _extract_env_vars(scan_draft):
        if env_count >= MAX_ENV_PROBES:
            break
        v_clean = v_name.strip()
        if v_clean not in seen_envs:
            seen_envs.add(v_clean)
            probes.append(probe_env_var(v_clean))
            env_count += 1

    # 13. Probe path/file absence if deletion or removal is explicitly asserted (bounded by budget)
    MAX_ABSENCE_PROBES = 5
    absence_patterns = [
        re.compile(
            rf"(?i)(?<!not )(?<!never )(?<!n't )\b(?:deleted|removed|purged|cleaned\s+up|dropped)\s+(?:the\s+)?(?:temporary\s+)?(?:file|script|artifact|directory|folder)?\s*[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?"
        ),
        re.compile(
            rf"(?i)(?<!not )(?<!never )(?<!n't )[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?\s+(?:was\s+|is\s+|has\s+been\s+)?(?:deleted|removed|purged|cleaned\s+up|dropped|gone)\b"
        ),
        re.compile(
            rf"(?i)[`'\"]?([/~a-zA-Z0-9_\-./]+(?:\.{ext_pattern}))[`'\"]?\s+(?:does\s+not\s+exist|no\s+longer\s+exists)\b"
        ),
        re.compile(
            r"(?i)(?<!not )(?<!never )(?<!n't )\b(?:deleted|removed|purged|cleaned\s+up)\s+(?:the\s+)?(?:directory|folder)\s+[`'\"]?([/~a-zA-Z0-9_\-./]+)[`'\"]?\b"
        ),
    ]
    absence_count = 0
    seen_absent_paths: set[str] = set()
    for pat in absence_patterns:
        if absence_count >= MAX_ABSENCE_PROBES:
            break
        for m in pat.finditer(scan_draft):
            if absence_count >= MAX_ABSENCE_PROBES:
                break
            if _is_negated_claim(scan_draft, m.start()):
                continue
            cand = m.group(1).strip().rstrip('.,;:!?)\'"')
            if not cand or os.path.basename(cand).lower() in ignored_basenames:
                continue
            expanded_path = os.path.expanduser(cand)
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.abspath(os.path.join(cwd, expanded_path))
            if expanded_path not in seen_absent_paths:
                seen_absent_paths.add(expanded_path)
                probes.append(probe_file_absence(expanded_path))
                absence_count += 1

    return probes
