#!/usr/bin/env python3
"""Report whether the running gateway has loaded the plugin code on disk, and restart it.

`check` is read-only: it compares the gateway process start time with the newest
commit that touched the plugin directory. `restart` detaches a helper in its own
session (so the gateway's SIGTERM cannot kill it), waits for a quiet window, sends
the gateway's graceful-restart signal, then re-checks.

Exit codes: 0 live, 3 stale, 4 cannot determine, 5 restart failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path.home() / ".hermes" / "advisor" / "gateway_live.log"
GATEWAY_PID_PATH = Path.home() / ".hermes" / "gateway.pid"
GATEWAY_STATE_PATH = Path.home() / ".hermes" / "gateway_state.json"
STATUS_PID_RE = re.compile(r"PID (\d+)")
# How long we allow the gateway to finish its in-flight turn and respawn before
# declaring the restart failed. The drain itself is not launchd-bounded: in-band
# SIGUSR1 restarts bypass ExitTimeOut (gateway/run_bootstrap.py:234).
DRAIN_GRACE_SECONDS = 60


def parse_status_pid(status_output: str) -> int | None:
    match = STATUS_PID_RE.search(status_output or "")
    return int(match.group(1)) if match else None


def process_start_epoch(pid: int) -> float | None:
    """Start time of *pid* as a unix timestamp, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    try:
        # `ps lstart` prints local time; a naive datetime stamps it as such.
        return datetime.strptime(out, "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError:
        return None


def newest_commit_epoch(repo: Path) -> float | None:
    """Committer time of the newest commit touching *repo*, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%ct"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return float(out) if out.isdigit() else None


def classify(pid_start: float | None, commit: float | None) -> str:
    """LIVE when the process started after the newest commit, STALE otherwise."""
    if pid_start is None or commit is None:
        return "unknown"
    return "live" if pid_start >= commit else "stale"


def _command_of(pid: int) -> str:
    try:
        return subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _is_wrapper(pid: int) -> bool:
    """True when *pid* is the launchd osascript wrapper rather than the gateway."""
    return "osascript" in _command_of(pid)


def _child_gateway_pid(parent: int) -> int | None:
    """First python child of *parent*: the wrapper's real gateway process."""
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,ppid=,command="], capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != str(parent):
            continue
        pid, command = int(parts[0]), parts[2]
        if "python" in command and "osascript" not in command:
            return pid
    return None


def gateway_pid() -> int | None:
    """PID of the gateway process itself, never the launchd osascript wrapper.

    Sending SIGUSR1 to the wrapper kills the wrapper; launchd then SIGTERMs the
    real gateway, which skips the drain and interrupts every active turn
    (observed: "Gateway drain timed out after 0.0s with 4 active agent(s)").
    `~/.hermes/gateway.pid` names the process that actually handles the signal;
    fall back to the status banner, resolving down from the wrapper when needed.
    """
    try:
        record = json.loads(GATEWAY_PID_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    pid = record.get("pid")
    if isinstance(pid, int) and record.get("kind") == "hermes-gateway" and process_start_epoch(pid) is not None and not _is_wrapper(pid):
        return pid
    try:
        out = subprocess.run(
            ["hermes", "gateway", "status"], capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pid = parse_status_pid(out)
    if pid is None:
        return None
    return _child_gateway_pid(pid) if _is_wrapper(pid) else pid


def check(repo: Path) -> int:
    pid = gateway_pid()
    if pid is None:
        print("stale: could not determine the gateway PID")
        return 4
    started = process_start_epoch(pid)
    commit = newest_commit_epoch(repo)
    verdict = classify(started, commit)
    if verdict == "unknown" or started is None or commit is None:
        print(f"unknown: pid={pid} start={started} commit={commit}")
        return 4
    when = datetime.fromtimestamp(commit, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"{verdict}: gateway pid {pid} started {datetime.fromtimestamp(started).astimezone().isoformat(timespec='seconds')}; "
          f"newest plugin commit {when}")
    return 0 if verdict == "live" else 3


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def gateway_activity() -> tuple[int, str] | None:
    """(active agent turns, gateway_state) from the gateway's own state file.

    None means the file is unreadable, which the caller treats as "not safe to
    restart": a restart armed against unknown activity is how turns get killed.
    """
    try:
        state = json.loads(GATEWAY_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    active = state.get("active_agents")
    if not isinstance(active, int):
        active = 0
    return active, str(state.get("gateway_state") or "")


def wait_until_idle(max_wait_seconds: int, poll_seconds: int = 10) -> bool:
    """Block until no agent turn is running. False when the wait timed out."""
    deadline = time.time() + max_wait_seconds
    while True:
        activity = gateway_activity()
        if activity is None:
            _log("gateway state unreadable; not restarting while activity is unknown")
            return False
        active, state = activity
        if active == 0 and state != "draining":
            return True
        if time.time() >= deadline:
            _log(f"still {active} active turn(s) after {max_wait_seconds}s; giving up without signalling")
            return False
        _log(f"waiting for idle: {active} active turn(s), state={state or 'unknown'}")
        time.sleep(poll_seconds)


def _restart_worker(repo: Path, wait_seconds: int, signal_only: bool = False,
                    idle: bool = False, max_wait_seconds: int = 0) -> int:
    _log(f"restart scheduled in {wait_seconds}s for {repo}")
    time.sleep(wait_seconds)
    if idle and not wait_until_idle(max_wait_seconds):
        return 6
    pid = gateway_pid()
    if pid is None:
        _log("no gateway pid; nothing to signal")
        return 4
    if _is_wrapper(pid):
        _log(f"refusing to signal launchd wrapper pid {pid}: killing it makes launchd SIGTERM the gateway and skip the drain")
        return 5
    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError as exc:
        _log(f"signal failed: {exc}")
        return 5
    _log(f"SIGUSR1 sent to {pid}")
    if signal_only:
        return 0
    deadline = time.time() + DRAIN_GRACE_SECONDS + 180
    while time.time() < deadline:
        time.sleep(5)
        new_pid = gateway_pid()
        if new_pid and new_pid != pid:
            _log(f"gateway respawned pid {pid} -> {new_pid}")
            return check(repo)
    _log("gateway did not respawn in time")
    return 5


def restart(repo: Path, wait_seconds: int, detach: bool, signal_only: bool = False,
            idle: bool = True, max_wait_seconds: int = 21600) -> int:
    if not detach:
        return _restart_worker(repo, wait_seconds, signal_only, idle, max_wait_seconds)
    if os.fork() != 0:
        print(f"restart detached; log: {LOG_PATH}")
        return 0
    os.setsid()
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    os._exit(_restart_worker(repo, wait_seconds, signal_only, idle, max_wait_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "restart"])
    parser.add_argument("--repo", type=Path, default=PLUGIN_ROOT)
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--no-detach", action="store_true")
    parser.add_argument("--signal-only", action="store_true", help="send the signal and exit without waiting")
    parser.add_argument("--no-idle", dest="idle", action="store_false",
                        help="signal even while turns are running (interrupts them; not recommended)")
    parser.add_argument("--max-wait-seconds", type=int, default=21600,
                        help="how long to wait for an idle gateway before giving up (default 6h)")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.repo)
    return restart(args.repo, args.wait_seconds, detach=not args.no_detach,
                   signal_only=args.signal_only, idle=args.idle,
                   max_wait_seconds=args.max_wait_seconds)


if __name__ == "__main__":
    sys.exit(main())
