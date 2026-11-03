#!/usr/bin/env python3
"""Report whether the running gateway has loaded the plugin code on disk, and restart it.

`check` is read-only: it compares the gateway process start time with the newest
commit that touched the plugin directory. `restart` detaches a helper in its own
session (so the gateway's SIGTERM cannot kill it), waits for a quiet window,
kickstarts the launchd job, then re-checks.

Exit codes: 0 live, 3 stale, 4 cannot determine, 5 restart failed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path.home() / ".hermes" / "advisor" / "gateway_live.log"
LAUNCHD_LABEL = "ai.hermes.gateway"
STATUS_PID_RE = re.compile(r"PID (\d+)")
# launchd clamps ExitTimeOut to 60s per user domain, so a turn still running when
# SIGTERM lands gets at most that long to finish.
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


def gateway_pid() -> int | None:
    try:
        out = subprocess.run(
            ["hermes", "gateway", "status"], capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_status_pid(out)


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


def _restart_worker(repo: Path, wait_seconds: int) -> int:
    _log(f"restart scheduled in {wait_seconds}s for {repo}")
    time.sleep(wait_seconds)
    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{LAUNCHD_LABEL}"],
        capture_output=True, text=True,
    )
    _log(f"kickstart exit={result.returncode} {result.stderr.strip()[:200]}")
    deadline = time.time() + DRAIN_GRACE_SECONDS + 120
    old_pid = gateway_pid()
    while time.time() < deadline:
        time.sleep(5)
        new_pid = gateway_pid()
        if new_pid and new_pid != old_pid:
            _log(f"gateway respawned pid {old_pid} -> {new_pid}")
            return check(repo)
    _log("gateway did not respawn in time")
    return 5


def restart(repo: Path, wait_seconds: int, detach: bool) -> int:
    if not detach:
        return _restart_worker(repo, wait_seconds)
    if os.fork() != 0:
        print(f"restart detached; log: {LOG_PATH}")
        return 0
    os.setsid()
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    os._exit(_restart_worker(repo, wait_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "restart"])
    parser.add_argument("--repo", type=Path, default=PLUGIN_ROOT)
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--no-detach", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.repo)
    return restart(args.repo, args.wait_seconds, detach=not args.no_detach)


if __name__ == "__main__":
    sys.exit(main())
