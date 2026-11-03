import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import gateway_live as gl


class TestGatewayLive(unittest.TestCase):
    def test_parse_status_pid(self):
        self.assertEqual(gl.parse_status_pid("✓ Gateway is supervised by launchd (PID 19375)"), 19375)
        self.assertIsNone(gl.parse_status_pid("no pid here"))

    def test_classify(self):
        self.assertEqual(gl.classify(200.0, 100.0), "live")
        self.assertEqual(gl.classify(100.0, 200.0), "stale")
        self.assertEqual(gl.classify(200.0, 200.0), "live")
        self.assertEqual(gl.classify(None, 100.0), "unknown")
        self.assertEqual(gl.classify(100.0, None), "unknown")

    def test_process_start_epoch_reads_own_pid(self):
        started = gl.process_start_epoch(os.getpid())
        self.assertIsNotNone(started)
        self.assertLess(abs(time.time() - started), 3600)

    def test_newest_commit_epoch_from_temp_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-q", "--allow-empty", "-m", "first"],
                check=True,
            )
            epoch = gl.newest_commit_epoch(repo)
            self.assertIsNotNone(epoch)
            self.assertLess(abs(time.time() - epoch), 120)

    def test_newest_commit_epoch_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(gl.newest_commit_epoch(Path(tmp)))

    def test_restart_worker_signals_then_verifies(self):
        pids = [111, 111, 222]
        signals = []
        with mock.patch.object(gl, "gateway_pid", lambda: pids.pop(0) if pids else 222), \
             mock.patch.object(gl.os, "kill", lambda pid, sig: signals.append((pid, sig))), \
             mock.patch.object(gl.time, "sleep", lambda _s: None), \
             mock.patch.object(gl.time, "time", lambda: 0.0), \
             mock.patch.object(gl, "check", lambda _repo: 0), \
             mock.patch.object(gl, "_log", lambda _msg: None), \
             mock.patch.object(gl, "DRAIN_GRACE_SECONDS", 999):
            rc = gl._restart_worker(Path("."), 0)
        self.assertEqual(rc, 0)
        self.assertEqual(signals, [(111, gl.signal.SIGUSR1)])

    def test_restart_worker_without_pid_does_not_signal(self):
        signals = []
        with mock.patch.object(gl, "gateway_pid", lambda: None), \
             mock.patch.object(gl.os, "kill", lambda pid, sig: signals.append(pid)), \
             mock.patch.object(gl.time, "sleep", lambda _s: None), \
             mock.patch.object(gl, "_log", lambda _msg: None):
            rc = gl._restart_worker(Path("."), 0)
        self.assertEqual(rc, 4)
        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
