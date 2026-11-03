import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
