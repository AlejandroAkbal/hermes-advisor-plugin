"""Unit tests for Hermes Advisor runtime helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure Hermes agent & plugins are importable
sys.path.insert(0, "/Users/alejandro/.hermes/hermes-agent")
import hermes_bootstrap
from hermes_cli.plugins import get_plugin_manager

mgr = get_plugin_manager()
mgr.discover_and_load()
runtime_mod = sys.modules["hermes_plugins.advisor.runtime"]

_is_substantive_turn = runtime_mod._is_substantive_turn
_find_guidelines = runtime_mod._find_guidelines


class TestAdvisorRuntime(unittest.TestCase):
    def test_activity_gate_casual_turn(self):
        history = [
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You are welcome!"},
        ]
        self.assertFalse(_is_substantive_turn("thanks", "You are welcome!", history))

    def test_activity_gate_casual_turn_after_tool_heavy_turn(self):
        # Multi-turn history: previous turn had tool calls, current turn is casual
        history = [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}]},
            {"role": "tool", "content": "file contents here..."},
            {"role": "assistant", "content": "Here is the file summary."},
            {"role": "user", "content": "thanks!"},
            {"role": "assistant", "content": "Glad to help!"},
        ]
        self.assertFalse(_is_substantive_turn("thanks!", "Glad to help!", history))

    def test_activity_gate_no_user_in_history(self):
        # Edge case: history contains no user message (e.g. only prior tool/assistant messages)
        history = [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}]},
            {"role": "tool", "content": "file contents here..."},
        ]
        # Should not falsely inherit prior tool calls
        self.assertFalse(_is_substantive_turn("ok", "Got it.", history))

    def test_activity_gate_tool_executing_turn(self):
        history = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "terminal"}}]},
            {"role": "tool", "content": "tests passed"},
            {"role": "assistant", "content": "Tests ran successfully."},
        ]
        self.assertTrue(_is_substantive_turn("run tests", "Tests ran successfully.", history))

    def test_activity_gate_code_response_without_tools(self):
        code_text = "Here is the snippet:\n```python\nprint('hello')\n```"
        history = [
            {"role": "user", "content": "show code"},
            {"role": "assistant", "content": code_text},
        ]
        self.assertTrue(_is_substantive_turn("show code", code_text, history))

    def test_activity_gate_long_technical_explanation(self):
        long_text = "A" * 250
        history = [
            {"role": "user", "content": "explain architecture"},
            {"role": "assistant", "content": long_text},
        ]
        self.assertTrue(_is_substantive_turn("explain architecture", long_text, history))

    def test_find_guidelines_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "AGENTS.md").write_text("Root guideline text")

            # Root level
            self.assertEqual(_find_guidelines(str(root)), "Root guideline text")

            # Subdirectory traversal
            subdir = root / "src" / "deep" / "nested"
            subdir.mkdir(parents=True)
            self.assertEqual(_find_guidelines(str(subdir)), "Root guideline text")

            # 4000-char truncation
            (root / "AGENTS.md").write_text("X" * 5000)
            self.assertEqual(len(_find_guidelines(str(subdir))), 4000)

            # Stopped at git boundary
            empty_dir = root / "empty"
            empty_dir.mkdir()
            (empty_dir / ".git").mkdir()
            self.assertEqual(_find_guidelines(str(empty_dir)), "")

    def test_head_tail_clipping(self):
        long_text = "HEADER_" + ("M" * 5000) + "_FOOTER"
        resp_text = long_text[:1500] + "\n... [truncated] ...\n" + long_text[-2500:]
        self.assertTrue(resp_text.startswith("HEADER_"))
        self.assertTrue(resp_text.endswith("_FOOTER"))
        self.assertIn("... [truncated] ...", resp_text)
        self.assertLess(len(resp_text), len(long_text))


if __name__ == "__main__":
    unittest.main()
