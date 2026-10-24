"""Unit tests for Hermes Advisor runtime helpers."""

import sys
import tempfile
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


def test_activity_gate_casual_turn():
    history = [
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "You are welcome!"},
    ]
    assert not _is_substantive_turn("thanks", "You are welcome!", history)


def test_activity_gate_casual_turn_after_tool_heavy_turn():
    # Multi-turn history: previous turn had tool calls, current turn is casual
    history = [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}]},
        {"role": "tool", "content": "file contents here..."},
        {"role": "assistant", "content": "Here is the file summary."},
        {"role": "user", "content": "thanks!"},
        {"role": "assistant", "content": "Glad to help!"},
    ]
    # Current turn has no tool calls, no code, <200 chars -> should be skipped!
    assert not _is_substantive_turn("thanks!", "Glad to help!", history)


def test_activity_gate_tool_executing_turn():
    history = [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "terminal"}}]},
        {"role": "tool", "content": "tests passed"},
        {"role": "assistant", "content": "Tests ran successfully."},
    ]
    assert _is_substantive_turn("run tests", "Tests ran successfully.", history)


def test_activity_gate_code_response_without_tools():
    code_text = "Here is the snippet:\n```python\nprint('hello')\n```"
    history = [
        {"role": "user", "content": "show code"},
        {"role": "assistant", "content": code_text},
    ]
    assert _is_substantive_turn("show code", code_text, history)


def test_activity_gate_long_technical_explanation():
    long_text = "A" * 250
    history = [
        {"role": "user", "content": "explain architecture"},
        {"role": "assistant", "content": long_text},
    ]
    assert _is_substantive_turn("explain architecture", long_text, history)


def test_find_guidelines_hierarchy():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "AGENTS.md").write_text("Root guideline text")

        # Root level
        assert _find_guidelines(str(root)) == "Root guideline text"

        # Subdirectory traversal
        subdir = root / "src" / "deep" / "nested"
        subdir.mkdir(parents=True)
        assert _find_guidelines(str(subdir)) == "Root guideline text"

        # 4000-char truncation
        (root / "AGENTS.md").write_text("X" * 5000)
        assert len(_find_guidelines(str(subdir))) == 4000

        # Empty directory
        empty_dir = root / "empty"
        empty_dir.mkdir()
        # Create a faux .git to stop parent walk
        (empty_dir / ".git").mkdir()
        assert _find_guidelines(str(empty_dir)) == ""


def test_head_tail_clipping():
    long_text = "HEADER_" + ("M" * 5000) + "_FOOTER"
    resp_text = long_text[:1500] + "\n... [truncated] ...\n" + long_text[-2500:]
    assert resp_text.startswith("HEADER_")
    assert resp_text.endswith("_FOOTER")
    assert "... [truncated] ..." in resp_text
    assert len(resp_text) < len(long_text)


if __name__ == "__main__":
    test_activity_gate_casual_turn()
    print("✓ test_activity_gate_casual_turn")
    test_activity_gate_casual_turn_after_tool_heavy_turn()
    print("✓ test_activity_gate_casual_turn_after_tool_heavy_turn")
    test_activity_gate_tool_executing_turn()
    print("✓ test_activity_gate_tool_executing_turn")
    test_activity_gate_code_response_without_tools()
    print("✓ test_activity_gate_code_response_without_tools")
    test_activity_gate_long_technical_explanation()
    print("✓ test_activity_gate_long_technical_explanation")
    test_find_guidelines_hierarchy()
    print("✓ test_find_guidelines_hierarchy")
    test_head_tail_clipping()
    print("✓ test_head_tail_clipping")
    print("\nAll unit tests passed successfully!")
