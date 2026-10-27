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
_get_git_diff = getattr(runtime_mod, "_get_git_diff", None)


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

    def test_format_history_scopes_tool_calls_to_current_turn(self):
        _format_history = runtime_mod.AdvisorRuntime._format_history
        history = [
            {"role": "user", "content": "previous task"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "old_tool", "arguments": "{}"}}]},
            {"role": "tool", "name": "old_tool", "content": "old result"},
            {"role": "assistant", "content": "old task done"},
            {"role": "user", "content": "current task"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "current_tool", "arguments": "{\"arg\": 1}"}}]},
            {"role": "tool", "name": "current_tool", "content": "current result"},
        ]
        formatted = _format_history(user_message="current task", response="current response", history=history)

        # Prior context should be summarized
        self.assertIn("User Directives & History", formatted)
        self.assertIn("previous task", formatted)

        # Current tool should be in the tool calls section
        self.assertIn("current_tool", formatted)
        self.assertIn("current result", formatted)

        # Old tool calls from previous turn MUST NOT be in the current turn's tool calls
        self.assertNotIn("old_tool", formatted)
        self.assertNotIn("old result", formatted)

    def test_format_history_preserves_earlier_user_authorizations(self):
        _format_history = runtime_mod.AdvisorRuntime._format_history
        # 12 turns of back-and-forth where early on user gives an authorization
        history = [
            {"role": "user", "content": "No, do everything on main. No need to change to another branch."},
            {"role": "assistant", "content": "Understood, will do everything on main."},
        ]
        for i in range(10):
            history.append({"role": "user", "content": f"step {i}"})
            history.append({"role": "assistant", "content": f"completed step {i}"})

        history.append({"role": "user", "content": "is it ready?"})
        history.append({"role": "assistant", "content": "Yes, all done."})

        formatted = _format_history(user_message="is it ready?", response="Yes, all done.", history=history)

        # The authorization from turn 1 MUST be visible to the advisor even after 10+ intermediate turns!
        self.assertIn("No, do everything on main. No need to change to another branch.", formatted)

    def test_collect_directives_budget_bounding(self):
        _collect_directives = runtime_mod._collect_directives
        # Simulated 200-message session with long log pastes
        user_msgs = ["Initial project goal: build a reliable tool."]
        for i in range(150):
            if i == 5:
                user_msgs.append("Important authorization: work on main directly.")
            elif i % 2 == 0:
                user_msgs.append("A" * 1500)  # long pasted log
            else:
                user_msgs.append(f"Command execution {i}")

        collected = _collect_directives(user_msgs)

        # 1. Total characters must be under budget (8,000 chars)
        total_chars = sum(len(d) for d in collected)
        self.assertLessEqual(total_chars, 8000)

        # 2. First message (framing) must be preserved
        self.assertEqual(collected[0], "Initial project goal: build a reliable tool.")

        # 3. Individual snippets must be capped at 600 chars (+ truncation marker)
        for d in collected[1:]:
            self.assertLessEqual(len(d), 650)

    def test_clip_directive_preserves_trailing_instructions(self):
        _clip_directive = runtime_mod._clip_directive
        # Giant pasted log with instruction at the end
        giant_paste = "LOG_TRACE_START " + ("X" * 10000) + " LOG_TRACE_END\nFix this, but stay on main and do not touch auth."
        clipped = _clip_directive(giant_paste, 600)

        # Leading context preserved
        self.assertTrue(clipped.startswith("LOG_TRACE_START"))
        # Trailing instruction preserved
        self.assertIn("stay on main and do not touch auth.", clipped)
        # Total length within budget
        self.assertLessEqual(len(clipped), 650)

    def test_advisor_depth_stateless_counting(self):
        _advisor_depth = runtime_mod._advisor_depth
        prefix = runtime_mod.ADVISOR_INJECTION_PREFIX

        # 1. Normal human turn -> depth 0
        h1 = [
            {"role": "user", "content": "build feature"},
            {"role": "assistant", "content": "built feature"},
        ]
        self.assertEqual(_advisor_depth(h1), 0)

        # 2. One advisor injection -> depth 1
        h2 = h1 + [
            {"role": "user", "content": f"{prefix}\n\n[CONCERN] bug in feature"},
            {"role": "assistant", "content": "fixed bug"},
        ]
        self.assertEqual(_advisor_depth(h2), 1)

        # 3. Second advisor injection -> depth 2
        h3 = h2 + [
            {"role": "user", "content": f"{prefix}\n\n[CONCERN] secondary bug"},
            {"role": "assistant", "content": "fixed secondary bug"},
        ]
        self.assertEqual(_advisor_depth(h3), 2)

        # 4. Human user sends a message -> resets to 0
        h4 = h3 + [
            {"role": "user", "content": "looks great, next task"},
            {"role": "assistant", "content": "working on next task"},
        ]
        self.assertEqual(_advisor_depth(h4), 0)

    def test_nothing_to_flag_produces_no_injections(self):
        from hermes_plugins.advisor.models import AdvisorState
        state = AdvisorState()
        # Clean response must yield 0 advice items
        self.assertEqual(state.parse_response("nothing to flag"), [])
        self.assertEqual(state.parse_response("Nothing to flag."), [])

    def test_pre_verify_nudge_on_issue(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        class MockCtx:
            class MockLLM:
                def complete(self, **kwargs):
                    return MockLLMResult("[CONCERN] Unhandled edge case in auth.")
            llm = MockLLM()

        rt = runtime_mod.AdvisorRuntime(MockCtx())
        # Mock _get_git_diff to return a diff
        orig_diff = runtime_mod._get_git_diff
        try:
            runtime_mod._get_git_diff = lambda cwd, paths: "diff --git a/foo.py b/foo.py\n+def foo(): pass"
            res = rt.on_pre_verify(
                session_id="test_sess",
                changed_paths=["foo.py"],
                attempt=0,
                final_response="All done."
            )
            self.assertIsNotNone(res)
            self.assertEqual(res.get("action"), "continue")
            self.assertIn("Advisor caught issue before finish", res.get("message", ""))
            self.assertIn("Unhandled edge case in auth", res.get("message", ""))
        finally:
            runtime_mod._get_git_diff = orig_diff

    def test_pre_verify_allows_on_nothing_to_flag(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        class MockCtx:
            class MockLLM:
                def complete(self, **kwargs):
                    return MockLLMResult("nothing to flag")
            llm = MockLLM()

        rt = runtime_mod.AdvisorRuntime(MockCtx())
        orig_diff = runtime_mod._get_git_diff
        try:
            runtime_mod._get_git_diff = lambda cwd, paths: "diff --git a/foo.py b/foo.py\n+def foo(): pass"
            res = rt.on_pre_verify(
                session_id="test_sess_clean",
                changed_paths=["foo.py"],
                attempt=0,
                final_response="All done."
            )
            # Must allow completion
            self.assertIsNone(res)
            # Session must be marked as pre-verified
            self.assertIn("test_sess_clean", rt._pre_verified_sessions)

            # on_post_llm_call right after must skip background review
            called = []
            rt._enqueue_review = lambda delta: called.append(delta)
            rt.on_post_llm_call(
                session_id="test_sess_clean",
                turn_id="turn_1",
                user_message="build it",
                assistant_response="done",
                conversation_history=[{"role": "user", "content": "build it"}]
            )
            self.assertEqual(called, [], "post_llm_call must skip review when turn was pre-verified")
        finally:
            runtime_mod._get_git_diff = orig_diff

    def test_pre_verify_attempt_cap(self):
        class MockCtx:
            pass
        rt = runtime_mod.AdvisorRuntime(MockCtx())
        # Attempt 2 or higher must return None immediately
        res = rt.on_pre_verify(
            session_id="test_sess",
            changed_paths=["foo.py"],
            attempt=2,
            final_response="Done."
        )
        self.assertIsNone(res)
        # Attempt >= 2 must NOT add to _pre_verified_sessions so post_llm_call can review remaining findings
        self.assertNotIn("test_sess", rt._pre_verified_sessions)

    def test_git_diff_real_repo_cases(self):
        import subprocess
        _get_diff = runtime_mod._get_git_diff
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
            
            # Tracked file
            f1 = Path(tmpdir) / "app.py"
            f1.write_text("x = 1\n")
            subprocess.run(["git", "add", "app.py"], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, check=True, capture_output=True)
            f1.write_text("x = 2\n")
            
            # Untracked new file
            f2 = Path(tmpdir) / "helper.py"
            f2.write_text("def helper(): pass\n")
            
            # .github/ workflow file (must NOT be skipped by .git check)
            gh_dir = Path(tmpdir) / ".github" / "workflows"
            gh_dir.mkdir(parents=True)
            f3 = gh_dir / "ci.yml"
            f3.write_text("name: Test CI\n")
            
            # Deleted file
            f4 = Path(tmpdir) / "obsolete.py"
            f4.write_text("remove me\n")
            subprocess.run(["git", "add", "obsolete.py"], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add obsolete"], cwd=tmpdir, check=True, capture_output=True)
            f4.unlink()
            
            diff_out = _get_diff(tmpdir, ["app.py", "helper.py", ".github/workflows/ci.yml", "obsolete.py"])
            self.assertIsNotNone(diff_out)
            self.assertIn("x = 2", diff_out, "Tracked change must appear in diff")
            self.assertIn("def helper(): pass", diff_out, "Untracked new file must appear in diff")
            self.assertIn("name: Test CI", diff_out, ".github/ workflow file must appear in diff")


if __name__ == "__main__":
    unittest.main()
