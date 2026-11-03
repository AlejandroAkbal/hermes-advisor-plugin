"""Unit tests for substantial verifier upgrades: ledger, prober, dialectic."""

import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

try:
    from hermes_plugins.advisor.dialectic import DialecticAuditor
    from hermes_plugins.advisor.ledger import audit_draft_ledger
    from hermes_plugins.advisor.prober import (
        probe_command_available,
        probe_directory_existence,
        probe_env_var,
        probe_file_existence,
        probe_git_branch,
        probe_git_clean_status,
        probe_git_commit,
        probe_git_remote,
        probe_git_tag,
        probe_port_listening,
        probe_process_running,
        probe_service_running,
        run_active_probes_for_draft,
    )
    from hermes_plugins.advisor.runtime import AdvisorRuntime
except ImportError:
    from advisor.dialectic import DialecticAuditor
    from advisor.ledger import audit_draft_ledger
    from advisor.prober import (
        probe_command_available,
        probe_directory_existence,
        probe_env_var,
        probe_file_existence,
        probe_git_branch,
        probe_git_clean_status,
        probe_git_commit,
        probe_git_remote,
        probe_git_tag,
        probe_port_listening,
        probe_process_running,
        probe_service_running,
        run_active_probes_for_draft,
    )
    from advisor.runtime import AdvisorRuntime


class TestDeterministicLedger(unittest.TestCase):
    def test_rejects_empty_and_placeholder_drafts(self):
        v = audit_draft_ledger("", [])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "empty_draft")

        v = audit_draft_ledger("Testing verify_turn", [])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "placeholder_draft")

        v = audit_draft_ledger("...", [])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "placeholder_draft")

    def test_corroboration_rule_catches_unrun_test_claims(self):
        # Claim tests pass, but no test tools ran
        v = audit_draft_ledger("All 36 tests pass cleanly.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertTrue(len(v.missing_checks) > 0)
        self.assertIn("pytest or unittest", v.missing_checks[0])

        # Claim tests pass, and unittest was run in tool_events
        v_ok = audit_draft_ledger(
            "All 36 tests pass cleanly.",
            tool_events=[{"tool": "terminal", "args": "python -m unittest", "result": "Ran 36 tests. OK"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_unrun_git_status_claim(self):
        v = audit_draft_ledger("The working tree is clean and ready.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("git status", v.missing_checks[0])

        v_ok = audit_draft_ledger(
            "The working tree is clean and ready.",
            tool_events=[{"tool": "terminal", "args": "git status", "result": "nothing to commit, working tree clean"}]
        )
        self.assertTrue(v_ok.passed)

        # Expanded git clean phrasing: 'no uncommitted changes'
        v_no_uncommitted = audit_draft_ledger("There are no uncommitted changes.", tool_events=[])
        self.assertFalse(v_no_uncommitted.passed)
        self.assertEqual(v_no_uncommitted.rule, "unbacked_assertion")

    def test_corroboration_rule_catches_unrun_git_pushed_expanded(self):
        # Unrun push claim with various phrasings
        for draft in [
            "Pushed changes to origin.",
            "Pushed the branch to remote.",
            "Changes pushed to origin successfully.",
        ]:
            v = audit_draft_ledger(draft, tool_events=[])
            self.assertFalse(v.passed, f"Expected failure for: {draft}")
            self.assertEqual(v.rule, "unbacked_assertion")
            self.assertIn("git push", v.missing_checks[0])

        # Negated push statements should NOT require push evidence
        for draft in [
            "No changes pushed to origin.",
            "Changes were not pushed to remote.",
            "Without changes pushed to origin, we stayed local.",
            "I have not yet pushed changes to origin.",
            "No changes were pushed to origin.",
            "Nothing was pushed to remote.",
            "The changes are ready to be pushed to origin.",
            "Changes have not yet been pushed to origin.",
        ]:
            v_neg = audit_draft_ledger(draft, tool_events=[])
            self.assertTrue(v_neg.passed, f"Expected pass for negated claim: {draft}")

        # Supported push
        v_ok = audit_draft_ledger(
            "Pushed changes to origin.",
            tool_events=[{"tool": "terminal", "args": "git push origin feat/x", "result": "Everything up-to-date"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_unrun_pr_created(self):
        v = audit_draft_ledger("Opened PR #15 on GitHub.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("gh pr create", v.missing_checks[0])

        # Negated PR creation should NOT require PR creation evidence
        for draft in [
            "I have not opened a PR yet.",
            "I have not yet opened a PR.",
            "No PR was opened.",
        ]:
            v_neg = audit_draft_ledger(draft, tool_events=[])
            self.assertTrue(v_neg.passed, f"Expected pass for negated claim: {draft}")

        # Unrelated gh pr list does NOT satisfy pr_created
        v_list = audit_draft_ledger(
            "Opened PR #15 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr list", "result": "[]"}]
        )
        self.assertFalse(v_list.passed)

        # Failed gh pr create does NOT satisfy pr_created
        v_fail = audit_draft_ledger(
            "Opened PR #15 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr create", "result": "GraphQL error: pull request create failed"}]
        )
        self.assertFalse(v_fail.passed)

        # Mismatched PR number does NOT satisfy pr_created
        v_mismatch = audit_draft_ledger(
            "Opened PR #99 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'x' --body 'y'", "result": "https://github.com/foo/bar/pull/15"}]
        )
        self.assertFalse(v_mismatch.passed)

        # Mismatched "pull request #N" phrasing does NOT satisfy pr_created
        v_mismatch_pr = audit_draft_ledger(
            "Opened pull request #99 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'x' --body 'y'", "result": "https://github.com/foo/bar/pull/15"}]
        )
        self.assertFalse(v_mismatch_pr.passed)

        # Substring prefix (e.g. PR #1 vs pull/15) does NOT satisfy pr_created
        v_sub = audit_draft_ledger(
            "Opened PR #1 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'x' --body 'y'", "result": "https://github.com/foo/bar/pull/15"}]
        )
        self.assertFalse(v_sub.passed)

        # Unrelated tool event containing the number (e.g. --limit 100) does NOT satisfy pr_created
        v_unrelated = audit_draft_ledger(
            "Opened PR #100 on GitHub.",
            tool_events=[
                {"tool": "terminal", "args": "git log --limit 100", "result": "100 commits"},
                {"tool": "terminal", "args": "gh pr create", "result": "https://github.com/foo/bar/pull/15"}
            ]
        )
        self.assertFalse(v_unrelated.passed)

        # Supported PR creation matching PR number
        v_ok = audit_draft_ledger(
            "Opened PR #15 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'x' --body 'y'", "result": "https://github.com/foo/bar/pull/15"}]
        )
        self.assertTrue(v_ok.passed)

        # Matched claim has no PR number (e.g. 'opened a new PR'), even though draft mentions PR #12 earlier
        v_diff_context = audit_draft_ledger(
            "Following feedback on PR #12, I opened a new PR.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'x' --body 'y'", "result": "https://github.com/foo/bar/pull/15"}]
        )
        self.assertTrue(v_diff_context.passed)

    def test_corroboration_rule_catches_unrun_coderabbit_reviewed(self):
        v = audit_draft_ledger("CodeRabbit review passed cleanly.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("coderabbit review", v.missing_checks[0])

        # Version check does NOT satisfy review requirement
        v_ver = audit_draft_ledger(
            "CodeRabbit review passed cleanly.",
            tool_events=[{"tool": "terminal", "args": "coderabbit --version", "result": "coderabbit version 0.1.0"}]
        )
        self.assertFalse(v_ver.passed)

        # Failed review does NOT satisfy review requirement
        v_fail = audit_draft_ledger(
            "CodeRabbit review passed cleanly.",
            tool_events=[{"tool": "terminal", "args": "coderabbit review --agent", "result": "Error: review failed: exit_code: 1"}]
        )
        self.assertFalse(v_fail.passed)

        # Successful review with finding notes containing 'failed' in body passes
        review_with_findings = """Review completed. Findings:
- In test_auth.py: tests failed to cover the logout route.
- In login.py: consider using secrets.compare_digest."""
        v_findings = audit_draft_ledger(
            "CodeRabbit review passed cleanly.",
            tool_events=[{"tool": "terminal", "args": "coderabbit review --agent", "result": review_with_findings}]
        )
        self.assertTrue(v_findings.passed)

        # Successful review passes
        v_ok = audit_draft_ledger(
            "CodeRabbit review passed cleanly.",
            tool_events=[{"tool": "terminal", "args": "coderabbit review --agent", "result": "Review completed. No issues found."}]
        )
        self.assertTrue(v_ok.passed)

    def test_catches_unacknowledged_tool_errors(self):
        tool_events = [
            {"tool": "terminal", "args": "git push", "result": "error: failed to push some refs, exit_code: 1"}
        ]
        # Draft claims unqualified success without mentioning failure
        v = audit_draft_ledger("All clean and done successfully.", tool_events=tool_events)
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unacknowledged_tool_failure")

        # Draft acknowledges the failure
        v_ack = audit_draft_ledger("Done successfully, except the push failed with an exit error.", tool_events=tool_events)
        self.assertTrue(v_ack.passed)

        # Test traceback pattern in tool errors
        tb_event = [{"tool": "terminal", "args": "python run.py", "result": "Traceback (most recent call last):\n  File 'x.py', line 1\nValueError: boom"}]
        v_tb = audit_draft_ledger("Everything works and is healthy.", tool_events=tb_event)
        self.assertFalse(v_tb.passed)
        self.assertEqual(v_tb.rule, "unacknowledged_tool_failure")

        # Test JSON tool output exit code matching
        json_err_event = [{"tool": "terminal", "args": "git push", "result": '{"output": "remote rejected", "exit_code": 1}'}]
        v_json = audit_draft_ledger("Everything works and is healthy.", tool_events=json_err_event)
        self.assertFalse(v_json.passed)
        self.assertEqual(v_json.rule, "unacknowledged_tool_failure")

        # Test claims text is checked when draft is non-empty
        v_claims = audit_draft_ledger(draft="Done.", claims=["All 36 tests pass"], tool_events=[])
        self.assertFalse(v_claims.passed)
        self.assertEqual(v_claims.rule, "unbacked_assertion")

    def test_corroboration_rule_catches_unrun_or_failed_git_committed(self):
        # Unrun commit claim
        v = audit_draft_ledger("Committed changes to the feature branch.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("git commit", v.missing_checks[0])

        # Negated commit claim should pass
        v_neg = audit_draft_ledger("I have not committed the changes yet.", tool_events=[])
        self.assertTrue(v_neg.passed)

        # Failed commit (nothing to commit)
        v_fail = audit_draft_ledger(
            "Committed changes to the feature branch.",
            tool_events=[{"tool": "terminal", "args": "git commit -m 'fix'", "result": "On branch feat\nnothing to commit, working tree clean"}]
        )
        self.assertFalse(v_fail.passed)

        # Successful commit
        v_ok = audit_draft_ledger(
            "Committed changes to the feature branch.",
            tool_events=[{"tool": "terminal", "args": "git commit -m 'fix'", "result": "[feat 1234abc] fix: resolve bug\n 1 file changed"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_failed_git_pushed(self):
        # Failed push
        v_fail = audit_draft_ledger(
            "Changes pushed to origin successfully.",
            tool_events=[{"tool": "terminal", "args": "git push origin feat", "result": "error: failed to push some refs to 'origin'\nexit_code: 1"}]
        )
        self.assertFalse(v_fail.passed)

        # Successful push
        v_ok = audit_draft_ledger(
            "Changes pushed to origin successfully.",
            tool_events=[{"tool": "terminal", "args": "git push origin feat", "result": "To github.com:user/repo.git\n * [new branch] feat -> feat"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_unrun_or_failed_pr_merged(self):
        # Unrun PR merge
        v = audit_draft_ledger("Merged PR #19 successfully.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("gh pr merge", v.missing_checks[0])

        # Failed PR merge
        v_fail = audit_draft_ledger(
            "Merged PR #19 successfully.",
            tool_events=[{"tool": "terminal", "args": "gh pr merge 19 --admin", "result": "GraphQL error: pull request merge failed\nexit_code: 1"}]
        )
        self.assertFalse(v_fail.passed)

        # Mismatched PR number in merge
        v_mismatch = audit_draft_ledger(
            "Merged PR #99 successfully.",
            tool_events=[{"tool": "terminal", "args": "gh pr merge 19 --admin", "result": "Merged PR #19"}]
        )
        self.assertFalse(v_mismatch.passed)

        # Successful PR merge via gh pr merge
        v_ok = audit_draft_ledger(
            "Merged PR #19 successfully.",
            tool_events=[{"tool": "terminal", "args": "gh pr merge 19 --admin", "result": "Merged PR #19 (AlejandroAkbal/hermes-advisor-plugin)"}]
        )
        self.assertTrue(v_ok.passed)

        # Successful PR merge corroborated via gh pr view
        v_view_ok = audit_draft_ledger(
            "PR #19 was merged into main.",
            tool_events=[{"tool": "terminal", "args": "gh pr view 19 --json state", "result": '{"state":"MERGED"}'}]
        )
        self.assertTrue(v_view_ok.passed)

        # Failed PR merge corroboration via gh pr view (still OPEN)
        v_view_open = audit_draft_ledger(
            "PR #19 was merged into main.",
            tool_events=[{"tool": "terminal", "args": "gh pr view 19 --json state", "result": '{"state":"OPEN"}'}]
        )
        self.assertFalse(v_view_open.passed)

        # Failed PR merge corroboration via gh pr view with misleading title containing 'merged'
        v_view_title_mislead = audit_draft_ledger(
            "PR #19 was merged into main.",
            tool_events=[{"tool": "terminal", "args": "gh pr view 19 --json state,title", "result": '{"state":"OPEN","title":"Not merged yet"}'}]
        )
        self.assertFalse(v_view_title_mislead.passed)

        # gh pr view cannot corroborate an active merge action claim
        v_action_denied = audit_draft_ledger(
            "Merged PR #19 successfully.",
            tool_events=[{"tool": "terminal", "args": "gh pr view 19 --json state", "result": '{"state":"MERGED"}'}]
        )
        self.assertFalse(v_action_denied.passed)

        # Successful PR merge corroborated via gh pr view with whitespace variation
        v_view_ws = audit_draft_ledger(
            "PR #19 was merged into main.",
            tool_events=[{"tool": "terminal", "args": "gh  pr  view 19 --json state", "result": '{"state":"MERGED"}'}]
        )
        self.assertTrue(v_view_ws.passed)

        # Successful PR merge corroborated via gh pr view with full pull request URL
        v_view_url = audit_draft_ledger(
            "PR #19 was merged into main.",
            tool_events=[{"tool": "terminal", "args": "gh pr view https://github.com/AlejandroAkbal/hermes-advisor-plugin/pull/19 --json state", "result": '{"state":"MERGED"}'}]
        )
        self.assertTrue(v_view_url.passed)

        # Unrelated PR view with title mentioning PR #19 cannot corroborate PR #19
        v_view_unrelated = audit_draft_ledger(
            "PR #19 was merged into main.",
            tool_events=[{"tool": "terminal", "args": "gh pr view 25 --json state,title", "result": '{"number":25,"state":"MERGED","title":"Fix follow-up for PR #19"}'}]
        )
        self.assertFalse(v_view_unrelated.passed)

        # Attribution lookbehind: reporting on cron/log past PR actions
        v_cron_report = audit_draft_ledger(
            "The cron opened PR #15 and the cron merged PR #15 earlier.",
            tool_events=[]
        )
        self.assertTrue(v_cron_report.passed)

    def test_corroboration_rule_catches_failed_test_suite(self):
        # Failed test suite with 'all tests pass' claim
        v_fail = audit_draft_ledger(
            "All tests pass.",
            tool_events=[{"tool": "terminal", "args": "python -m unittest", "result": "FAILED (failures=1, errors=0)\nexit_code: 1"}]
        )
        self.assertFalse(v_fail.passed)

        # Successful test suite with 0 failures explicitly in string
        v_ok_zero = audit_draft_ledger(
            "All tests pass.",
            tool_events=[{"tool": "terminal", "args": "python -m unittest", "result": "failures=0, errors=0. OK"}]
        )
        self.assertTrue(v_ok_zero.passed)

        # Pytest collection/setup errors
        v_fail_pytest_err = audit_draft_ledger(
            "All tests pass.",
            tool_events=[{"tool": "terminal", "args": "pytest", "result": "1 error in 0.3s"}]
        )
        self.assertFalse(v_fail_pytest_err.passed)

        # Non-git prose like 'committed to quality' should not trigger git_committed rule
        v_prose = audit_draft_ledger("We are committed to quality and fixing the issue.", tool_events=[])
        self.assertTrue(v_prose.passed)

        # git -c user.name=... commit should corroborate git_committed
        v_git_c = audit_draft_ledger(
            "Committed the fix to the branch.",
            tool_events=[{"tool": "terminal", "args": "git -c user.name='Bot' commit -m 'fix'", "result": "[feat 123] fix"}]
        )
        self.assertTrue(v_git_c.passed)

        # Mismatched PR merge across multiple events (one failed for 19, one succeeded for 20)
        v_multi_merge = audit_draft_ledger(
            "Merged PR #19 successfully.",
            tool_events=[
                {"tool": "terminal", "args": "gh pr merge 19", "result": "exit_code: 1\nerror: merge conflict"},
                {"tool": "terminal", "args": "gh pr merge 20", "result": "Merged PR #20"},
            ]
        )
        self.assertFalse(v_multi_merge.passed)

    def test_probe_text_does_not_corroborate_test_suite_execution(self):
        # Arbitrary probe evidence containing 'pytest' must not corroborate a test run without a tool event
        v = audit_draft_ledger(
            "All tests pass.",
            tool_events=[],
            evidence="[Active Probe CONFIRMED]: test -e /tmp/pytest.log -> exists=True"
        )
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")

    def test_multiple_cited_prs_must_all_be_corroborated(self):
        # Draft cites two merged PRs, but only one was merged
        v = audit_draft_ledger(
            "Merged PR #19 and merged PR #20 successfully.",
            tool_events=[{"tool": "terminal", "args": "gh pr merge 19 --admin", "result": "https://github.com/org/repo/pull/19"}]
        )
        self.assertFalse(v.passed)
        self.assertIn("PR #20", v.reason)

    def test_operation_identity_avoids_spurious_pr_number_matches(self):
        # PR title cites #99, but actual PR created was #15
        v = audit_draft_ledger(
            "Opened PR #99.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'Follow-up to PR #99'", "result": "https://github.com/org/repo/pull/15"}]
        )
        self.assertFalse(v.passed)
        self.assertIn("PR #99", v.reason)

    def test_evidence_parameter_is_checked_in_corroboration(self):
        # Corroborating evidence supplied in evidence parameter (e.g. from active probes)
        v = audit_draft_ledger(
            "The working tree is clean.",
            tool_events=[],
            evidence="[Active Probe CONFIRMED]: `git -C /repo status --porcelain` -> working tree clean"
        )
        self.assertTrue(v.passed)

        # Contradicted probe evidence must not satisfy corroboration
        v_contra = audit_draft_ledger(
            "The working tree is clean.",
            tool_events=[],
            evidence="[Active Probe CONTRADICTED]: `git -C /repo status --porcelain` -> M file.py"
        )
        self.assertFalse(v_contra.passed)

        # Inconclusive probe evidence must not satisfy corroboration
        v_incon = audit_draft_ledger(
            "The working tree is clean.",
            tool_events=[],
            evidence="[Active Probe INCONCLUSIVE]: `git -C /repo status --porcelain` -> Probe inconclusive: timeout"
        )
        self.assertFalse(v_incon.passed)


class TestActiveProber(unittest.TestCase):
    def test_probe_file_existence(self):
        with tempfile.NamedTemporaryFile() as tmp:
            res = probe_file_existence(tmp.name)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)

        res_missing = probe_file_existence("/tmp/non_existent_file_xyz_123.txt")
        self.assertTrue(res_missing.executed)
        self.assertFalse(res_missing.success)

    def test_probe_git_clean_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            res = probe_git_clean_status(tmp_dir)
            self.assertIsNotNone(res)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)

    def test_run_active_probes_for_draft(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            draft = "The working tree is clean and ready."
            probes = run_active_probes_for_draft(draft, tmp_dir)
            self.assertTrue(len(probes) >= 1)
            self.assertIn("working tree clean", probes[0].claim)
            self.assertTrue(probes[0].success)

            # Test expanded phrasing 'no uncommitted changes'
            draft_expanded = "There are no uncommitted changes."
            probes_expanded = run_active_probes_for_draft(draft_expanded, tmp_dir)
            self.assertTrue(len(probes_expanded) >= 1)
            self.assertTrue(probes_expanded[0].success)

    def test_probe_git_branch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init", "-b", "my-feature"], cwd=tmp_dir, check=True, capture_output=True)
            res = probe_git_branch(tmp_dir)
            self.assertIsNotNone(res)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)
            self.assertEqual(res.output, "my-feature")

    def test_run_active_probes_for_branch_and_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init", "-b", "feat/my-probe"], cwd=tmp_dir, check=True, capture_output=True)
            # Create a test file
            test_file = os.path.join(tmp_dir, "test_artifact.txt")
            with open(test_file, "w") as f:
                f.write("hello")

            draft = f"I am on branch feat/my-probe and created {test_file}."
            probes = run_active_probes_for_draft(draft, tmp_dir)
            self.assertTrue(len(probes) >= 2)
            branch_probe = next((p for p in probes if "branch" in p.claim), None)
            file_probe = next((p for p in probes if "file exists" in p.claim), None)
            self.assertIsNotNone(branch_probe)
            self.assertIsNotNone(file_probe)
            assert branch_probe is not None
            assert file_probe is not None
            self.assertTrue(branch_probe.success)
            self.assertTrue(file_probe.success)

            # Test path before exists: 'artifact.json exists'
            json_file = os.path.join(tmp_dir, "config.json")
            with open(json_file, "w") as f:
                f.write("{}")
            draft_pre = f"{json_file} exists in the directory."
            probes_pre = run_active_probes_for_draft(draft_pre, tmp_dir)
            self.assertTrue(any("config.json" in p.claim and p.success for p in probes_pre))

            # Test non-branch phrasing 'on Linux' and 'on feature parity' do not trigger false branch probe
            draft_os = "All tests pass on Linux, working on feature parity."
            probes_os = run_active_probes_for_draft(draft_os, tmp_dir)
            self.assertFalse(any("branch" in p.claim for p in probes_os))

            # Test negated file creation and non-path words like 'Node.js' or '3.5' do not trigger file probe
            draft_neg = "I have not created config.json yet. Saved 3.5 seconds, written in Node.js."
            probes_neg = run_active_probes_for_draft(draft_neg, tmp_dir)
            self.assertFalse(any("config.json" in p.claim for p in probes_neg))
            self.assertFalse(any("3.5" in p.claim for p in probes_neg))
            self.assertFalse(any("Node.js" in p.claim for p in probes_neg))

    @unittest.skipUnless(shutil.which("lsof"), "lsof not available")
    def test_probe_port_listening(self):
        """Test probing TCP port listening state."""
        # Invalid port
        res_invalid = probe_port_listening(999999)
        self.assertFalse(res_invalid.executed)
        self.assertFalse(res_invalid.success)

        # Listening port via ephemeral loopback socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            res_listen = probe_port_listening(port)
            self.assertTrue(res_listen.executed)
            self.assertTrue(res_listen.success)

        # Once socket is closed, port is unbound (falls back to connection check which is inconclusive on refused connect)
        res_unbound = probe_port_listening(port)
        self.assertFalse(res_unbound.success)
        self.assertFalse(res_unbound.executed)
        self.assertIn("inconclusive", res_unbound.output.lower())

    @unittest.skipUnless(shutil.which("ps"), "ps not available")
    def test_probe_process_running(self):
        """Test probing process running state and zombie process rejection."""
        # Invalid PID
        res_invalid = probe_process_running(-1)
        self.assertFalse(res_invalid.executed)
        self.assertFalse(res_invalid.success)

        # Current running process PID
        current_pid = os.getpid()
        res_current = probe_process_running(current_pid)
        self.assertTrue(res_current.executed)
        self.assertTrue(res_current.success)
        self.assertIn(str(current_pid), res_current.output)

        # Non-existent PID
        with mock.patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=["ps"], returncode=1, stdout="", stderr=""),
        ):
            res_dead = probe_process_running(99998)
            self.assertTrue(res_dead.executed)
            self.assertFalse(res_dead.success)

        # Zombie process PID rejection
        with mock.patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=["ps"],
                returncode=0,
                stdout="12345 Z+ alejandro <defunct>\n",
                stderr="",
            ),
        ):
            res_zombie = probe_process_running(12345)
            self.assertTrue(res_zombie.executed)
            self.assertFalse(res_zombie.success)
            self.assertIn("zombie", res_zombie.output.lower())

    def test_probe_command_available(self):
        """Test probing command and CLI tool availability in PATH."""
        # Known existing command (e.g. python3)
        res_existing = probe_command_available("python3")
        self.assertTrue(res_existing.executed)
        self.assertTrue(res_existing.success)
        self.assertIn("available at", res_existing.output)

        # Non-existent command
        res_missing = probe_command_available("non_existent_binary_xyz_98765")
        self.assertTrue(res_missing.executed)
        self.assertFalse(res_missing.success)
        self.assertIn("not found in PATH", res_missing.output)

        # Invalid command name
        res_invalid = probe_command_available("rm -rf /")
        self.assertFalse(res_invalid.executed)
        self.assertFalse(res_invalid.success)

    def test_probe_git_tag(self):
        """Test probing git tags in a repository."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, check=True, capture_output=True)
            with open(os.path.join(tmp_dir, "f.txt"), "w") as f:
                f.write("test")
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "tag base"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "tag", "v1.2.3"], cwd=tmp_dir, check=True, capture_output=True)

            # Existing tag
            res_tag = probe_git_tag("v1.2.3", tmp_dir)
            self.assertIsNotNone(res_tag)
            self.assertTrue(res_tag.executed)
            self.assertTrue(res_tag.success)
            self.assertIn("tag v1.2.3 exists", res_tag.output)

            # Non-existent tag
            res_missing_tag = probe_git_tag("v9.9.9", tmp_dir)
            self.assertIsNotNone(res_missing_tag)
            self.assertTrue(res_missing_tag.executed)
            self.assertFalse(res_missing_tag.success)

    def test_probe_git_commit(self):
        """Test probing git commits in a repository."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, check=True, capture_output=True)
            test_file = os.path.join(tmp_dir, "f.txt")
            with open(test_file, "w") as f:
                f.write("content")
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "initial test commit"], cwd=tmp_dir, check=True, capture_output=True)

            log_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_dir, check=True, capture_output=True, text=True)
            head_sha = log_proc.stdout.strip()

            # Existing commit
            res = probe_git_commit(head_sha[:7], tmp_dir)
            self.assertIsNotNone(res)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)
            self.assertIn("initial test commit", res.output)

            # Non-existent commit
            res_bad = probe_git_commit("0123456789abcdef", tmp_dir)
            self.assertIsNotNone(res_bad)
            self.assertTrue(res_bad.executed)
            self.assertFalse(res_bad.success)

    def test_probe_service_running_not_found(self):
        """Test that an absent launchd service safely returns None."""
        # Service absent from caller's launchctl domain safely returns None rather than false CONTRADICTED
        res = probe_service_running("com.nonexistent.service.xyz.12345")
        self.assertIsNone(res)

    @unittest.skipUnless(shutil.which("ps"), "ps not available")
    def test_run_active_probes_for_port_pid_and_commit(self):
        """Test draft probe extraction for ports, PIDs, and commit SHAs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, check=True, capture_output=True)
            with open(os.path.join(tmp_dir, "f.txt"), "w") as f:
                f.write("content")
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "probe integration"], cwd=tmp_dir, check=True, capture_output=True)

            log_proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_dir, check=True, capture_output=True, text=True)
            short_sha = log_proc.stdout.strip()

            cur_pid = os.getpid()
            draft = (
                f"The updater is running (PID {cur_pid}), bound to port 59999, listening on 192.168.1.10:8080, "
                f"committed in {short_sha}, and service com.logi.optionsplus.updater. is running."
            )
            with unittest.mock.patch(
                f"{run_active_probes_for_draft.__module__}.probe_service_running",
                return_value=None,
            ) as mock_svc:
                probes = run_active_probes_for_draft(draft, tmp_dir)

            mock_svc.assert_any_call("com.logi.optionsplus.updater")
            self.assertNotIn("com.logi.optionsplus.updater.", [c.args[0] for c in mock_svc.call_args_list])

            claims = [p.claim for p in probes]
            self.assertTrue(any(f"PID {cur_pid}" in c for c in claims))
            self.assertTrue(any("port 59999" in c for c in claims))
            self.assertTrue(any("port 8080" in c for c in claims))
            self.assertFalse(any("port 192" in c for c in claims))
            self.assertTrue(any(f"commit {short_sha}" in c for c in claims))

            # The current PID and commit should be successful
            pid_probe = next(p for p in probes if f"PID {cur_pid}" in p.claim)
            self.assertTrue(pid_probe.success)

            commit_probe = next(p for p in probes if f"commit {short_sha}" in p.claim)
            self.assertTrue(commit_probe.success)

    def test_run_active_probes_for_command_and_tag(self):
        """Test draft probe extraction for commands and git tags."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, check=True, capture_output=True)
            with open(os.path.join(tmp_dir, "f.txt"), "w") as f:
                f.write("content")
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "tag commit"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "tag", "v2.0.0"], cwd=tmp_dir, check=True, capture_output=True)

            draft = "The python3 CLI is installed and the release is tagged as v2.0.0."
            probes = run_active_probes_for_draft(draft, tmp_dir)
            claims = [p.claim for p in probes]
            self.assertTrue(any("command python3 available" in c for c in claims))
            self.assertTrue(any("tag v2.0.0 exists" in c for c in claims))

            cmd_probe = next(p for p in probes if "command python3 available" in p.claim)
            self.assertTrue(cmd_probe.success)

            tag_probe = next(p for p in probes if "tag v2.0.0 exists" in p.claim)
            self.assertTrue(tag_probe.success)

            # Negated assertions (with or without intervening words) should not trigger affirmative probes
            negated_draft = "We did not create git tag v9.9.9, and have not yet installed fake_cli CLI."
            neg_probes = run_active_probes_for_draft(negated_draft, tmp_dir)
            neg_claims = [p.claim for p in neg_probes]
            self.assertFalse(any("fake_cli" in c for c in neg_claims))
            self.assertFalse(any("v9.9.9" in c for c in neg_claims))

            # Git tag with options and message flags captures tag name and not message string
            opt_draft = 'Ran git tag -a -m "release v2" v2.0.0 for release.'
            opt_probes = run_active_probes_for_draft(opt_draft, tmp_dir)
            opt_claims = [p.claim for p in opt_probes]
            self.assertTrue(any("tag v2.0.0 exists" in c for c in opt_claims))
            self.assertFalse(any("-a" in c for c in opt_claims))
            self.assertFalse(any("release" in c for c in opt_claims))

            # Combined options such as -am also capture the tag name rather than message
            combined_draft = 'Ran git tag -am "release v2" v2.0.0 for release.'
            comb_probes = run_active_probes_for_draft(combined_draft, tmp_dir)
            comb_claims = [p.claim for p in comb_probes]
            self.assertTrue(any("tag v2.0.0 exists" in c for c in comb_claims))
            self.assertFalse(any("-am" in c for c in comb_claims))
            self.assertFalse(any("release" in c for c in comb_claims))

            # Tag deletion commands (e.g. -d, --delete) do not trigger existence probes
            del_draft = "Deleted obsolete tag: git tag -d v0.9.0 and git tag --delete v0.8.0."
            del_probes = run_active_probes_for_draft(del_draft, tmp_dir)
            del_claims = [p.claim for p in del_probes]
            self.assertFalse(any("v0.9.0" in c for c in del_claims))
            self.assertFalse(any("v0.8.0" in c for c in del_claims))

            # Tag names with build metadata (+) are fully matched
            subprocess.run(["git", "tag", "v2.0.0+build.1"], cwd=tmp_dir, check=True, capture_output=True)
            plus_draft = "The release is tagged as v2.0.0+build.1."
            plus_probes = run_active_probes_for_draft(plus_draft, tmp_dir)
            plus_claims = [p.claim for p in plus_probes]
            self.assertTrue(any("tag v2.0.0+build.1 exists" in c for c in plus_claims))

            # Tag probe budget bounds execution to at most MAX_TAG_PROBES
            many_tags = " ".join(f"tagged as v1.0.{i}" for i in range(20))
            budget_probes = run_active_probes_for_draft(many_tags, tmp_dir)
            tag_probe_count = sum(1 for p in budget_probes if "tag" in p.claim)
            self.assertLessEqual(tag_probe_count, 5)


class TestDialecticAuditor(unittest.TestCase):
    def test_arbiter_dismisses_pedantic_stylistic_accusations(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        # Challenger returns a pedantic stylistic accusation
        challenger_text = """[
            {
                "claim": "Area assignment succeeded",
                "category": "stylistic_omission",
                "evidence_gap": "The agent did not explain how IR remote limits work in detail."
            }
        ]"""
        # Arbiter dismisses it and approves
        arbiter_text = """{
            "status": "approved",
            "reason": "All core claims verified. Pedantic accusation dismissed."
        }"""

        calls = []
        def mock_complete(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return MockLLMResult(challenger_text)
            return MockLLMResult(arbiter_text)

        auditor = DialecticAuditor(mock_complete)
        accusations = auditor.run_challenger("Area assignment succeeded", "success=True", "", "")
        self.assertEqual(len(accusations), 1)

        decision = auditor.run_arbiter("Area assignment succeeded", "success=True", "", accusations)
        self.assertEqual(decision["status"], "approved")

    def test_arbiter_upholds_substantive_blocker(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        challenger_text = """[
            {
                "claim": "All 36 tests pass",
                "category": "unverified_claim",
                "evidence_gap": "No test runner was executed in this turn."
            }
        ]"""
        arbiter_text = """{
            "status": "rejected",
            "reason": "Draft claims tests pass, but no test runner was executed.",
            "missing_checks": ["Run python -m unittest"],
            "unbacked_claims": ["All 36 tests pass"]
        }"""

        calls = []
        def mock_complete(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return MockLLMResult(challenger_text)
            return MockLLMResult(arbiter_text)

        auditor = DialecticAuditor(mock_complete)
        accusations = auditor.run_challenger("All 36 tests pass", "No tools ran", "", "")
        decision = auditor.run_arbiter("All 36 tests pass", "No tools ran", "", accusations)
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["missing_checks"], ["Run python -m unittest"])

    def test_dialectic_handles_challenger_failure_and_non_dict_items(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        # Test non-dict item filtering
        auditor = DialecticAuditor(lambda **kw: MockLLMResult('["not_a_dict", {"claim": "c", "category": "cat", "evidence_gap": "gap"}]'))
        accusations = auditor.run_challenger("draft", "tools", "", "")
        self.assertIsNotNone(accusations)
        self.assertEqual(len(accusations), 1)
        self.assertEqual(accusations[0]["claim"], "c")

        # Test failure returns None and arbiter returns error status
        auditor_fail = DialecticAuditor(lambda **kw: (_ for _ in ()).throw(RuntimeError("LLM exploded")))
        accusations_fail = auditor_fail.run_challenger("draft", "tools", "", "")
        self.assertIsNone(accusations_fail)

        res_err = auditor_fail.run_arbiter("draft", "tools", "", accusations=None)
        self.assertEqual(res_err["status"], "error")

        # Test arbiter failure returns error status
        res_arbiter_err = auditor_fail.run_arbiter("draft", "tools", "", accusations=[{"claim": "c", "category": "cat", "evidence_gap": "gap"}])
        self.assertEqual(res_arbiter_err["status"], "error")

    def test_probe_directory_existence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res_true = probe_directory_existence(tmpdir)
            self.assertTrue(res_true.executed)
            self.assertTrue(res_true.success)
            self.assertIn("isdir=True", res_true.output)

            non_existent = os.path.join(tmpdir, "missing_subdir")
            res_false = probe_directory_existence(non_existent)
            self.assertTrue(res_false.executed)
            self.assertFalse(res_false.success)
            self.assertIn("isdir=False", res_false.output)

    def test_probe_env_var(self):
        test_key = "HERMES_ADVISOR_TEST_ENV_VAR"
        try:
            os.environ[test_key] = "active_value"
            res_set = probe_env_var(test_key)
            self.assertTrue(res_set.executed)
            self.assertTrue(res_set.success)
            self.assertIn(f"{test_key} is set", res_set.output)

            # Test dollar-prefix stripping
            res_dollar = probe_env_var(f"${test_key}")
            self.assertTrue(res_dollar.executed)
            self.assertTrue(res_dollar.success)

            # Test unset variable
            res_unset = probe_env_var("HERMES_DEFINITELY_UNSET_VAR_XYZ")
            self.assertTrue(res_unset.executed)
            self.assertFalse(res_unset.success)
            self.assertIn("is not set", res_unset.output)

            # Test invalid variable name
            res_invalid = probe_env_var("invalid-name-with-dashes!")
            self.assertFalse(res_invalid.executed)
            self.assertFalse(res_invalid.success)
        finally:
            os.environ.pop(test_key, None)

    def test_probe_git_remote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Not a git repo
            self.assertIsNone(probe_git_remote("origin", tmpdir))

            subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", "https://github.com/example/repo.git"], cwd=tmpdir, check=True, capture_output=True)

            res = probe_git_remote("origin", tmpdir)
            self.assertIsNotNone(res)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)
            self.assertEqual(res.output, "https://github.com/example/repo.git")

            res_missing = probe_git_remote("upstream", tmpdir)
            self.assertIsNotNone(res_missing)
            self.assertTrue(res_missing.executed)
            self.assertFalse(res_missing.success)
            self.assertIn("not found", res_missing.output)

            # Invalid / flag injection rejection
            self.assertIsNone(probe_git_remote("-v", tmpdir))
            self.assertIsNone(probe_git_remote("origin; rm -rf /", tmpdir))

    def test_run_active_probes_for_dir_remote_and_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", "https://github.com/test/repo.git"], cwd=tmpdir, check=True, capture_output=True)
            test_dir = os.path.join(tmpdir, "my_export_folder")
            os.makedirs(test_dir)

            os.environ["HERMES_PROBE_TEST_ENV"] = "1"
            try:
                draft = f"Created directory {test_dir}. Remote origin is set. Environment variable HERMES_PROBE_TEST_ENV is set."
                probes = run_active_probes_for_draft(draft, tmpdir)
                commands = [p.command for p in probes]
                self.assertTrue(any(f"test -d {test_dir}" in c for c in commands))
                self.assertTrue(any("remote get-url origin" in c for c in commands))
                self.assertTrue(any("HERMES_PROBE_TEST_ENV" in c for c in commands))

                # Negated claims should not be probed
                neg_draft = f"I did not create directory {test_dir} and env var HERMES_PROBE_TEST_ENV is not set."
                neg_probes = run_active_probes_for_draft(neg_draft, tmpdir)
                neg_cmds = [p.command for p in neg_probes]
                self.assertFalse(any(f"test -d {test_dir}" in c for c in neg_cmds))
                self.assertFalse(any("HERMES_PROBE_TEST_ENV" in c for c in neg_cmds))
            finally:
                os.environ.pop("HERMES_PROBE_TEST_ENV", None)

    def test_record_tool_call_records_consult_advisor(self):
        class MockCtx:
            pass

        rt = AdvisorRuntime(MockCtx())
        rt.record_tool_call(
            tool_name="consult_advisor",
            args={"task": "Refactor router", "proposed_approach": "Use map"},
            result="Looks solid, ensure backwards compatibility.",
            session_id="test_sess",
        )
        events = rt._turn_tool_events.get("test_sess", [])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tool"], "consult_advisor")
        self.assertEqual(events[0]["args"]["task"], "Refactor router")
        self.assertIn("backwards compatibility", events[0]["result"])

    def test_pr_merged_with_repo_slug_in_output(self):
        # gh pr merge with repo-qualified output should successfully match PR number
        events = [{
            "tool": "terminal",
            "args": "gh pr merge --repo AlejandroAkbal/hermes-advisor-plugin --merge --admin",
            "result": "✓ Merged pull request AlejandroAkbal/hermes-advisor-plugin#23 (main).",
        }]
        v = audit_draft_ledger("Merged PR #23 successfully.", tool_events=events)
        self.assertTrue(v.passed, f"Expected pass for repo-qualified merge output, got {v.reason}")

    def test_pr_merged_with_options_and_subject(self):
        # Options containing numbers like --subject 'Release 42' should not confuse the PR number 23
        events = [{
            "tool": "terminal",
            "args": "gh pr merge --subject 'Release 42' 23 --merge",
            "result": "✓ Merged pull request #23",
        }]
        v = audit_draft_ledger("Merged PR #23 successfully.", tool_events=events)
        self.assertTrue(v.passed, f"Expected pass for correct PR argument 23, got {v.reason}")

        # PR #42 should NOT be approved because the actual argument is 23
        v_wrong = audit_draft_ledger("Merged PR #42 successfully.", tool_events=events)
        self.assertFalse(v_wrong.passed)

    def test_probe_git_remote_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "https://user:secret123@github.com/example/repo.git"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )
            res = probe_git_remote("origin", tmpdir)
            self.assertIsNotNone(res)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)
            self.assertNotIn("secret123", res.output)
            self.assertIn("user:***@github.com", res.output)

    def test_bare_directory_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)
            draft = "Created directory logs."
            probes = run_active_probes_for_draft(draft, tmpdir)
            commands = [p.command for p in probes]
            self.assertTrue(any(f"test -d {logs_dir}" in c for c in commands))


if __name__ == "__main__":
    unittest.main()
