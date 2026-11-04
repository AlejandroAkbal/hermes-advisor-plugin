# Advisor self-improvement runbook

The 8-hour cron runs this loop. Work only inside this repo; never edit
`~/.hermes/hermes-agent`. One small, generic improvement per run.

## Loop

1. **Measure.** `scripts/self_improve.py --hours 24` with
   `PYTHONPATH=~/.hermes/hermes-agent:~/.hermes/plugins`. Read the approval rate,
   the rejection reasons and the audit latency (median/p90).
2. **Pick one failure mode.** Target the leading recognized cause, prefer
   deterministic auto-remediation (extend `prober.py`) over prompt wording.
3. **Change.** Small and generic. Diff under ~150 lines. No new dependencies.
4. **Test.** `python3 -m unittest discover -s tests`. The gate must exit 0:
   chain with `&&`, never pipe it through `head`/`tail` without `set -o pipefail`.
5. **Ship.** Feature branch → push → `gh pr create --repo
   AlejandroAkbal/hermes-advisor-plugin` → CodeRabbit
   (`coderabbit review --agent --base-commit main --committed`). CodeRabbit's
   free quota is 3 reviews/hour; if it is exhausted, say so and rely on tests.
6. **Merge.** `gh pr merge --repo AlejandroAkbal/hermes-advisor-plugin --merge
   --admin`. Full automatic, no human review.
7. **Make it live.** A merged change is dead code until the gateway loads it:
   `python3 scripts/gateway_live.py check`
   - exit 0 — the gateway started after the newest commit: live.
   - exit 3 — stale. Arm the idle-gated restart and let it fire in a gap:
     `cronjob_manage(action="create", name="Delayed gateway restart",
     schedule="in 2m", repeat=1, no_agent=True, script="gateway-live-restart.py",
     deliver="local", prompt="Restart the gateway so the merged advisor changes load.")`
     The worker detaches, polls `~/.hermes/gateway_state.json`, and only sends
     SIGUSR1 when `active_agents == 0`; it waits up to six hours and then gives
     up **without signalling** (exit 6). Never add `--no-idle` to a scheduled
     restart: that interrupts whatever is running.
   - Never report the run as done while check exits 3. Report "merged, restart
     armed" plus the PID start time and commit time the check printed.
8. **Report.** The change, the test count, the latency/approval numbers, and the
   liveness line.

## Hard rules

- Never run a gateway restart command directly from an agent turn: the terminal
  guard refuses it because SIGTERM would kill the command. Step 7 exists for that.
- Never signal the PID printed by `hermes gateway status`: that is the launchd
  osascript wrapper. SIGUSR1 kills the wrapper, launchd then SIGTERMs the real
  gateway, and the drain is skipped. `scripts/gateway_live.py` reads
  `~/.hermes/gateway.pid` instead and refuses wrapper PIDs.
- Restarts are idle-gated for a reason: the gateway's own drain caps at
  `agent.restart_after_turn_timeout` (7200s here) and force-drains past it, which
  interrupts live turns. Signalling only when `active_agents == 0` means there is
  nothing to interrupt, so sessions never notice the restart.
- `gateway-live-restart.py` (in `~/.hermes/scripts/`) runs the idle-gated restart
  detached, so it survives the gateway it is restarting.
- Sessions persist in `state.db`; a restart costs at most one interrupted turn,
  and with idle-gating even that does not happen.
