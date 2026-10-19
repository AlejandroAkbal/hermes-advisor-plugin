#!/usr/bin/env bash
# Pin this plugin's Hermes plugin-catalog entry to the commit you just pushed.
#
# The catalog (hermes-agent:plugin-catalog/<name>.yaml) installs exactly one
# 40-char SHA, and every bump is a new reviewed PR (catalog admission rule 4).
# So the entry must never point at a branch, and it must never be hand-edited
# from memory: this script reads the SHA from origin/main, checks that the
# pushed manifest is actually the one the catalog will accept (version +
# provides_hooks — the two things that silently failed admission once), rewrites
# the entry, and runs the catalog validator.
#
# Usage:
#   scripts/pin-catalog-entry.sh                      # defaults below
#   scripts/pin-catalog-entry.sh --entry PATH --hermes PATH
#   scripts/pin-catalog-entry.sh --check              # verify, write nothing
#
# Exit: 0 pinned+valid, 1 nothing pushed yet / entry would be rejected, 2 bad args.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_DIR="${HERMES_DIR:-$HOME/hermes-agent}"
ENTRY=""
CHECK_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --entry)  ENTRY="${2:?--entry needs a path}"; shift 2 ;;
        --hermes) HERMES_DIR="${2:?--hermes needs a path}"; shift 2 ;;
        --check)  CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

NAME="$(sed -n 's/^name: *//p' "$PLUGIN_DIR/plugin.yaml" | head -1 | tr -d '"')"
ENTRY="${ENTRY:-$HERMES_DIR/plugin-catalog/$NAME.yaml}"
VALIDATOR="$HERMES_DIR/scripts/validate_plugin_catalog.py"

[ -f "$ENTRY" ]     || { echo "ERROR: no catalog entry at $ENTRY" >&2; exit 1; }
[ -f "$VALIDATOR" ] || { echo "ERROR: no validator at $VALIDATOR (is --hermes right?)" >&2; exit 1; }

cd "$PLUGIN_DIR"
git fetch -q origin
SHA="$(git rev-parse origin/main)"
HEAD="$(git rev-parse HEAD)"
if [ "$HEAD" != "$SHA" ]; then
    echo "ERROR: origin/main is not what you have checked out — push first." >&2
    echo "  HEAD=$(git rev-parse --short "$HEAD")  origin/main=$(git rev-parse --short "$SHA")" >&2
    exit 1
fi

# What the catalog will look at, read from the pushed commit (not the worktree,
# so uncommitted edits cannot make the check pass for code nobody can install).
MANIFEST="$(git show "$SHA:plugin.yaml")"
VERSION="$(printf '%s\n' "$MANIFEST" | sed -n 's/^version: *"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' | head -1)"
HOOKS="$(printf '%s\n' "$MANIFEST" | grep -c '^provides_hooks:' || true)"
TOOLS="$(printf '%s\n' "$MANIFEST" | grep -c '^provides_tools:' || true)"
echo "pushed:  $SHA"
echo "version: ${VERSION:-<none>}   provides_hooks:${HOOKS} provides_tools:${TOOLS}"

if [ -z "$VERSION" ]; then
    echo "ERROR: pushed plugin.yaml has no version: — catalog entries are versioned." >&2
    exit 1
fi

# The manifest must declare what register() registers, or admission fails with
# "undeclared hooks registered". Cheap proxy for that: hooks declared at all.
if grep -q 'register_hook' "$PLUGIN_DIR"/*.py 2>/dev/null && [ "$HOOKS" -lt 1 ]; then
    echo "ERROR: the plugin registers hooks but the pushed plugin.yaml declares none." >&2
    echo "       Add a top-level 'provides_hooks:' list (legacy 'hooks:' is ignored)." >&2
    exit 1
fi

CURRENT_SHA="$(sed -n 's/^sha: *//p' "$ENTRY" | head -1 | tr -d '"')"
if [ "$CURRENT_SHA" = "$SHA" ] && [ "$CHECK_ONLY" -eq 0 ]; then
    echo "entry already pinned to this commit: $ENTRY"
else
    if [ "$CHECK_ONLY" -eq 1 ]; then
        echo "would pin: ${CURRENT_SHA:-<unset>} -> $SHA"
    else
        # macOS/BSD sed -i needs an argument; GNU accepts it. Pick at runtime.
        if sed --version >/dev/null 2>&1; then INPLACE=(-i); else INPLACE=(-i ''); fi
        sed "${INPLACE[@]}" \
            -e "s|^sha: .*|sha: \"$SHA\"|" \
            -e "s|/blob/[^/]*/README.md|/blob/$SHA/README.md|" \
            "$ENTRY"
        echo "pinned: ${CURRENT_SHA:-<unset>} -> $SHA"
    fi
fi

# Validate through the repo's own runner: the validator needs ruamel.yaml, which
# lives in the Hermes dev environment, not in a bare interpreter (and macOS's
# env -S mangles the validator's own shebang, so invoke it explicitly).
cd "$HERMES_DIR"
"$HERMES_DIR/scripts/run-in-hermes-env" python3 "$VALIDATOR" "$ENTRY"

if [ "$CHECK_ONLY" -eq 0 ]; then
    echo
    echo "Next: commit the entry + open the catalog PR (rule 4 — a SHA bump is a new PR):"
    echo "  git -C $HERMES_DIR add plugin-catalog/$NAME.yaml"
    echo "  git -C $HERMES_DIR commit -m 'plugin-catalog: $NAME @ $SHA'"
fi
