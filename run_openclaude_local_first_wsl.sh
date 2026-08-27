#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOCAL_HYBRID_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${LITELLM_PYTHON:-$HOME/litellm-hybrid/.venv/bin/python}"
COMMAND="${1:-run}"
shift || true

case "$COMMAND" in
    prepare|baseline|dispatch)
        if [[ "$COMMAND" == "dispatch" ]]; then
            COMMAND=run
        fi
        exec "$PYTHON" "$ROOT/openclaude_harness/dispatch.py" \
            "$COMMAND" "$@"
        ;;
    repair)
        exec "$PYTHON" "$ROOT/openclaude_harness/repair.py" "$@"
        ;;
    summarize)
        exec "$PYTHON" "$ROOT/openclaude_harness/summarize.py" "$@"
        ;;
    run)
        DISPATCH_ARGS=("$@")
        REPAIR_ARGS=()
        index=0
        while (( index < ${#DISPATCH_ARGS[@]} )); do
            argument=${DISPATCH_ARGS[$index]}
            if [[ "$argument" == "--case" || "$argument" == "--arm" ]]; then
                REPAIR_ARGS+=("$argument" "${DISPATCH_ARGS[$((index + 1))]}")
                index=$((index + 2))
            else
                index=$((index + 1))
            fi
        done
        "$PYTHON" "$ROOT/openclaude_harness/dispatch.py" run \
            "${DISPATCH_ARGS[@]}"
        "$PYTHON" "$ROOT/openclaude_harness/repair.py" \
            "${REPAIR_ARGS[@]}"
        exec "$PYTHON" "$ROOT/openclaude_harness/summarize.py"
        ;;
    *)
        echo "Usage: $0 {prepare|baseline|dispatch|repair|summarize|run} [options]" >&2
        exit 2
        ;;
esac
