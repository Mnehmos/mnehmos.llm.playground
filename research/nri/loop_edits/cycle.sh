#!/bin/bash
# advance one checkpoint and (optionally) run the agent's OBS pre-checks (never installs anything)
cd "$(dirname "$0")/.."
python run_loop.py advance | grep -E "===|acc|IIA\(M\)|pooled|copy|interaction share|route main|unit-transfer|per-unit|mult|extras|identical"
if [ -n "$1" ]; then python loop_edits/precheck_rewire.py "$(cat "$1")"; fi
