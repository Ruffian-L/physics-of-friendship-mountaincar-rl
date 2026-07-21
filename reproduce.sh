#!/usr/bin/env bash
# One-command reproduction of the main result: the 2,000-episode component
# ablation (Phase 5). Bootstraps a virtualenv with pinned dependencies if
# one doesn't exist, then runs the study headlessly.
#
#   bash reproduce.sh          # full run: 5 configs × 3 seeds × 2000 episodes (~45–90 min)
#   bash reproduce.sh --quick  # smoke test: 5 configs × 1 seed × 200 episodes (~5 min)
#
# Expected full-run outcome (mean win rate over seeds 42/137/271):
#   full / no_tda / no_splats ≈ 77–78%   |   no_bridge ≈ 31%   |   baseline ≈ 25%
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}

if [ ! -x .venv/bin/python3 ]; then
    echo "→ Creating .venv and installing pinned dependencies..."
    "$PYTHON" -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

ARGS=()
if [ "${1:-}" = "--quick" ]; then
    ARGS=(--episodes 200 --seeds 1)
    echo "→ Quick smoke test: 200 episodes × 1 seed per config"
else
    echo "→ Full ablation: 2000 episodes × 3 seeds per config (expect ~45–90 min)"
fi

cd src
MPLBACKEND=Agg ../.venv/bin/python3 -m experiments.ablation_study ${ARGS[@]+"${ARGS[@]}"}

echo
echo "Done. Results (JSON + PNG) are in results/."
