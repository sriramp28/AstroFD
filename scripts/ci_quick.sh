#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplcache}"
mkdir -p "${MPLCONFIGDIR}"

python tools/run_validation_suite.py --quick
