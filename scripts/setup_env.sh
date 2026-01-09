#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv-astrofd}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -d "${ENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ "${INSTALL_CUPY:-0}" = "1" ]; then
  pip install cupy
fi

cat > scripts/env.sh <<'EOF'
#!/usr/bin/env bash
# Source this file before running for consistent threading behavior.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplcache}"
EOF
chmod +x scripts/env.sh

mkdir -p .mplcache

echo "Environment ready at ${ENV_DIR}."
echo "Source scripts/env.sh to apply thread and Matplotlib cache settings."
