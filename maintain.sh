#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PY="${PY:-${VENV_DIR}/bin/python}"
PIP="${PIP:-${VENV_DIR}/bin/pip}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  upgrade       Upgrade pip + all deps (best-effort)
  freeze        Write exact versions to requirements.txt
  lint          Run ruff (if installed)
  fmt           Run black + ruff --fix (if installed)
  clean         Remove caches and results/*
  smoke         Quick smoke test (1 epoch CNN + Hybrid)
  backup        Tar.gz results/ with timestamp
  check         Print env + torch + cuda info
  recreate      Wipe and recreate .venv, reinstall deps
EOF
}

cmd="${1:-}"; [[ -z "$cmd" ]] && { usage; exit 1; }

req_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: venv Python not found at $PY"
    echo "Tip: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
  fi
}

run_black() {
  if [[ -x "$VENV_DIR/bin/black" ]]; then "$VENV_DIR/bin/black" "$@"; elif command -v black >/dev/null 2>&1; then black "$@"; else echo "black not installed"; return 1; fi
}

run_ruff() {
  if [[ -x "$VENV_DIR/bin/ruff" ]]; then "$VENV_DIR/bin/ruff" "$@"; elif command -v ruff >/dev/null 2>&1; then ruff "$@"; else echo "ruff not installed"; return 1; fi
}

case "$cmd" in
  upgrade)
    req_venv
    "$PIP" install --upgrade pip wheel setuptools || true
    if [[ -f requirements.txt ]]; then
      "$PIP" install --upgrade -r requirements.txt || true
    fi
    echo "Upgrades attempted. Consider: ./maintain.sh freeze"
    ;;

  freeze)
    req_venv
    "$PIP" freeze | sed '/^-e /d' > requirements.txt
    echo "requirements.txt updated."
    ;;

  lint)
    run_ruff check src || true
    ;;

  fmt)
    run_black src || true
    run_ruff check --fix src || true
    ;;

  clean)
    rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache
    rm -rf results/* 2>/dev/null || true
    echo "Cleaned caches and results."
    ;;

  smoke)
    req_venv
    "$PY" src/train.py --model cnn --epochs 1
    "$PY" src/train.py --model hybrid --epochs 1 --timesteps 5
    echo "Smoke tests done."
    ;;

  backup)
    ts="$(date +%Y%m%d_%H%M%S)"
    tar czf "results_${ts}.tar.gz" results/ 2>/dev/null || echo "No results/ to archive."
    ls -lh results_*.tar.gz 2>/dev/null || true
    ;;

  check)
    req_venv
    "$PY" - <<'PYINFO'
import sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PYINFO
    ;;

  recreate)
    rm -rf "$VENV_DIR"
    /usr/bin/python3.11 -m venv "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    if [[ -f requirements.txt ]]; then
      pip install -r requirements.txt
    else
      pip install torch torchvision snntorch matplotlib tqdm numpy
    fi
    echo "Recreated venv and reinstalled deps."
    ;;

  *)
    usage; exit 1 ;;
esac

