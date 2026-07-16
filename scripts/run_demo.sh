#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
PYTEST_BIN="$VENV_DIR/bin/pytest"
MODE="${1:-serve}"

ensure_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
}

install_deps() {
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

prepare_env() {
  if [[ ! -f "$ROOT_DIR/.env" && -f "$ROOT_DIR/.env.example" ]]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  fi
}

run_recommendation_demo() {
  "$PYTHON_BIN" -m app.cli run-task \
    --task-name movie_homepage \
    --task-type top_k_recommendation \
    --dataset-name movielens_sample \
    --description "Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu." \
    --top-k 5 \
    --constraints "implicit feedback" "local cpu" \
    --candidate-count 3
}

run_churn_demo() {
  "$PYTHON_BIN" -m app.cli run-task \
    --task-name customer_churn \
    --task-type tabular_classification \
    --dataset-name churn_sample \
    --description "Build a customer churn classification workflow. Use local cpu, compare baseline and score-based methods, and optimize f1." \
    --top-k 5 \
    --constraints "local cpu" \
    --candidate-count 2
}

ensure_venv
install_deps
prepare_env

case "$MODE" in
  serve)
    exec "$PYTHON_BIN" -m app.cli serve --host 127.0.0.1 --port "${PORT:-8000}"
    ;;
  recommend)
    run_recommendation_demo
    ;;
  churn)
    run_churn_demo
    ;;
  test)
    exec "$PYTEST_BIN"
    ;;
  *)
    echo "Usage: ./scripts/run_demo.sh [serve|recommend|churn|test]" >&2
    exit 1
    ;;
esac
