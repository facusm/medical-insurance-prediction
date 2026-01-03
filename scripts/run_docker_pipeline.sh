#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts optuna

echo "[run] Building image + running end-to-end pipeline..."
docker compose up --build --abort-on-container-exit --exit-code-from app

echo "[run] Done. Check ./artifacts for outputs."
