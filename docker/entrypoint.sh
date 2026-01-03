#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Waiting for Postgres..."

python - <<'PY'
import os, time
import psycopg2

host = os.getenv("POSTGRES_HOST", "postgres")
port = int(os.getenv("POSTGRES_PORT", "5432"))
db   = os.getenv("POSTGRES_DB", "metlife")
user = os.getenv("POSTGRES_USER", "metlife")
pwd  = os.getenv("POSTGRES_PASSWORD", "metlife")

for i in range(60):
    try:
        conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=pwd)
        conn.close()
        print("[entrypoint] Postgres is ready")
        break
    except Exception as e:
        time.sleep(2)
else:
    raise SystemExit("[entrypoint] Postgres not ready after retries")
PY

echo "[entrypoint] Step 2: create DB tables + load CSV into training_dataset"
python -m scripts.create_db_and_load_data

echo "[entrypoint] Step 3: training pipeline"
python -m scripts.training

echo "[entrypoint] Step 4: scoring pipeline"
python -m scripts.scoring

echo "[entrypoint] Done ✅"
