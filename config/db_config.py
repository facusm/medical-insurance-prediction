import os

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "metlife"),
    "user": os.getenv("POSTGRES_USER", "metlife"),
    "password": os.getenv("POSTGRES_PASSWORD", "metlife"),
}
