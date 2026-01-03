import pandas as pd
from sqlalchemy import text
from config.db_config import DB_CONFIG
from src.db.connection import get_engine

CSV_PATH = "data/dataset.csv"

def main():
    engine = get_engine(DB_CONFIG)

    # 1. Crear tablas
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS training_dataset"))
        conn.execute(text("DROP TABLE IF EXISTS training_dataset_raw"))

        conn.execute(text("""
            CREATE TABLE training_dataset_raw (
                age INTEGER,
                sex VARCHAR(10),
                bmi NUMERIC(5,2),
                children INTEGER,
                smoker VARCHAR(5),
                region VARCHAR(20),
                charges NUMERIC(10,2)
            );
        """))

        conn.execute(text("""
            CREATE TABLE training_dataset (
                id SERIAL PRIMARY KEY,
                age INTEGER NOT NULL,
                sex VARCHAR(10) NOT NULL,
                bmi NUMERIC(5,2) NOT NULL,
                children INTEGER NOT NULL,
                smoker BOOLEAN NOT NULL,
                region VARCHAR(20) NOT NULL,
                charges NUMERIC(10,2) NOT NULL
            );
        """))

    # 2. Cargar CSV
    df = pd.read_csv(CSV_PATH)

    # 3. Insertar en staging
    df.to_sql(
        "training_dataset_raw",
        engine,
        if_exists="append",
        index=False
    )

    # 4. Transformar a tabla final
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO training_dataset (
                age, sex, bmi, children, smoker, region, charges
            )
            SELECT
                age,
                sex,
                bmi,
                children,
                smoker = 'yes' AS smoker,
                region,
                charges
            FROM training_dataset_raw;
        """))

    print("✅ Paso 2 completado: base creada y dataset cargado")

if __name__ == "__main__":
    main()
