import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from config.db_config import DB_CONFIG
from src.db.connection import get_engine
from src.utils.io import load_model, save_metrics

MODEL_PATH = "artifacts/best_model.joblib"

def main():
    engine = get_engine(DB_CONFIG)

    # a) leer tabla holdout (no usada en training/optuna) y samplear 10
    df_holdout = pd.read_sql("SELECT * FROM holdout_test_dataset", engine)
    df_sample = df_holdout.sample(n=10, random_state=42)

    # b) cargar modelo y predecir
    model = load_model(MODEL_PATH)

    X = df_sample.drop(columns=["id", "charges"])
    y_true = df_sample["charges"].astype(float)
    y_pred = model.predict(X)

    # c) append predicciones
    df_sample["predicted_charges"] = y_pred
    df_sample.to_sql("scoring_results", engine, if_exists="replace", index=False)

    # d) métrica final (RMSE)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    save_metrics({"SCORING_RMSE_10rows": rmse}, "artifacts/scoring_metrics.txt")

    print("✅ Punto 4 completo")
    print(f"RMSE (10 filas holdout): {rmse:.4f}")

if __name__ == "__main__":
    main()
