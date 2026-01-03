from config.db_config import DB_CONFIG
from src.db.connection import get_engine

from src.data_access.load_training_data import load_training_data
from src.features.preprocessing import preprocess_data
from src.models.train import train_model
from src.models.evaluate import cv_rmse_for_params, evaluate_holdout_rmse
from src.utils.io import save_model, save_metrics
import os


def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("optuna", exist_ok=True)
    
    optuna_storage = os.getenv("OPTUNA_STORAGE_URL", "sqlite:///optuna_study.db")

    engine = get_engine(DB_CONFIG)

    # 1) cargar datos (todos)
    df = load_training_data(engine)

    # 2) preparación de datos + hold-out test split
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Se guarda el hold-out test en DB para usarlo en scoring.py luego 
    test_df = df.loc[X_test.index, ["id", "age", "sex", "bmi", "children", "smoker", "region", "charges"]]
    test_df.to_sql("holdout_test_dataset", engine, if_exists="replace", index=False)

    # 3) busqueda de hiperparámetros y entrenamiento del modelo con los mejores hiperparámetros
    model, study = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        target_trials=50,  
        storage_url=optuna_storage,
        study_name="lightgbm_insurance_charges",
        n_startup_trials=5,
        seed=42,
        n_splits=5,
    )

    # 4) Evaluación final del modelo
    metrics = {}
    metrics["CV_RMSE_best_trial"] = float(study.best_value)  # lo que optimizó Optuna
    metrics.update(cv_rmse_for_params(X_train, y_train, preprocessor, study.best_params, n_splits=5, seed=42))
    metrics.update(evaluate_holdout_rmse(model, X_test, y_test))  

    # 5) guardado de artefactos
    save_model(model, "artifacts/best_model.joblib")
    save_metrics(metrics, "artifacts/metrics.txt")

    print("Saved: artifacts/best_model.joblib")
    print("Saved: artifacts/metrics.txt")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
