import numpy as np
import optuna

from optuna.samplers import TPESampler
from optuna.trial import TrialState

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def objective(trial, X, y, preprocessor, n_splits=5, seed=42):
    """
    Función objetivo para Optuna.:
    - Samplea hiperparámetros de LightGBM
    - corre K-Fold CV
    - retorna promedio RMSE
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

        "num_leaves": trial.suggest_int("num_leaves", 20, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),

        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),

        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),

        "random_state": seed,
        "n_jobs": -1,
    }
    model = LGBMRegressor(**params, verbosity=-1)


    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)

    return float(np.mean(rmses))


def train_model(
    X_train,
    y_train,
    preprocessor,
    target_trials=50,
    storage_url="sqlite:///optuna_study.db",
    study_name="lightgbm_insurance_charges",
    n_startup_trials=5,
    seed=42,
    n_splits=5,
):
    """
    Entrena un modelo LightGBM usando Optuna (TPE) con almacenamiento persistente.

    Si un estudio ya existe en el almacenamiento, se cargará.
    Solo ejecutará los ensayos restantes necesarios para alcanzar `target_trials` (contando solo los ensayos COMPLETOS).

    retorna el pipeline final (preprocesador + modelo entrenado) y el objeto estudio de Optuna.
    """

    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    remaining = max(0, target_trials - completed)

    print(
        f"[Optuna] Study='{study_name}' | COMPLETE={completed} | "
        f"Target={target_trials} | Remaining={remaining}"
    )

    if remaining > 0:
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                preprocessor,
                n_splits=n_splits,
                seed=seed,
            ),
            n_trials=remaining,
        )
    else:
        print("Estudio de Optuna ya existe y ha alcanzado el objetivo.")

    # Entrenar el modelo final con los mejores hiperparámetros
    best_params = study.best_params

    best_model = LGBMRegressor(
        **best_params,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model),
        ]
    )

    final_pipeline.fit(X_train, y_train)

    return final_pipeline, study
