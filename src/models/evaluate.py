import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_holdout_rmse(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {"TEST_RMSE": rmse(y_test, preds)}

def cv_rmse_for_params(
    X,
    y,
    preprocessor,
    params: dict,
    n_splits: int = 5,
    seed: int = 42,
    return_folds: bool = False,
) -> dict:
    """
    Recalcula CV RMSE (mean/std) usando los best_params.
    """

    model = LGBMRegressor(**params, random_state=seed, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rmses = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        pipeline.fit(X_tr, y_tr)
        pred = pipeline.predict(X_va)
        rmses.append(rmse(y_va, pred))

    out = {
        "CV_RMSE_mean": float(np.mean(rmses)),
        "CV_RMSE_std": float(np.std(rmses, ddof=1)),
        "CV_folds": int(n_splits),
    }

    if return_folds:
        out["CV_RMSE_folds"] = [float(x) for x in rmses]

    return out
