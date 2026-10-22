from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "population_millions",
    "gdp_per_capita_usd",
    "internet_penetration_pct",
    "inflation_pct",
    "unemployment_pct",
]
TARGET = "business_index"

def train_and_evaluate(data: pd.DataFrame, artifact_dir="artifacts", random_state=42):
    """Train several regressors, compare them with a mean baseline, save best model."""
    required = ["country", *FEATURES, TARGET]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = data[required].dropna()
    if len(clean) < 8:
        raise ValueError("At least 8 complete rows are required to train the demo model")

    X, y = clean[FEATURES], clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    models = {
        "baseline_dummy": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, max_depth=5
        ),
    }
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
            "r2": float(r2_score(y_test, pred)),
        }
        fitted[name] = model

    candidates = [n for n in models if n != "baseline_dummy"]
    best_name = min(candidates, key=lambda n: results[n]["mae"])
    # Refit chosen model on all available rows for the prediction API.
    best_model = fitted[best_name]
    best_model.fit(X, y)

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "features": FEATURES, "model_name": best_name}, out / "model.joblib")
    import json
    with (out / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "models": results}, f, indent=2)

    # A readable comparison plot, including baseline and both candidate models.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    maes = [results[n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, maes)
    ax.set_title("Model comparison: mean absolute error (lower is better)")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "model_vs_baseline.png", dpi=140)
    plt.close(fig)
    return {"selected_model": best_name, "models": results}
