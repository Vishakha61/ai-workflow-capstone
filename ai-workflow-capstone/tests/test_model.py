import pandas as pd
from app.model import FEATURES, train_and_evaluate

def test_training_compares_multiple_models_and_saves_artifacts(tmp_path):
    data = pd.read_csv("data/country_business_data.csv")
    result = train_and_evaluate(data, artifact_dir=tmp_path)
    assert result["selected_model"] in {"linear_regression", "random_forest"}
    assert {"baseline_dummy", "linear_regression", "random_forest"} <= set(result["models"])
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "model_metrics.json").exists()
    assert (tmp_path / "model_vs_baseline.png").exists()

def test_missing_features_are_rejected():
    data = pd.DataFrame({"country": ["A"] * 10, "business_index": list(range(10))})
    try:
        train_and_evaluate(data)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for missing features")
