import os
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import api
from app.model import train_and_evaluate

@pytest.fixture()
def isolated_app(tmp_path, monkeypatch):
    """Use temporary data/model/log files so tests never touch production artifacts."""
    data_path = tmp_path / "test_data.csv"
    model_path = tmp_path / "test_model.joblib"
    log_path = tmp_path / "test_app.log"
    source = pd.read_csv("data/country_business_data.csv")
    source.to_csv(data_path, index=False)
    train_and_evaluate(source, artifact_dir=tmp_path / "artifacts")
    import joblib
    joblib.dump(joblib.load(tmp_path / "artifacts" / "model.joblib"), model_path)
    monkeypatch.setattr(api, "DATA_PATH", data_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api, "logger", api.configure_logging(log_path))
    monkeypatch.setattr(api, "_request_count", 0)
    monkeypatch.setattr(api, "_total_latency_ms", 0.0)
    monkeypatch.setattr(api, "_max_latency_ms", 0.0)
    return TestClient(api.app), log_path
