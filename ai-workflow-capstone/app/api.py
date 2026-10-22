from pathlib import Path
from time import perf_counter
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.logging_config import configure_logging
from app.model import FEATURES

app = FastAPI(title="Country Business Prediction API", version="1.0.0")
logger = configure_logging()

DATA_PATH = Path("data/country_business_data.csv")
MODEL_PATH = Path("artifacts/model.joblib")
_request_count = 0
_total_latency_ms = 0.0
_max_latency_ms = 0.0

def load_data():
    return pd.read_csv(DATA_PATH)

def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python scripts/train_model.py` first.")
    return joblib.load(MODEL_PATH)

@app.middleware("http")
async def record_performance(request, call_next):
    global _request_count, _total_latency_ms, _max_latency_ms
    started = perf_counter()
    response = await call_next(request)
    elapsed = (perf_counter() - started) * 1000
    _request_count += 1
    _total_latency_ms += elapsed
    _max_latency_ms = max(_max_latency_ms, elapsed)
    logger.info("method=%s path=%s status=%s latency_ms=%.2f",
                request.method, request.url.path, response.status_code, elapsed)
    response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.get("/metrics")
def metrics():
    avg = _total_latency_ms / _request_count if _request_count else 0.0
    return {
        "request_count": _request_count,
        "average_latency_ms": round(avg, 2),
        "max_latency_ms": round(_max_latency_ms, 2),
    }

@app.get("/predict")
def predict(country: str):
    try:
        data = load_data()
        matches = data[data["country"].str.casefold() == country.strip().casefold()]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Country '{country}' not found")
        bundle = load_bundle()
        prediction = float(bundle["model"].predict(matches[FEATURES])[0])
        logger.info("prediction country=%s value=%.3f", country, prediction)
        return {"country": matches.iloc[0]["country"], "predicted_business_index": round(prediction, 3),
                "model": bundle.get("model_name", "unknown")}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

@app.get("/predict/all")
def predict_all():
    try:
        data = load_data()
        bundle = load_bundle()
        predictions = bundle["model"].predict(data[FEATURES])
        rows = [
            {"country": country, "predicted_business_index": round(float(value), 3)}
            for country, value in zip(data["country"], predictions)
        ]
        return {
            "count": len(rows),
            "countries": rows,
            "combined_average": round(sum(x["predicted_business_index"] for x in rows) / len(rows), 3),
            "model": bundle.get("model_name", "unknown"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Aggregate prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
