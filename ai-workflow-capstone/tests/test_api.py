def test_health_and_country_prediction(isolated_app):
    client, _ = isolated_app
    assert client.get("/health").status_code == 200
    response = client.get("/predict", params={"country": "India"})
    assert response.status_code == 200
    body = response.json()
    assert body["country"] == "India"
    assert isinstance(body["predicted_business_index"], float)

def test_unknown_country_returns_404(isolated_app):
    client, _ = isolated_app
    assert client.get("/predict", params={"country": "Atlantis"}).status_code == 404

def test_all_country_predictions_and_metrics(isolated_app):
    client, _ = isolated_app
    response = client.get("/predict/all")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 1
    assert len(body["countries"]) == body["count"]
    assert isinstance(body["combined_average"], float)
    metrics = client.get("/metrics").json()
    assert metrics["request_count"] >= 2
    assert metrics["average_latency_ms"] >= 0
