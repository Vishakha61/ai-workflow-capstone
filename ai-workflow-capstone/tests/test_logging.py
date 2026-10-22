from app.logging_config import configure_logging

def test_logger_writes_to_isolated_file(tmp_path):
    log_file = tmp_path / "isolated" / "test.log"
    logger = configure_logging(log_file)
    logger.info("unit-test-log-entry")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    assert "unit-test-log-entry" in log_file.read_text(encoding="utf-8")
