"""Unit tests for structured JSON logging."""

import json
import logging

from src.common.logger import JSONFormatter, get_logger, setup_logging


def test_json_formatter_structure():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-12345"
    record.user = "test_analyst"

    formatted_str = formatter.format(record)
    log_data = json.loads(formatted_str)

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test_logger"
    assert log_data["message"] == "Test log message"
    assert log_data["request_id"] == "req-12345"
    assert log_data["user"] == "test_analyst"
    assert "timestamp" in log_data


def test_setup_logging_and_get_logger():
    setup_logging(log_level="DEBUG", json_format=True)
    logger = get_logger("bds39.test")
    assert logger.level == logging.NOTSET # Inherits from root logger
    assert logging.getLogger().level == logging.DEBUG
