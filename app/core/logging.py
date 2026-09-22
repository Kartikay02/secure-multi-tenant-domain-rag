"""Structured logging configuration with correlation ID context tracking."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Context variable for request correlation ID
correlation_id_ctx_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Retrieve the current request correlation ID from context."""
    return correlation_id_ctx_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the request correlation ID into context."""
    correlation_id_ctx_var.set(correlation_id)


class JSONLogFormatter(logging.Formatter):
    """Production JSON log formatter for structured log aggregators with secret scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        from app.observability.sanitizer import Sanitizer
        from app.observability.tracer import get_trace_id

        raw_msg = record.getMessage()
        clean_msg = Sanitizer.sanitize_string(raw_msg)

        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": clean_msg,
            "logger": record.name,
            "correlation_id": get_correlation_id() or getattr(record, "correlation_id", ""),
            "trace_id": get_trace_id() or getattr(record, "trace_id", ""),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if available
        if record.exc_info:
            log_payload["exception"] = Sanitizer.sanitize_string(
                self.formatException(record.exc_info)
            )

        # Include custom extra attributes
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_payload["data"] = Sanitizer.sanitize_dict(record.extra_data)

        return json.dumps(log_payload, ensure_ascii=False)


class ConsoleLogFormatter(logging.Formatter):
    """Human-readable formatter with correlation ID and secret scrubbing for local development."""

    def format(self, record: logging.LogRecord) -> str:
        from app.observability.sanitizer import Sanitizer

        corr_id = get_correlation_id() or getattr(record, "correlation_id", "-")
        corr_display = f"[{corr_id}] " if corr_id != "-" else ""
        time_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        msg = Sanitizer.sanitize_string(record.getMessage())

        base_str = f"{time_str} | {record.levelname:<7} | {corr_display}{record.name}:{record.lineno} - {msg}"

        if record.exc_info:
            base_str += f"\n{Sanitizer.sanitize_string(self.formatException(record.exc_info))}"

        return base_str


def set_log_level(log_level: str) -> None:
    """Dynamically reconfigure log level threshold at runtime."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)


def setup_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """Configure root logger and formatters."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)

    if log_format.lower() == "json":
        stream_handler.setFormatter(JSONLogFormatter())
    else:
        stream_handler.setFormatter(ConsoleLogFormatter())

    root_logger.addHandler(stream_handler)

    # Adjust external noisy loggers
    for noisy_logger_name in ("uvicorn.access", "httpx", "httpcore"):
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Create and return a named logger."""
    return logging.getLogger(name)
