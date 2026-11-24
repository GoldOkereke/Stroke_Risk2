import json
import logging as std_logging
from contextvars import ContextVar
from pathlib import Path

from .config import settings

# Define a request_id ContextVar (default "-" for non-request logs)
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Create a JSON formatter that outputs {"time": ..., "level": ..., "message": ..., "request_id": ...}
class JsonFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        log_entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        return json.dumps(log_entry)


# Make the JSON formatter read request_id from the ContextVar on every log call
# This is done in JsonFormatter.format via request_id_var.get() for each record.

# Export a logger = logging.getLogger("app")
logger = std_logging.getLogger("app")

# Set the log level from settings.log_level
logger.setLevel(settings.log_level)

# Attach two handlers:

# Console handler → prints to terminal
console_handler = std_logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)

# File handler → writes to logs/app.log
log_path = Path(settings.log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)
file_handler = std_logging.FileHandler(log_path, encoding="utf-8")
file_handler.setFormatter(JsonFormatter())
logger.addHandler(file_handler)

# What is a ContextVar?
# Think of it as a "thread-local variable" that works with async code. When a request comes in, you set request_id_var.set("abc-123"). Every log call inside that request automatically includes "abc-123" — even if you're deep in a call stack. When the request ends, you reset it.

# Search terms:

# python-json-logger custom formatter example

# contextvars ContextVar get set python

# logging RotatingFileHandler (optional, but good practice)
