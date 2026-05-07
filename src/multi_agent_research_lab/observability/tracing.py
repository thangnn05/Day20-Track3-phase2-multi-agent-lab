"""Tracing hooks.

Provides a provider-agnostic trace_span context manager with JSON export support.
Can be extended to integrate with LangSmith, Langfuse, or OpenTelemetry.
"""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# In-memory trace store for the current process
_trace_log: list[dict[str, Any]] = []


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[dict[str, Any]]:
    """Context manager that records timing and metadata for a named span.

    Usage::

        with trace_span("researcher_agent", {"query": q}) as span:
            # ... do work ...
            span["attributes"]["extra_key"] = value
        # span["duration_seconds"] is now set
    """

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception:
        span["status"] = "error"
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        _trace_log.append(span)
        logger.debug("Span [%s] completed in %.3fs (status=%s)",
                      name, span["duration_seconds"], span["status"])


def get_trace_log() -> list[dict[str, Any]]:
    """Return all recorded spans from this process."""
    return list(_trace_log)


def clear_trace_log() -> None:
    """Reset the in-memory trace log."""
    _trace_log.clear()


def export_trace_json(path: str | Path = "reports/trace.json") -> Path:
    """Write the accumulated trace log to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_trace_log, indent=2, default=str), encoding="utf-8")
    logger.info("Trace exported to %s (%d spans)", out, len(_trace_log))
    return out
