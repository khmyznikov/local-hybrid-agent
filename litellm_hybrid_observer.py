import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

LOG_PATH = Path(
    os.getenv(
        "HYBRID_ROUTING_LOG",
        "/home/gkhmyznikov/litellm-hybrid/routing-events.jsonl",
    )
)
_LOCK = threading.Lock()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _write(record: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        LOG_PATH.touch(mode=0o600, exist_ok=True)
        LOG_PATH.chmod(0o600)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


class HybridRoutingObserver(CustomLogger):
    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        litellm_params = _as_dict(kwargs.get("litellm_params"))
        metadata = _as_dict(litellm_params.get("metadata"))
        routing = _as_dict(metadata.get("routing_decision"))
        usage = _as_dict(getattr(response_obj, "usage", None))
        hidden = _as_dict(getattr(response_obj, "_hidden_params", None))
        duration = None
        if start_time is not None and end_time is not None:
            try:
                duration = (end_time - start_time).total_seconds()
            except (AttributeError, TypeError):
                pass
        _write(
            {
                "status": "success",
                "requested_model": kwargs.get("model"),
                "routed_model": (
                    routing.get("routed_model")
                    or hidden.get("router_model_name")
                    or getattr(response_obj, "router_model_name", None)
                ),
                "routing_cause": routing.get("cause"),
                "tier": routing.get("tier"),
                "routing_score": routing.get("score"),
                "routing_signals": routing.get("signals"),
                "response_model": getattr(response_obj, "model", None),
                "response_id": getattr(response_obj, "id", None),
                "duration_seconds": duration,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        error = kwargs.get("exception") or response_obj
        _write(
            {
                "status": "failure",
                "requested_model": kwargs.get("model"),
                "error_type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
            }
        )


hybrid_observer = HybridRoutingObserver()
