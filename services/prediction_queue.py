"""
Async scoring helper for low-latency paths (micro-batch simulation without Kafka).
Uses a thread pool via asyncio.to_thread; for production, swap for a real queue consumer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    depth: int
    processed_total: int
    avg_latency_ms: float | None


class AsyncPredictionRunner:
    """Tracks latency for async-wrapped batch scoring (queue depth stays 0 in this stub)."""

    def __init__(self) -> None:
        self._processed = 0
        self._lat_sum = 0.0

    async def run(
        self,
        scorer: Callable[[dict[str, Any]], dict[str, Any]],
        payload: dict[str, Any],
        *,
        timeout_s: float = 0.25,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            out = await asyncio.wait_for(asyncio.to_thread(scorer, payload), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning("Prediction exceeded timeout %.3fs", timeout_s)
            raise
        dt = (time.perf_counter() - t0) * 1000.0
        self._processed += 1
        self._lat_sum += dt
        out["async_latency_ms"] = round(dt, 3)
        return out

    def stats(self) -> QueueStats:
        avg = (self._lat_sum / self._processed) if self._processed else None
        return QueueStats(depth=0, processed_total=self._processed, avg_latency_ms=avg)


_runner: AsyncPredictionRunner | None = None


def get_or_create_queue() -> AsyncPredictionRunner:
    global _runner
    if _runner is None:
        _runner = AsyncPredictionRunner()
    return _runner
