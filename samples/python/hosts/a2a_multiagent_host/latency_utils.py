import time
from dataclasses import dataclass
from typing import Any, Awaitable, Tuple


@dataclass
class CallTiming:
    label: str
    latency_ms: float


async def timed(label: str, awaitable: Awaitable[Any]) -> Tuple[Any, CallTiming]:
    """
    Time any async call and print JSON-RPC latency.
    Returns (result, CallTiming).
    """
    start = time.perf_counter()
    result = await awaitable
    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0
    timing = CallTiming(label=label, latency_ms=latency_ms)
    print(f"[JSONRPC][CALL] {label}: {latency_ms:.2f} ms")
    return result, timing
