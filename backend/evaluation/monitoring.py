import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MetricsMonitor:
    max_events: int = 500
    ingest_events: deque[dict] = field(default_factory=deque)
    query_events: deque[dict] = field(default_factory=deque)
    errors: deque[dict] = field(default_factory=deque)

    def record_ingest(self, source: str, chunks: int, latency_ms: float, duplicate: bool = False) -> None:
        self._append(self.ingest_events, {
            "ts": time.time(),
            "source": source,
            "chunks": chunks,
            "latency_ms": round(latency_ms, 2),
            "duplicate": duplicate,
        })

    def record_query(
        self,
        query: str,
        latency_ms: float,
        dense_count: int,
        sparse_count: int,
        graph_count: int,
        fused_count: int,
        returned_count: int,
    ) -> None:
        self._append(self.query_events, {
            "ts": time.time(),
            "query": query[:160],
            "latency_ms": round(latency_ms, 2),
            "dense_count": dense_count,
            "sparse_count": sparse_count,
            "graph_count": graph_count,
            "fused_count": fused_count,
            "returned_count": returned_count,
        })

    def record_error(self, route: str, error: str) -> None:
        self._append(self.errors, {
            "ts": time.time(),
            "route": route,
            "error": error[:500],
        })

    def snapshot(self, graph_stats: dict | None = None) -> dict:
        query_latencies = [event["latency_ms"] for event in self.query_events]
        ingest_latencies = [event["latency_ms"] for event in self.ingest_events]
        return {
            "ingest": {
                "count": len(self.ingest_events),
                "total_chunks": sum(event["chunks"] for event in self.ingest_events),
                "duplicates": sum(1 for event in self.ingest_events if event["duplicate"]),
                "avg_latency_ms": self._average(ingest_latencies),
            },
            "query": {
                "count": len(self.query_events),
                "avg_latency_ms": self._average(query_latencies),
                "p95_latency_ms": self._percentile(query_latencies, 0.95),
                "avg_dense_count": self._average([event["dense_count"] for event in self.query_events]),
                "avg_sparse_count": self._average([event["sparse_count"] for event in self.query_events]),
                "avg_graph_count": self._average([event["graph_count"] for event in self.query_events]),
                "avg_fused_count": self._average([event["fused_count"] for event in self.query_events]),
            },
            "graph": graph_stats or {},
            "errors": {
                "count": len(self.errors),
                "recent": list(self.errors)[-5:],
            },
            "recent_queries": list(self.query_events)[-10:],
        }

    def reset(self) -> None:
        self.ingest_events.clear()
        self.query_events.clear()
        self.errors.clear()

    def _append(self, target: deque, event: dict) -> None:
        target.append(event)
        while len(target) > self.max_events:
            target.popleft()

    def _average(self, values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
        return round(ordered[index], 2)
