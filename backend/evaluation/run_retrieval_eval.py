import argparse
import json
from pathlib import Path

from backend.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_rows(rows: list[dict], k: int) -> dict:
    recalls = []
    precisions = []
    reciprocal_ranks = []
    hit_rates = []
    ndcgs = []
    for row in rows:
        relevant_ids = set(row.get("relevant_ids", []))
        retrieved_ids = row.get("retrieved_ids", [])
        recalls.append(recall_at_k(relevant_ids, retrieved_ids, k))
        precisions.append(precision_at_k(relevant_ids, retrieved_ids, k))
        reciprocal_ranks.append(mean_reciprocal_rank(relevant_ids, retrieved_ids))
        hit_rates.append(hit_rate_at_k(relevant_ids, retrieved_ids, k))
        ndcgs.append(ndcg_at_k(relevant_ids, retrieved_ids, k))

    total = len(rows) or 1
    return {
        "queries": len(rows),
        f"recall@{k}": sum(recalls) / total,
        f"precision@{k}": sum(precisions) / total,
        f"hit_rate@{k}": sum(hit_rates) / total,
        f"ndcg@{k}": sum(ndcgs) / total,
        "mrr": sum(reciprocal_ranks) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval results from a local JSONL file.")
    parser.add_argument("path", type=Path, help="JSONL with relevant_ids and retrieved_ids fields.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate_rows(load_jsonl(args.path), args.k), indent=2))


if __name__ == "__main__":
    main()
