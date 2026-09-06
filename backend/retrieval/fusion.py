from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    doc_id: str
    score: float


def reciprocal_rank_fusion(rankings: list[list[RankedItem]], k: int = 60) -> list[RankedItem]:
    fused_scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused_scores[item.doc_id] += 1.0 / (k + rank)

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [RankedItem(doc_id=doc_id, score=score) for doc_id, score in ordered]
