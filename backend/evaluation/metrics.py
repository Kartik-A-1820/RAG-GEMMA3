def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(relevant_ids & retrieved) / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant_ids) / len(retrieved)


def mean_reciprocal_rank(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1.0 / index
    return 0.0
