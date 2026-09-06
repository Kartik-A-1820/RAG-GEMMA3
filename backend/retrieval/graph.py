import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from backend.retrieval.fusion import RankedItem


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+\-]*(?:\s+[A-Z][A-Za-z0-9+\-]*){0,3}|[A-Za-z]+(?:RAG|LLM|BM25|API|GPU|CPU|CUDA|FAISS|Chroma|GraphRAG)[A-Za-z0-9+\-]*)\b"
)
DOMAIN_TERM_PATTERN = re.compile(r"\b[a-z][a-z0-9\-]{4,}\b")
TECH_TERMS = {
    "api", "bm25", "chroma", "cuda", "dense retrieval", "embedding", "faiss",
    "gemma", "graph", "graphrag", "hybrid rag", "knowledge graph", "llm",
    "local model", "reranker", "retrieval", "rrf", "sparse retrieval",
    "vector store",
}
STOP_ENTITIES = {
    "about", "after", "again", "along", "also", "because", "before", "between",
    "could", "from", "into", "other", "should", "their", "there", "these",
    "thing", "those", "through", "using", "where", "which", "while", "would",
    "the", "this", "that", "it", "and", "or", "a", "an",
}


@dataclass(frozen=True)
class GraphHit:
    doc_id: str
    score: float
    matched_entities: list[str]


def normalize_entity(entity: str) -> str:
    cleaned = re.sub(r"\s+", " ", entity.strip().lower())
    return cleaned.strip(".,:;()[]{}")


def extract_entities(text: str, max_entities: int = 24) -> list[str]:
    entities = []
    seen = set()
    lowered = text.lower()

    for term in TECH_TERMS:
        if term in lowered and term not in seen:
            seen.add(term)
            entities.append(term)

    for match in ENTITY_PATTERN.findall(text):
        entity = normalize_entity(match)
        if len(entity) < 3 or entity in STOP_ENTITIES or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)

    for match in DOMAIN_TERM_PATTERN.findall(lowered):
        entity = normalize_entity(match)
        if entity in STOP_ENTITIES or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)

    return entities[:max_entities]


class KnowledgeGraphIndex:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.entity_to_docs: dict[str, set[str]] = defaultdict(set)
        self.doc_to_entities: dict[str, set[str]] = defaultdict(set)
        self.edges: Counter[tuple[str, str]] = Counter()
        self.doc_metadata: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.entity_to_docs = defaultdict(set, {
            entity: set(doc_ids) for entity, doc_ids in payload.get("entity_to_docs", {}).items()
        })
        self.doc_to_entities = defaultdict(set, {
            doc_id: set(entities) for doc_id, entities in payload.get("doc_to_entities", {}).items()
        })
        self.edges = Counter({
            tuple(edge.split("||", 1)): weight for edge, weight in payload.get("edges", {}).items()
        })
        self.doc_metadata = payload.get("doc_metadata", {})

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_to_docs": {
                entity: sorted(doc_ids) for entity, doc_ids in sorted(self.entity_to_docs.items())
            },
            "doc_to_entities": {
                doc_id: sorted(entities) for doc_id, entities in sorted(self.doc_to_entities.items())
            },
            "edges": {
                "||".join(edge): weight for edge, weight in sorted(self.edges.items())
            },
            "doc_metadata": self.doc_metadata,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build(self, documents: dict[str, Document]) -> None:
        self.entity_to_docs = defaultdict(set)
        self.doc_to_entities = defaultdict(set)
        self.edges = Counter()
        self.doc_metadata = {}

        for doc_id, document in documents.items():
            entities = extract_entities(document.page_content)
            document.metadata["graph_entities"] = entities
            self.doc_metadata[doc_id] = {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "chunk_index": document.metadata.get("chunk_index"),
            }
            for entity in entities:
                self.entity_to_docs[entity].add(doc_id)
                self.doc_to_entities[doc_id].add(entity)
            for index, left in enumerate(entities):
                for right in entities[index + 1:]:
                    edge = tuple(sorted((left, right)))
                    self.edges[edge] += 1
        self.save()

    def add_documents(self, documents: dict[str, Document]) -> None:
        merged = {
            doc_id: Document(page_content="", metadata={})
            for doc_id in self.doc_to_entities
        }
        merged.update(documents)
        self.build(merged)

    def search(self, query: str, k: int = 5) -> list[GraphHit]:
        query_entities = extract_entities(query, max_entities=12)
        if not query_entities:
            return []

        scores: defaultdict[str, float] = defaultdict(float)
        matched: defaultdict[str, set[str]] = defaultdict(set)
        for entity in query_entities:
            direct_docs = self.entity_to_docs.get(entity, set())
            for doc_id in direct_docs:
                scores[doc_id] += 2.0
                matched[doc_id].add(entity)

            for edge, weight in self.edges.items():
                if entity not in edge:
                    continue
                neighbor = edge[0] if edge[1] == entity else edge[1]
                for doc_id in self.entity_to_docs.get(neighbor, set()):
                    scores[doc_id] += 0.25 * math.log1p(weight)
                    matched[doc_id].add(neighbor)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [
            GraphHit(doc_id=doc_id, score=score, matched_entities=sorted(matched[doc_id]))
            for doc_id, score in ranked
        ]

    def ranked_items(self, query: str, k: int) -> list[RankedItem]:
        return [RankedItem(doc_id=hit.doc_id, score=hit.score) for hit in self.search(query, k)]

    def stats(self) -> dict:
        return {
            "entities": len(self.entity_to_docs),
            "relations": len(self.edges),
            "documents": len(self.doc_to_entities),
        }

    def snapshot(self, limit: int = 40) -> dict:
        entity_weights = Counter({
            entity: len(doc_ids) for entity, doc_ids in self.entity_to_docs.items()
        })
        top_entities = entity_weights.most_common(limit)
        allowed = {entity for entity, _weight in top_entities}
        edges = [
            {"source": left, "target": right, "weight": weight}
            for (left, right), weight in self.edges.most_common(limit * 2)
            if left in allowed and right in allowed
        ]
        return {
            "stats": self.stats(),
            "nodes": [
                {"id": entity, "label": entity, "weight": weight}
                for entity, weight in top_entities
            ],
            "edges": edges[:limit],
        }
