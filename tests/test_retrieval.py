from langchain_core.documents import Document

from backend.config import Settings
from backend.evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from backend.evaluation.run_retrieval_eval import evaluate_rows
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.fusion import RankedItem, reciprocal_rank_fusion
from backend.retrieval.rerankers import NoOpReranker


class FakeVectorStore:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []

    def get(self, include=None):
        return {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

    def add_texts(self, texts, metadatas, ids):
        self.ids.extend(ids)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

    def similarity_search(self, query, k):
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(self.documents, self.metadatas)
        ][:k]


def test_bm25_returns_lexical_match_first():
    index = BM25Index()
    index.build({
        "a": "alpha beta gamma",
        "b": "invoice payment refund policy",
        "c": "graph neural retrieval",
    })

    hits = index.search("refund invoice", k=2)

    assert hits[0].doc_id == "b"
    assert hits[0].score > 0


def test_rrf_combines_dense_and_sparse_rankings():
    fused = reciprocal_rank_fusion([
        [RankedItem("dense-only", 1.0), RankedItem("shared", 0.5)],
        [RankedItem("shared", 2.0), RankedItem("sparse-only", 1.0)],
    ])

    assert fused[0].doc_id == "shared"
    assert {item.doc_id for item in fused} == {"dense-only", "shared", "sparse-only"}


def test_noop_reranker_preserves_order_and_limit():
    docs = [Document(page_content=str(i)) for i in range(3)]

    assert NoOpReranker().rerank("query", docs, limit=2) == docs[:2]


def test_basic_eval_metrics():
    relevant = {"a", "c"}
    retrieved = ["b", "a", "c"]

    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert mean_reciprocal_rank(relevant, retrieved) == 0.5


def test_eval_runner_aggregates_rows():
    rows = [
        {"relevant_ids": ["a"], "retrieved_ids": ["b", "a"]},
        {"relevant_ids": ["c"], "retrieved_ids": ["c", "d"]},
    ]

    result = evaluate_rows(rows, k=2)

    assert result["queries"] == 2
    assert result["recall@2"] == 1.0
    assert result["mrr"] == 0.75


def test_hybrid_retriever_adds_stable_chunk_ids_and_retrieves():
    settings = Settings(dense_k=2, sparse_k=2, fused_k=2)
    vectorstore = FakeVectorStore()
    retriever = HybridRetriever(vectorstore, settings, reranker=NoOpReranker())
    docs = [
        Document(page_content="refund invoice policy", metadata={"source": "a.txt", "chunk_index": 0}),
        Document(page_content="graph retrieval notes", metadata={"source": "b.txt", "chunk_index": 0}),
    ]

    added, count = retriever.add_documents(docs, file_hash="hash-1")
    result = retriever.retrieve("refund invoice")

    assert added is True
    assert count == 2
    assert all(metadata.get("chunk_id") for metadata in vectorstore.metadatas)
    assert result.documents[0].metadata["source"] == "a.txt"
    assert result.sparse_count >= 1
