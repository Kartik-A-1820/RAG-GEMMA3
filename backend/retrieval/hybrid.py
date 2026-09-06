from dataclasses import dataclass
from uuid import uuid4

from langchain_core.documents import Document

from backend.config import Settings
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.fusion import RankedItem, reciprocal_rank_fusion
from backend.retrieval.graph import KnowledgeGraphIndex
from backend.retrieval.rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    dense_count: int
    sparse_count: int
    graph_count: int
    fused_count: int
    graph_stats: dict


class HybridRetriever:
    def __init__(self, vectorstore, settings: Settings, reranker: BaseReranker | None = None) -> None:
        self.vectorstore = vectorstore
        self.settings = settings
        self.reranker = reranker or self._build_reranker(settings)
        self.bm25 = BM25Index()
        self.graph = KnowledgeGraphIndex(settings.graph_path)
        self.documents_by_id: dict[str, Document] = {}
        self.rebuild_sparse_index()

    def _build_reranker(self, settings: Settings) -> BaseReranker:
        if settings.enable_reranker and settings.reranker_model:
            return CrossEncoderReranker(settings.reranker_model)
        return NoOpReranker()

    def existing_hashes(self) -> list[dict]:
        return self.vectorstore.get().get("metadatas", [])

    def add_documents(self, documents: list[Document], file_hash: str) -> tuple[bool, int]:
        if any((metadata or {}).get("hash") == file_hash for metadata in self.existing_hashes()):
            return False, 0

        ids = [str(uuid4()) for _ in documents]
        for doc_id, document in zip(ids, documents):
            document.metadata["chunk_id"] = doc_id
        self.vectorstore.add_texts(
            texts=[document.page_content for document in documents],
            metadatas=[document.metadata for document in documents],
            ids=ids,
        )
        self.rebuild_sparse_index()
        return True, len(documents)

    def rebuild_sparse_index(self) -> None:
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        ids = raw.get("ids", [])
        texts = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        self.documents_by_id = {}
        bm25_documents = {}
        for doc_id, text, metadata in zip(ids, texts, metadatas):
            if not text:
                continue
            metadata = metadata or {}
            chunk_id = metadata.get("chunk_id", doc_id)
            document = Document(page_content=text, metadata=metadata)
            self.documents_by_id[chunk_id] = document
            bm25_documents[chunk_id] = text
        self.bm25.build(bm25_documents)
        self.graph.build(self.documents_by_id)

    def retrieve(self, query: str) -> RetrievalResult:
        dense_docs = self.vectorstore.similarity_search(query, k=self.settings.dense_k)
        dense_ranked = [
            RankedItem(doc_id=self._doc_key(document), score=1.0 / rank)
            for rank, document in enumerate(dense_docs, start=1)
        ]
        dense_by_key = {self._doc_key(document): document for document in dense_docs}

        sparse_hits = self.bm25.search(query, k=self.settings.sparse_k)
        sparse_ranked = [RankedItem(doc_id=hit.doc_id, score=hit.score) for hit in sparse_hits]

        graph_hits = self.graph.search(query, k=self.settings.graph_k)
        graph_ranked = [RankedItem(doc_id=hit.doc_id, score=hit.score) for hit in graph_hits]

        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked, graph_ranked], k=self.settings.rrf_k)
        candidates = []
        seen = set()
        for item in fused:
            document = dense_by_key.get(item.doc_id) or self.documents_by_id.get(item.doc_id)
            if document is None or item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            candidates.append(document)

        reranked = self.reranker.rerank(query, candidates, self.settings.fused_k)
        return RetrievalResult(
            documents=reranked,
            dense_count=len(dense_docs),
            sparse_count=len(sparse_hits),
            graph_count=len(graph_hits),
            fused_count=len(candidates),
            graph_stats=self.graph.stats(),
        )

    def _doc_key(self, document: Document) -> str:
        metadata = document.metadata or {}
        if metadata.get("chunk_id"):
            return metadata["chunk_id"]
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "na")
        chunk = metadata.get("chunk_index", "na")
        return f"{source}:{page}:{chunk}"
