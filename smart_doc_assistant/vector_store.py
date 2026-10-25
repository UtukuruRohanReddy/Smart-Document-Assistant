from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", openai_api_key: Optional[str] = None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self._local_model = None

        if self.openai_api_key:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The OpenAI package is missing. Install project dependencies with `pip install -r requirements.txt`."
                ) from exc
            self.client = OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None

    def _get_local_model(self):
        if self._local_model is None:
            self._local_model = SentenceTransformer(self.model_name)
        return self._local_model

    def _embed_locally(self, texts: List[str]) -> np.ndarray:
        model = self._get_local_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def embed_texts(
        self,
        texts: List[str],
        target_dimension: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        if target_dimension == 384:
            vectors = self._embed_locally(texts)
        elif target_dimension is not None:
            if self.client is None:
                raise RuntimeError(
                    "This document index uses OpenAI embeddings. Enter a valid OpenAI API key to search or add documents."
                )
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                vectors = [item.embedding for item in response.data]
                vectors = np.asarray(vectors, dtype="float32")
            except Exception as exc:
                raise RuntimeError(
                    "OpenAI embedding failed. Check that your API key is valid and has API billing enabled. "
                    "If your existing index uses local embeddings, clear it before reprocessing with OpenAI."
                ) from exc
        elif self.client is not None:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                vectors = [item.embedding for item in response.data]
            except Exception as exc:
                raise RuntimeError(
                    "OpenAI embedding failed. Check that your API key is valid and has API billing enabled."
                ) from exc
        else:
            vectors = self._embed_locally(texts)

        embeddings = np.asarray(vectors, dtype="float32")
        if target_dimension is not None and embeddings.shape[1] != target_dimension:
            raise ValueError(
                f"The selected embedding provider returned {embeddings.shape[1]} dimensions, "
                f"but the current index requires {target_dimension}."
            )
        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(
        self,
        query: str,
        target_dimension: Optional[int] = None,
    ) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("The question cannot be empty.")
        return self.embed_texts([query], target_dimension=target_dimension)[0]


class FAISSDocumentStore:
    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "faiss_index.index"
        self.metadata_path = self.storage_dir / "chunks_metadata.json"
        self._index = None
        self._metadata: List[dict[str, Any]] = []
        self._load_metadata()
        self._load_index()

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._metadata = []
                self.metadata_path.write_text("[]", encoding="utf-8")

    def _load_index(self):
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        else:
            self._index = faiss.IndexFlatIP(384)

    def _save_index(self):
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

    def _save_metadata(self):
        self.metadata_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(self, file_name: str, chunks: List[str], embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype="float32")
        if not chunks or embeddings.size == 0:
            raise ValueError("No valid chunks or embeddings were provided.")
        if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
            raise ValueError("Each document chunk must have exactly one embedding.")

        if self._index.ntotal == 0:
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
        elif embeddings.shape[1] != self._index.d:
            raise ValueError(
                "These embeddings do not match the current index. Clear the index before changing embedding providers."
            )

        self._index.add(embeddings)

        for idx, chunk in enumerate(chunks):
            self._metadata.append(
                {
                    "id": len(self._metadata),
                    "file_name": file_name,
                    "text": chunk,
                    "source": f"{file_name}#chunk-{idx + 1}",
                }
            )

        self._save_index()
        self._save_metadata()

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 4,
        file_names: Optional[List[str]] = None,
    ) -> List[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be greater than zero.")
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self._index.d:
            raise ValueError(
                "The selected embedding provider does not match the indexed documents. Clear the index and reprocess documents with the current provider."
            )
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, self._index.ntotal)
        results: List[dict[str, Any]] = []
        allowed_files = set(file_names) if file_names is not None else None
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            metadata = self._metadata[int(idx)]
            if allowed_files is not None and metadata["file_name"] not in allowed_files:
                continue
            item = metadata.copy()
            item["score"] = float(score)
            results.append(item)
            if len(results) == k:
                break
        return results

    def remove_document(self, file_name: str) -> bool:
        removed_indices = [
            index for index, item in enumerate(self._metadata)
            if item["file_name"] == file_name
        ]
        if not removed_indices:
            return False

        removed = set(removed_indices)
        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        keep_indices = [index for index in range(len(self._metadata)) if index not in removed]
        dimension = self._index.d
        self._index = faiss.IndexFlatIP(dimension)
        if keep_indices:
            self._index.add(np.asarray(vectors[keep_indices], dtype="float32"))

        self._metadata = [item for index, item in enumerate(self._metadata) if index not in removed]
        for index, item in enumerate(self._metadata):
            item["id"] = index
        self._save_index()
        self._save_metadata()
        return True

    def clear(self):
        dimension = self._index.d if self._index is not None else 384
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata = []
        self._save_index()
        self._save_metadata()

    def metadata(self) -> List[dict[str, Any]]:
        return list(self._metadata)

    def document_summary(self) -> List[dict[str, Any]]:
        summary: dict[str, int] = {}
        for item in self._metadata:
            summary[item["file_name"]] = summary.get(item["file_name"], 0) + 1
        return [{"file_name": name, "chunks": count} for name, count in summary.items()]

    def document_chunks(self, file_name: str) -> List[str]:
        return [
            item["text"]
            for item in self._metadata
            if item["file_name"] == file_name
        ]
