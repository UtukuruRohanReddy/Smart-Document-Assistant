from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, List, Optional

import numpy as np

GEMINI_EMBEDDING_BATCH_SIZE = 64
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_DIMENSION = 768
GEMINI_REQUEST_TIMEOUT_MS = 10_000
SEARCH_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "did",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "of", "on",
    "or", "please", "tell", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "would", "you",
}


def _normalize_l2(vectors: np.ndarray) -> None:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms != 0)


class _NumpyFlatIP:
    def __init__(self, dimension: int, vectors: Optional[np.ndarray] = None):
        self.d = dimension
        self._vectors = (
            np.asarray(vectors, dtype="float32").reshape(-1, dimension)
            if vectors is not None
            else np.empty((0, dimension), dtype="float32")
        )

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.concatenate(
            (self._vectors, np.asarray(vectors, dtype="float32").reshape(-1, self.d)),
            axis=0,
        )

    def search(self, query: np.ndarray, k: int):
        query = np.asarray(query, dtype="float32").reshape(-1, self.d)
        scores = query @ self._vectors.T
        count = min(k, self.ntotal)
        indices = np.argsort(-scores, axis=1)[:, :count]
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        if count < k:
            missing = k - count
            indices = np.pad(indices, ((0, 0), (0, missing)), constant_values=-1)
            sorted_scores = np.pad(
                sorted_scores,
                ((0, 0), (0, missing)),
                constant_values=-np.inf,
            )
        return sorted_scores, indices

    def reconstruct_n(self, start: int, count: int) -> np.ndarray:
        return self._vectors[start:start + count].copy()


def _search_terms(text: str) -> set[str]:
    terms = set()
    for word in re.findall(r"\b[\w'-]+\b", text.lower()):
        if len(word) < 2 or word in SEARCH_STOP_WORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = f"{word[:-3]}y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        terms.add(word)
    return terms


class EmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", gemini_api_key: Optional[str] = None):
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key
        self._local_model = None

        if self.gemini_api_key:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "The Google Gen AI package is missing. Install project dependencies with `pip install -r requirements.txt`."
                ) from exc
            self.client = genai.Client(
                api_key=self.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=GEMINI_REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        else:
            self.client = None

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.model_name)
        return self._local_model

    def _embed_locally(self, texts: List[str]) -> np.ndarray:
        model = self._get_local_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def _embed_with_gemini(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> np.ndarray:
        if self.client is None:
            raise RuntimeError(
                "This document index uses Gemini embeddings. Enter a valid Google Gemini API key to search or add documents."
            )
        embeddings = []
        try:
            from google.genai import types

            for start in range(0, len(texts), GEMINI_EMBEDDING_BATCH_SIZE):
                response = self.client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=texts[start:start + GEMINI_EMBEDDING_BATCH_SIZE],
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=GEMINI_EMBEDDING_DIMENSION,
                    ),
                )
                embeddings.extend(item.values for item in response.embeddings)
        except Exception as exc:
            raise RuntimeError(
                "Gemini embedding failed. Check that your Google AI Studio API key is valid and Gemini API access is enabled."
            ) from exc
        return np.asarray(embeddings, dtype="float32")

    def embed_texts(
        self,
        texts: List[str],
        target_dimension: Optional[int] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        if target_dimension == 384:
            vectors = self._embed_locally(texts)
        elif target_dimension is not None:
            vectors = self._embed_with_gemini(texts, task_type=task_type)
        elif self.client is not None:
            vectors = self._embed_with_gemini(texts, task_type=task_type)
        else:
            vectors = self._embed_locally(texts)

        embeddings = np.asarray(vectors, dtype="float32")
        if target_dimension is not None and embeddings.shape[1] != target_dimension:
            raise ValueError(
                f"The selected embedding provider returned {embeddings.shape[1]} dimensions, "
                f"but the current index requires {target_dimension}."
            )
        _normalize_l2(embeddings)
        return embeddings

    def embed_query(
        self,
        query: str,
        target_dimension: Optional[int] = None,
    ) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("The question cannot be empty.")
        task_type = "RETRIEVAL_QUERY" if self.client is not None and target_dimension != 384 else "RETRIEVAL_DOCUMENT"
        return self.embed_texts(
            [query],
            target_dimension=target_dimension,
            task_type=task_type,
        )[0]


class FAISSDocumentStore:
    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "faiss_index.index"
        self.numpy_index_path = self.storage_dir / "embeddings.npy"
        self.text_search_only_path = self.storage_dir / "text_search_only"
        self.metadata_path = self.storage_dir / "chunks_metadata.json"
        self._index = None
        self.vector_search_available = True
        self._metadata: List[dict[str, Any]] = []
        self._load_metadata()
        self._load_index()

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Document metadata is corrupted: {self.metadata_path}. "
                    "Restore a valid backup or clear the local data directory."
                ) from exc
            if not isinstance(self._metadata, list) or any(
                not isinstance(item, dict)
                or not {"id", "file_name", "text", "source"}.issubset(item)
                for item in self._metadata
            ):
                raise ValueError(
                    f"Document metadata has an invalid structure: {self.metadata_path}."
                )

    def _load_index(self):
        if self.text_search_only_path.exists():
            self.vector_search_available = False
            return
        if self.numpy_index_path.exists():
            vectors = np.load(self.numpy_index_path)
            if vectors.ndim != 2 or len(vectors) != len(self._metadata):
                raise ValueError(
                    "The vector index and document metadata do not match. "
                    "Clear the document index and reprocess the affected documents."
                )
            self._index = _NumpyFlatIP(vectors.shape[1], vectors)
        elif self.index_path.exists():
            self.text_search_only_path.touch()
            self.vector_search_available = False
            return
        else:
            self._index = _NumpyFlatIP(384)
        if self._index.ntotal != len(self._metadata):
            raise ValueError(
                "The vector index and document metadata do not match. "
                "Clear the document index and reprocess the affected documents."
            )

    def _save_index(self):
        if self._index is not None:
            np.save(self.numpy_index_path, self._index.reconstruct_n(0, self._index.ntotal))

    def _save_metadata(self):
        self.metadata_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(
        self,
        file_name: str,
        chunks: List[str],
        embeddings: Optional[np.ndarray],
    ):
        if not chunks:
            raise ValueError("No valid document chunks were provided.")
        if self.vector_search_available:
            if embeddings is None:
                raise ValueError("Embeddings are required while vector search is enabled.")
            embeddings = np.asarray(embeddings, dtype="float32")
            if embeddings.size == 0:
                raise ValueError("No valid embeddings were provided.")
            if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
                raise ValueError("Each document chunk must have exactly one embedding.")

            if self._index.ntotal == 0:
                dimension = embeddings.shape[1]
                self._index = _NumpyFlatIP(dimension)
            elif embeddings.shape[1] != self._index.d:
                raise ValueError(
                    "These embeddings do not match the current index. Clear the index before changing embedding providers."
                )

            self._index.add(embeddings)
        else:
            self.text_search_only_path.touch()

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
        if not self.vector_search_available or self._index is None or self._index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self._index.d:
            raise ValueError(
                "The selected embedding provider does not match the indexed documents. Clear the index and reprocess documents with the current provider."
            )
        _normalize_l2(query)
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

    def search_text(
        self,
        query: str,
        k: int = 4,
        file_names: Optional[List[str]] = None,
    ) -> List[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be greater than zero.")
        query_terms = _search_terms(query)
        if not query_terms:
            return []

        allowed_files = set(file_names) if file_names is not None else None
        documents = [
            item for item in self._metadata
            if allowed_files is None or item["file_name"] in allowed_files
        ]
        if not documents:
            return []

        document_terms = [_search_terms(item["text"]) for item in documents]
        document_frequency = {
            term: sum(term in terms for terms in document_terms)
            for term in query_terms
        }
        weights = {
            term: math.log1p(len(documents) / frequency)
            for term, frequency in document_frequency.items()
            if frequency
        }
        if not weights:
            return []

        results = []
        for item, terms in zip(documents, document_terms):
            matched = query_terms & terms
            if not matched:
                continue
            item = item.copy()
            item["score"] = sum(weights[term] for term in matched) / sum(weights.values())
            results.append(item)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def remove_document(self, file_name: str) -> bool:
        removed_indices = [
            index for index, item in enumerate(self._metadata)
            if item["file_name"] == file_name
        ]
        if not removed_indices:
            return False

        if not self.vector_search_available:
            removed = set(removed_indices)
            self._metadata = [
                item for index, item in enumerate(self._metadata) if index not in removed
            ]
            for index, item in enumerate(self._metadata):
                item["id"] = index
            self._save_metadata()
            return True

        removed = set(removed_indices)
        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        keep_indices = [index for index in range(len(self._metadata)) if index not in removed]
        dimension = self._index.d
        self._index = _NumpyFlatIP(dimension)
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
        self.index_path.unlink(missing_ok=True)
        self.numpy_index_path.unlink(missing_ok=True)
        self.text_search_only_path.unlink(missing_ok=True)
        self._index = _NumpyFlatIP(dimension)
        self.vector_search_available = True
        self._metadata = []
        self._save_index()
        self._save_metadata()

    def metadata(self) -> List[dict[str, Any]]:
        return list(self._metadata)

    def document_summary(self) -> List[dict[str, Any]]:
        summary: dict[str, int] = {}
        for item in self._metadata:
            summary[item["file_name"]] = summary.get(item["file_name"], 0) + 1
        return [
            {"file_name": name, "chunks": count}
            for name, count in sorted(summary.items(), key=lambda entry: entry[0].lower())
        ]

    def document_chunks(self, file_name: str) -> List[str]:
        return [
            item["text"]
            for item in self._metadata
            if item["file_name"] == file_name
        ]
