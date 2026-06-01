from typing import Any, Dict, List, Optional

import numpy as np
import semchunk
from dingir.rag.stores.base import BaseStore, StoreConfig


class LocalStore(BaseStore):
    """A completely transparent local in-memory vector storage framework using numpy."""

    def __init__(self, config: StoreConfig, embedding_model: Any):
        super().__init__(config, embedding_model)
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.__name__ = f"query_local_store_{config.collection_name}"
        self.__doc__ = f"Searches memory matrix table '{config.collection_name}' for matching reference data."

    def ingest(self, document_text: str, meta: Optional[Dict[str, Any]] = None):
        """Slices text into semantically cohesive blocks using a fallback character-length counter."""
        chunks = semchunk.chunk(
            document_text,
            chunk_size=self.config.chunk_size,
            token_counter=lambda text: len(text),
        )
        if not chunks or not self.embedding_model:
            return

        vectors = self.embedding_model.embed(chunks)
        for v, chunk in zip(vectors, chunks):
            arr = np.array(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            self.vectors.append(arr / norm if norm > 0 else arr)
            self.metadata.append({"text": chunk, **(meta or {})})

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Executes a flat vector dot-product similarity search over active system memory."""
        if not self.vectors or not self.embedding_model:
            return "Data store empty."

        q_vec = self.embedding_model.embed([query])[0]
        q_arr = np.array(q_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        similarities = [float(np.dot(q_arr, v)) for v in self.vectors]
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return "\n\n---\n\n".join(
            [self.metadata[idx]["text"] for idx in ranked_indices]
        )
