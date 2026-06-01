import uuid
from typing import Any, Dict, List, Optional

from dingir.rag.stores.base import BaseStore, StoreConfig


class ChromaStore(BaseStore):
    """Self-contained storage client wrapper for ChromaDB execution targets."""

    def __init__(
        self, config: StoreConfig, embedding_model: Optional[Any] = None
    ):
        super().__init__(config, embedding_model)
        import chromadb

        self.__name__ = f"query_chroma_{config.collection_name}"
        self.__doc__ = f"Queries the persistent Chroma store table '{config.collection_name}' for records."

        if config.api_key:
            self.client = chromadb.CloudClient(
                tenant=config.tenant,
                database=config.database,
                api_key=config.api_key,
            )
        elif config.host:
            self.client = chromadb.HttpClient(
                host=config.host, port=config.port
            )
        elif config.path:
            self.client = chromadb.PersistentClient(path=config.path)
        else:
            self.client = chromadb.Client()

        chroma_emb_fn = None
        if embedding_model:

            class ChromaEmbeddingBridge:
                def __call__(self, input: List[str]) -> List[List[float]]:
                    return embedding_model.embed(input)

            chroma_emb_fn = ChromaEmbeddingBridge()

        self.collection = self.client.get_or_create_collection(
            name=config.collection_name, embedding_function=chroma_emb_fn
        )

    def ingest(self, document_text: str, meta: Optional[Dict[str, Any]] = None):
        """Slices and writes chunks directly to the persistent target database instance."""
        import semchunk

        chunks = semchunk.chunk(
            document_text,
            chunk_size=self.config.chunk_size,
            token_counter=lambda text: len(text),
        )
        if not chunks:
            return

        generated_ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            documents=chunks,
            ids=generated_ids,
            metadatas=[meta or {} for _ in chunks] if meta else None,
        )

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Queries the vector collection utilizing either custom driver embedding pipelines or default internal extractors."""
        if self.embedding_model:
            query_vector = self.embedding_model.embed([query])[0]
            results = self.collection.query(
                query_embeddings=[query_vector], n_results=top_k
            )
        else:
            results = self.collection.query(
                query_texts=[query], n_results=top_k
            )

        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
        ):
            return "No matching structural records found."
        return "\n\n---\n\n".join(results["documents"][0])
