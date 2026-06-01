from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StoreConfig:
    """Isolated infrastructure configuration block for vector store adapters."""

    collection_name: str = "dingir_default"
    chunk_size: int = 400
    path: Optional[str] = None
    host: Optional[str] = None
    port: int = 8000
    api_key: Optional[str] = None
    tenant: str = "default_tenant"
    database: str = "default_database"
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class BaseStore(ABC):
    def __init__(
        self, config: StoreConfig, embedding_model: Optional[Any] = None
    ):
        self.config = config
        self.embedding_model = embedding_model

    @abstractmethod
    def ingest(self, document_text: str, meta: Optional[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> str:
        pass

    def query(self, text: str, llm: Any, top_k: int = 3) -> str:
        """The Direct Shot: Retrieves context and runs a fast single-turn prompt completion."""
        context = self.retrieve(text, top_k=top_k)
        grounding_sys = (
            "You are a fact-grounded assistant. Answer the user request using ONLY the "
            "provided context records. If the answer is unavailable, state it explicitly.\n\n"
            f"=== CONTEXT REFERENCE RECORDS ===\n{context}"
        )
        response = llm.request(
            system=grounding_sys,
            messages=[{"role": "user", "content": text}],
            tools=[],
        )
        return response["content"]

    def __call__(self, query: str) -> str:
        """Enables the Store to act directly as a pluggable Agent tool signature."""
        return self.retrieve(query)
