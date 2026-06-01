from dingir.rag.stores.base import StoreConfig
from dingir.rag.stores.chroma import ChromaStore
from dingir.rag.stores.native import LocalStore

Local = LocalStore
Chroma = ChromaStore

__all__ = ["StoreConfig", "Local", "Chroma"]
