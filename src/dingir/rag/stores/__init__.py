from dingir.rag.stores.base import StoreConfig
from dingir.rag.stores.native import LocalStore
from dingir.rag.stores.chroma import ChromaStore

Local = LocalStore
Chroma = ChromaStore

__all__ = ["StoreConfig", "Local", "Chroma"]