from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLM(ABC):
    def __init__(self, id: str, config: Any):
        self.id = id
        self.config = config

    @abstractmethod
    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        pass
