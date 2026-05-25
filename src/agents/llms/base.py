from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dingir.config import ModelConfig

class BaseLLM(ABC):
    def __init__(self, id: str, config: Any):
        self.id = id
        if isinstance(config, list):
            self.config = ModelConfig.merge(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            raise TypeError("Configuration must be a single ModelConfig or a list of ModelConfigs, not a dictionary or other type.")

    @abstractmethod
    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        pass
