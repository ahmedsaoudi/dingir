from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Unified behavioral parameters mapped symmetrically across all backend execution drivers."""
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)
