import functools
import inspect
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


def track_explicit_args(init_func):
    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init_func)
        bound = sig.bind(self, *args, **kwargs)
        self._explicitly_set = {
            name for name in bound.arguments if name != "self"
        }
        init_func(self, *args, **kwargs)

    return wrapper


@dataclass
class ModelConfig:
    """Unified behavioral parameters mapped symmetrically across all backend execution drivers."""

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    logit_bias: Optional[Dict[str, float]] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None

    @track_explicit_args
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        min_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = (
            stop_sequences if stop_sequences is not None else []
        )
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.response_format = response_format
        self.logit_bias = logit_bias
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_p = min_p
        self.repeat_penalty = repeat_penalty

    @classmethod
    def merge(cls, configs: List["ModelConfig"]) -> "ModelConfig":
        merged = cls()
        for cfg in configs:
            if cfg is None:
                continue
            if not isinstance(cfg, ModelConfig):
                raise TypeError(
                    f"Expected ModelConfig instance, got {type(cfg).__name__}"
                )

            explicit = getattr(cfg, "_explicitly_set", None)
            if explicit is None:
                # Fallback to copy all fields if not decorated
                for f in fields(cfg):
                    setattr(merged, f.name, getattr(cfg, f.name))
            else:
                for field_name in explicit:
                    if hasattr(merged, field_name):
                        setattr(merged, field_name, getattr(cfg, field_name))
        return merged
