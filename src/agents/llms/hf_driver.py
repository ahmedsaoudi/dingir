import os
from typing import Any, Dict, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig
from huggingface_hub import InferenceClient


class HuggingFace(BaseLLM):
    """Symmetrical Hugging Face Serverless Inference API driver.
    If no token is provided, it falls back to the HF_TOKEN environment variable.
    """

    def __init__(
        self, id: str, config: ModelConfig, token: Optional[str] = None
    ):
        super().__init__(id, config)
        self.client = InferenceClient(
            model=id, token=token or os.environ.get("HF_TOKEN")
        )

    def request(
        self,
        system: Optional[str],
        messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> Dict[str, Any]:
        # Formulate a standard chat conversation sequence payload
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            formatted_messages.append(
                {"role": m["role"], "content": m["content"]}
            )

        # Utilise the unified chat_completion API
        chat_kwargs = {
            "messages": formatted_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
            if self.config.temperature > 0
            else 0.01,
        }
        if self.config.top_p is not None:
            chat_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            chat_kwargs["top_k"] = self.config.top_k
        if self.config.stop_sequences:
            chat_kwargs["stop"] = self.config.stop_sequences
        if self.config.seed is not None:
            chat_kwargs["seed"] = self.config.seed
        if self.config.response_format is not None:
            chat_kwargs["response_format"] = self.config.response_format
        if self.config.presence_penalty != 0.0:
            chat_kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.frequency_penalty != 0.0:
            chat_kwargs["frequency_penalty"] = self.config.frequency_penalty
        if self.config.timeout is not None:
            chat_kwargs["timeout"] = self.config.timeout

        response = self.client.chat_completion(**chat_kwargs)

        choice = response.choices[0].message
        return {
            "content": choice.content or "",
            "tool_calls": None,  # Serverless text-generation defaults skip native tool-use tracking
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Maps incoming strings to feature extraction matrices using serverless pooling."""
        # Ensure we target feature extraction safely across any text transformer model
        response = self.client.feature_extraction(text=texts)
        # Handle conversion from numpy/list output variants gracefully
        if hasattr(response, "tolist"):
            return response.tolist()
        return list(response)
