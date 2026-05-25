import os
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient
from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig

class HuggingFace(BaseLLM):
    """
    Symmetrical Hugging Face Serverless Inference API driver.
    If no token is provided, it falls back to the HF_TOKEN environment variable.
    """
    def __init__(self, id: str, config: ModelConfig, token: Optional[str] = None):
        super().__init__(id, config)
        self.client = InferenceClient(model=id, token=token or os.environ.get("HF_TOKEN"))

    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        # Formulate a standard chat conversation sequence payload
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            formatted_messages.append({"role": m["role"], "content": m["content"]})

        # Utilise the unified chat_completion API 
        response = self.client.chat_completion(
            messages=formatted_messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else 0.01
        )
        
        choice = response.choices[0].message
        return {
            "content": choice.content or "",
            "tool_calls": None  # Serverless text-generation defaults skip native tool-use tracking 
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Maps incoming strings to feature extraction matrices using serverless pooling."""
        # Ensure we target feature extraction safely across any text transformer model
        response = self.client.feature_extraction(text=texts)
        # Handle conversion from numpy/list output variants gracefully
        if hasattr(response, "tolist"):
            return response.tolist()
        return list(response)