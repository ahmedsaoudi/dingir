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
        self, id: str, config: ModelConfig, api_key: Optional[str] = None, native_tools: Optional[bool] = None
    ):
        super().__init__(id, config)
        self.client = InferenceClient(
            model=id, token=api_key or os.environ.get("HF_TOKEN")
        )
        self.use_native_tools = native_tools if native_tools is not None else False

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if "temperature" in kwargs and kwargs["temperature"] <= 0:
            kwargs["temperature"] = 0.01

        chat_kwargs = {
            "messages": formatted_messages,
            **kwargs
        }
        
        if tools:
            chat_kwargs["tools"] = self._get_serialized_tools(tools, formatted_messages)

        response = self.client.chat_completion(**chat_kwargs)

        choice = response.choices[0].message
        
        tc_out = None
        if getattr(choice, "tool_calls", None):
            tc_out = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.tool_calls
            ]
            
        return {
            "content": choice.content or "",
            "tool_calls": tc_out,
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Maps incoming strings to feature extraction matrices using serverless pooling."""
        # Ensure we target feature extraction safely across any text transformer model
        response = self.client.feature_extraction(text=texts)
        # Handle conversion from numpy/list output variants gracefully
        if hasattr(response, "tolist"):
            return response.tolist()
        return list(response)
