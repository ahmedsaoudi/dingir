import os
from typing import Any, Dict, Generator, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig
from huggingface_hub import InferenceClient


class HuggingFace(BaseLLM):
    """Symmetrical Hugging Face Serverless Inference API driver.
    If no token is provided, it falls back to the HF_TOKEN environment variable.
    """

    def __init__(
        self, id: str, config: ModelConfig, api_key: Optional[str] = None
    ):
        super().__init__(id, config)
        self.client = InferenceClient(
            model=id, token=api_key or os.environ.get("HF_TOKEN")
        )
        self.use_native_tools = True

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        if "temperature" in kwargs and kwargs["temperature"] <= 0:
            kwargs["temperature"] = 0.01

        chat_kwargs = {
            "messages": formatted_messages,
            **kwargs
        }
        
        if tools and self.use_native_tools:
            chat_kwargs["tools"] = self._get_serialized_tools(tools, formatted_messages)

        raw_request = {
            "model": self.id,
            **chat_kwargs
        }
        response = self.client.chat_completion(**chat_kwargs)
        self._log_raw_api_call(raw_request, response)

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

    def execute_stream(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        if "temperature" in kwargs and kwargs["temperature"] <= 0:
            kwargs["temperature"] = 0.01

        chat_kwargs = {
            "messages": formatted_messages,
            "stream": True,
            **kwargs
        }

        if tools and self.use_native_tools:
            chat_kwargs["tools"] = self._get_serialized_tools(tools, formatted_messages)

        stream = self.client.chat_completion(**chat_kwargs)

        full_content = ""
        tool_calls_accum: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                full_content += delta.content
                yield {"type": "content_delta", "content": delta.content}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    if getattr(tc_delta, "id", None):
                        tool_calls_accum[idx]["id"] = tc_delta.id
                    if getattr(tc_delta, "function", None):
                        if getattr(tc_delta.function, "name", None):
                            tool_calls_accum[idx]["name"] += tc_delta.function.name
                        if getattr(tc_delta.function, "arguments", None):
                            tool_calls_accum[idx]["arguments"] += tc_delta.function.arguments

        tc_out = None
        if tool_calls_accum:
            tc_out = [
                tool_calls_accum[idx]
                for idx in sorted(tool_calls_accum.keys())
            ]

        raw_request = {
            "model": self.id,
            **chat_kwargs
        }
        raw_response = {
            "content": full_content,
            "tool_calls": tc_out,
            "reasoning_content": None,
        }
        self._log_raw_api_call(raw_request, raw_response)

        yield {
            "type": "done",
            "content": full_content,
            "tool_calls": tc_out,
            "reasoning_content": None,
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Maps incoming strings to feature extraction matrices using serverless pooling."""
        # Ensure we target feature extraction safely across any text transformer model
        response = self.client.feature_extraction(text=texts)
        # Handle conversion from numpy/list output variants gracefully
        if hasattr(response, "tolist"):
            return response.tolist()
        return list(response)
