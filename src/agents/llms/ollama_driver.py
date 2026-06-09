from typing import Any, Dict, List, Optional

import ollama
from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig


class Ollama(BaseLLM):
    def __init__(
        self, id: str, config: ModelConfig, base_url: Optional[str] = None, native_tools: Optional[bool] = None
    ):
        super().__init__(id, config)
        self.client = ollama.Client(host=base_url) if base_url else ollama
        self.use_native_tools = native_tools if native_tools is not None else True

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        if not self.use_native_tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        options = {}
        for k, v in kwargs.items():
            if k == "max_tokens":
                options["num_predict"] = v
            elif k == "response_format":
                pass
            else:
                options[k] = v

        chat_kwargs = {
            "model": self.id,
            "messages": formatted_messages,
            "options": options,
        }

        if tools:
            chat_kwargs["tools"] = self._get_serialized_tools(tools, formatted_messages)

        if "response_format" in kwargs and kwargs["response_format"]:
            fmt = kwargs["response_format"]
            if isinstance(fmt, dict) and fmt.get("type") == "json_object":
                chat_kwargs["format"] = "json"
            elif fmt == "json":
                chat_kwargs["format"] = "json"

        response = self.client.chat(**chat_kwargs)
        message = response.get("message", {})

        tc_out = None
        if message.get("tool_calls"):
            tc_out = [
                {
                    "id": "ollama_call",
                    "name": tc.get("function", {}).get("name"),
                    "arguments": tc.get("function", {}).get("arguments"),
                }
                for tc in message["tool_calls"]
            ]

        return {"content": message.get("content", ""), "tool_calls": tc_out}
