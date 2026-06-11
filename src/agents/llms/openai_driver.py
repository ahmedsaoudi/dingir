from typing import Any, Dict, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig
from openai import OpenAI as OpenAIClient


class OpenAI(BaseLLM):
    def __init__(
        self,
        id: str,
        config: ModelConfig,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(id, config)
        
        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if self.config.max_retries is not None:
            client_kwargs["max_retries"] = self.config.max_retries
        if self.config.timeout is not None:
            client_kwargs["timeout"] = self.config.timeout
            
        self.sync_client = OpenAIClient(**client_kwargs)
        self.use_native_tools = True

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        use_native_tools = getattr(self, "use_native_tools", True)
        if not use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        # Filter standard OpenAI API parameters
        openai_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "response_format",
            "logit_bias",
            "timeout",
        }
        
        params = {k: v for k, v in kwargs.items() if k in openai_params}

        if tools and use_native_tools:
            params["tools"] = self._get_serialized_tools(tools, formatted_messages)

        cleaned_messages = []
        for msg in formatted_messages:
            role = msg.get("role")
            content = msg.get("content")
            
            cleaned = {"role": role}
            if content is not None:
                cleaned["content"] = content
                
            if role == "system":
                if "name" in msg and msg["name"] is not None:
                    cleaned["name"] = msg["name"]
            elif role == "user":
                if "name" in msg and msg["name"] is not None:
                    cleaned["name"] = msg["name"]
            elif role == "assistant":
                if "name" in msg and msg["name"] is not None:
                    cleaned["name"] = msg["name"]
                if "tool_calls" in msg and msg["tool_calls"] is not None:
                    cleaned_tool_calls = []
                    for tc in msg["tool_calls"]:
                        tc_name = tc.get("name")
                        tc_args = tc.get("arguments")
                        if not tc_name and "function" in tc:
                            tc_name = tc["function"].get("name")
                            tc_args = tc["function"].get("arguments")
                        
                        tc_cleaned = {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": tc_args,
                            }
                        }
                        cleaned_tool_calls.append(tc_cleaned)
                    cleaned["tool_calls"] = cleaned_tool_calls
            elif role == "tool":
                cleaned["tool_call_id"] = msg.get("tool_call_id")
            cleaned_messages.append(cleaned)

        response = self.sync_client.chat.completions.create(
            model=self.id,
            messages=cleaned_messages,
            **params
        )
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
            "reasoning_content": getattr(choice, "reasoning_content", None),
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.sync_client.embeddings.create(
            input=texts, model=self.id
        )
        return [e.embedding for e in response.data]

