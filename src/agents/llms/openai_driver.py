import inspect
import os
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
        
        resolved_api_key = (
            api_key
            or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        resolved_base_url = base_url or os.environ.get(
            "OPENAI_COMPATIBLE_BASE_URL"
        )
        
        client_kwargs = {}
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
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
        
        if not self.use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        # Separate standard kwargs from extra_body
        extra_body = {}
        standard_kwargs = {}
        
        # Mapping known standard arguments vs extra arguments
        valid_openai_kwargs = {
            "temperature", "max_tokens", "top_p", "stop",
            "presence_penalty", "frequency_penalty", "seed",
            "response_format", "logit_bias", "timeout"
        }
        
        for k, v in kwargs.items():
            if k in valid_openai_kwargs:
                standard_kwargs[k] = v
            else:
                extra_body[k] = v
                
        if extra_body:
            standard_kwargs["extra_body"] = extra_body

        if tools and self.use_native_tools:
            standard_kwargs["tools"] = self._get_serialized_tools(tools, formatted_messages)

        response = self.sync_client.chat.completions.create(
            model=self.id,
            messages=formatted_messages,
            **standard_kwargs
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
