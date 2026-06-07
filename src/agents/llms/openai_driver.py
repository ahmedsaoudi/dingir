import inspect
from typing import Any, Dict, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig
from openai import OpenAI as OpenAIClient


def convert_to_openai_tool(func) -> Dict[str, Any]:
    if isinstance(func, dict):
        return func
    sig = inspect.signature(func)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if name in ["top_k"]:  # Skip standard store macro parameter internals
            continue
        p_type = "string"
        if param.annotation in (int, float):
            p_type = "number"
        elif param.annotation == bool:
            p_type = "boolean"
        properties[name] = {
            "type": p_type,
            "description": f"The argument target value for {name}.",
        }
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Execute function {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


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
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if self.config.max_retries is not None:
            client_kwargs["max_retries"] = self.config.max_retries
        if self.config.timeout is not None:
            client_kwargs["timeout"] = self.config.timeout
        self.sync_client = OpenAIClient(**client_kwargs)

    def request(
        self,
        system: Optional[str],
        messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> Dict[str, Any]:
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            msg = {"role": m["role"], "content": m["content"]}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments"),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
                msg["role"] = "tool"
                msg["name"] = m.get("name")
            formatted_messages.append(msg)

        args = {
            "model": self.id,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            args["top_p"] = self.config.top_p
        if self.config.stop_sequences:
            args["stop"] = self.config.stop_sequences
        if self.config.presence_penalty != 0.0:
            args["presence_penalty"] = self.config.presence_penalty
        if self.config.frequency_penalty != 0.0:
            args["frequency_penalty"] = self.config.frequency_penalty
        if self.config.seed is not None:
            args["seed"] = self.config.seed
        if self.config.response_format is not None:
            args["response_format"] = self.config.response_format
        if self.config.logit_bias is not None:
            args["logit_bias"] = self.config.logit_bias
        if self.config.timeout is not None:
            args["timeout"] = self.config.timeout

        if tools:
            args["tools"] = [convert_to_openai_tool(t) for t in tools]

        response = self.sync_client.chat.completions.create(**args)
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

        reasoning = getattr(choice, "reasoning_content", None)
        content = choice.content or ""

        # Fallback to parse tags <think>...</think> or <|think|>...</|think|> if no native reasoning_content is returned
        if not reasoning and content:
            import re
            match = re.search(r'<(?:think|\|think\|)>(.*?)</(?:think|\|think\|)>', content, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                content = re.sub(r'<(?:think|\|think\|)>.*?</(?:think|\|think\|)>\s*', '', content, flags=re.DOTALL).strip()

        return {
            "content": content,
            "tool_calls": tc_out,
            "reasoning_content": reasoning,
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Integrated symmetrical embedding implementation hook."""
        response = self.sync_client.embeddings.create(
            input=texts, model=self.id
        )
        return [e.embedding for e in response.data]
