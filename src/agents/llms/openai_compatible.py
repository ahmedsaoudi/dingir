import os
from typing import Any, Dict, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.agents.llms.openai_driver import convert_to_openai_tool
from dingir.config import ModelConfig
from openai import OpenAI as OpenAIClient


class OpenAICompatible(BaseLLM):
    """Driver for LLM providers offering an OpenAI-compatible API (e.g. DeepSeek, Anyscale, Together, Groq, local vLLM).
    If no api_key or base_url are provided, it falls back to environment variables:
    - OPENAI_COMPATIBLE_API_KEY or OPENAI_API_KEY
    - OPENAI_COMPATIBLE_BASE_URL
    """

    def __init__(
        self,
        id: str,
        config: ModelConfig,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        native_tools: Optional[bool] = None,
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

        self.use_native_tools = native_tools if native_tools is not None else True

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
            role = m["role"]
            content = m["content"] or ""

            if not self.use_native_tools:
                # Local API servers often do not render tool/function roles in chat templates.
                # Format tool calls and results as standard assistant and user text messages.
                if role == "assistant" and m.get("tool_calls"):
                    calls_text = []
                    for tc in m["tool_calls"]:
                        calls_text.append(
                            f"[Called tool '{tc.get('name')}' with arguments: {tc.get('arguments')}]"
                        )
                    calls_summary = "\n".join(calls_text)
                    content = (
                        f"{content}\n{calls_summary}"
                        if content
                        else calls_summary
                    )

                if role == "tool":
                    role = "user"
                    content = f"[Tool Result for '{m.get('name')}']:\n{content}"

                msg = {"role": role, "content": content}
            else:
                msg = {"role": role, "content": content}
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

        # Custom compatible parameters routed through extra_body to bypass client-side validation
        extra_body = {}
        if self.config.min_p is not None:
            extra_body["min_p"] = self.config.min_p
        if self.config.repeat_penalty is not None:
            extra_body["repetition_penalty"] = self.config.repeat_penalty
        if self.config.top_k is not None:
            extra_body["top_k"] = self.config.top_k

        if extra_body:
            args["extra_body"] = extra_body

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

            match = re.search(
                r"<(?:think|\|think\|)>(.*?)</(?:think|\|think\|)>",
                content,
                re.DOTALL,
            )
            if match:
                reasoning = match.group(1).strip()
                content = re.sub(
                    r"<(?:think|\|think\|)>.*?</(?:think|\|think\|)>\s*",
                    "",
                    content,
                    flags=re.DOTALL,
                ).strip()

        return {
            "content": content,
            "tool_calls": tc_out,
            "reasoning_content": reasoning,
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.sync_client.embeddings.create(
            input=texts, model=self.id
        )
        return [e.embedding for e in response.data]
