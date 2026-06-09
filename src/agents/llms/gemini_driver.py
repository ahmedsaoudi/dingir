from typing import Any, Dict, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig
from google.genai import Client, types


class Gemini(BaseLLM):
    def __init__(
        self, id: str, config: ModelConfig, api_key: Optional[str] = None
    ):
        super().__init__(id, config)
        self.client = Client(api_key=api_key) if api_key else Client()

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        contents = []
        system = None
        
        for m in formatted_messages:
            if m["role"] == "system":
                system = m["content"]
                continue
                
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role, parts=[types.Part.from_text(text=m["content"])]
                )
            )

        gen_config_kwargs = {}
        for k, v in kwargs.items():
            if k == "max_tokens":
                gen_config_kwargs["max_output_tokens"] = v
            elif k == "stop":
                gen_config_kwargs["stop_sequences"] = v
            elif k == "response_format":
                pass
            else:
                gen_config_kwargs[k] = v
                
        if system:
            gen_config_kwargs["system_instruction"] = system

        if "response_format" in kwargs and kwargs["response_format"]:
            fmt = kwargs["response_format"]
            if isinstance(fmt, dict) and fmt.get("type") == "json_object":
                gen_config_kwargs["response_mime_type"] = "application/json"
            elif fmt == "json":
                gen_config_kwargs["response_mime_type"] = "application/json"

        gen_config = types.GenerateContentConfig(**gen_config_kwargs)
        if tools:
            schemas = self._get_serialized_tools(tools, formatted_messages)
            gen_config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=s["function"]["name"],
                            description=s["function"]["description"],
                            parameters=s["function"]["parameters"],
                        )
                        for s in schemas
                    ]
                )
            ]

        response = self.client.models.generate_content(
            model=self.id, contents=contents, config=gen_config
        )
        tool_calls = (
            [
                {"id": "gemini_call", "name": fc.name, "arguments": fc.args}
                for fc in response.function_calls
            ]
            if response.function_calls
            else None
        )
        return {"content": response.text or "", "tool_calls": tool_calls}

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.id, contents=texts
        )
        if isinstance(response.embeddings, list):
            return [e.values for e in response.embeddings]
        return [response.embeddings.values]
