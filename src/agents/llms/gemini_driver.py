from typing import List, Dict, Any, Optional
from google.genai import Client, types
from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig

class Gemini(BaseLLM):
    def __init__(self, id: str, config: ModelConfig, api_key: Optional[str] = None):
        super().__init__(id, config)
        self.client = Client(api_key=api_key) if api_key else Client()

    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
            
        gen_config_kwargs = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "system_instruction": system,
        }
        if self.config.top_p is not None:
            gen_config_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            gen_config_kwargs["top_k"] = self.config.top_k
        if self.config.stop_sequences:
            gen_config_kwargs["stop_sequences"] = self.config.stop_sequences
        if self.config.seed is not None:
            gen_config_kwargs["seed"] = self.config.seed
        if self.config.response_format:
            fmt = self.config.response_format
            if isinstance(fmt, dict) and fmt.get("type") == "json_object":
                gen_config_kwargs["response_mime_type"] = "application/json"
            elif fmt == "json":
                gen_config_kwargs["response_mime_type"] = "application/json"

        gen_config = types.GenerateContentConfig(**gen_config_kwargs)
        if tools:
            gen_config.tools = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.__name__, description=t.__doc__ or f"Execute {t.__name__}", parameters=types.Schema(type="OBJECT"))
                for t in tools
            ])]

        response = self.client.models.generate_content(model=self.id, contents=contents, config=gen_config)
        tool_calls = [{"id": "gemini_call", "name": fc.name, "arguments": fc.args} for fc in response.function_calls] if response.function_calls else None
        return {"content": response.text or "", "tool_calls": tool_calls}

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.embed_content(model=self.id, contents=texts)
        if isinstance(response.embeddings, list):
            return [e.values for e in response.embeddings]
        return [response.embeddings.values]