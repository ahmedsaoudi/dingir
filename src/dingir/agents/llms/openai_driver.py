import inspect
from typing import List, Dict, Any, Optional
from openai import OpenAI as OpenAIClient
from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig

def convert_to_openai_tool(func) -> Dict[str, Any]:
    sig = inspect.signature(func)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if name in ["top_k"]: # Skip standard store macro parameter internals
            continue
        p_type = "string"
        if param.annotation in (int, float):
            p_type = "number"
        elif param.annotation == bool:
            p_type = "boolean"
        properties[name] = {"type": p_type, "description": f"The argument target value for {name}."}
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Execute function {func.__name__}",
            "parameters": {"type": "object", "properties": properties, "required": required}
        }
    }

class OpenAI(BaseLLM):
    def __init__(self, id: str, config: ModelConfig, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(id, config)
        client_kwargs = {}
        if api_key: client_kwargs["api_key"] = api_key
        if base_url: client_kwargs["base_url"] = base_url
        self.sync_client = OpenAIClient(**client_kwargs)

    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            msg = {"role": m["role"], "content": m["content"]}
            if m.get("tool_calls"): msg["tool_calls"] = m["tool_calls"]
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
        if tools:
            args["tools"] = [convert_to_openai_tool(t) for t in tools]

        response = self.sync_client.chat.completions.create(**args)
        choice = response.choices[0].message
        
        # Normalize tool calls structures down to standard primitive dict shapes
        tc_out = None
        if getattr(choice, 'tool_calls', None):
            tc_out = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments} for tc in choice.tool_calls]
            
        return {"content": choice.content or "", "tool_calls": tc_out}

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Integrated symmetrical embedding implementation hook."""
        response = self.sync_client.embeddings.create(input=texts, model=self.id)
        return [e.embedding for e in response.data]