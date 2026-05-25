from typing import List, Dict, Any, Optional
import ollama
from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig

class Ollama(BaseLLM):
    def __init__(self, id: str, config: ModelConfig, base_url: Optional[str] = None):
        super().__init__(id, config)
        self.client = ollama.Client(host=base_url) if base_url else ollama

    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        formatted = []
        if system: formatted.append({"role": "system", "content": system})
        for m in messages: formatted.append({"role": m["role"], "content": m["content"]})
        
        options = {"temperature": self.config.temperature, "num_predict": self.config.max_tokens}
        
        # Symmetrically maps formatted functional objects down to Ollama definitions
        ollama_tools = []
        for t in tools:
            ollama_tools.append({
                "type": "function",
                "function": {"name": t.__name__, "description": t.__doc__ or "", "parameters": {"type": "object", "properties": {}}}
            })

        response = self.client.chat(model=self.id, messages=formatted, options=options, tools=ollama_tools if tools else None)
        message = response.get("message", {})
        
        tc_out = None
        if message.get("tool_calls"):
            tc_out = [{"id": "ollama_call", "name": tc.get("function", {}).get("name"), "arguments": tc.get("function", {}).get("arguments")} for tc in message["tool_calls"]]
            
        return {"content": message.get("content", ""), "tool_calls": tc_out}