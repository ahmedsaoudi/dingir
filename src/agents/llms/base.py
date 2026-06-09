import re
import json
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from dingir.config import ModelConfig


class BaseLLM(ABC):
    def __init__(self, id: str, config: Any):
        self.id = id
        if isinstance(config, list):
            self.config = ModelConfig.merge(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            raise TypeError(
                "Configuration must be a single ModelConfig or a list of ModelConfigs, not a dictionary or other type."
            )

    def request(
        self,
        system: Optional[str],
        messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> Dict[str, Any]:
        """Standardized template method for LLM requests."""
        use_native_tools = getattr(self, "use_native_tools", True)
        
        if tools and not use_native_tools:
            system = self._format_fallback_prompt(system, tools, messages)
            
        formatted_messages = self._format_messages(system, messages)
        kwargs = self._map_config()
        
        # Execute provider-specific logic
        response = self.execute(formatted_messages, tools, **kwargs)
        
        # Apply standard post-processing (e.g., parsing <think> tags if reasoning_content is absent)
        content = response.get("content", "")
        reasoning = response.get("reasoning_content")
        
        if not reasoning and content:
            parsed_content, parsed_reasoning = self._parse_reasoning(content)
            if parsed_reasoning:
                response["content"] = parsed_content
                response["reasoning_content"] = parsed_reasoning
                
        return response

    @abstractmethod
    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Child classes must implement the API call and response extraction."""
        pass

    def _format_messages(
        self, system: Optional[str], messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Standard translation of system and messages into a unified sequence."""
        formatted = []
        if system:
            formatted.append({"role": "system", "content": system})
        for m in messages:
            formatted.append({"role": m["role"], "content": m.get("content", "")})
            # tool calls are left intact for specific drivers to format
            if "tool_calls" in m:
                formatted[-1]["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                formatted[-1]["tool_call_id"] = m["tool_call_id"]
                formatted[-1]["name"] = m.get("name")
        return formatted

    def _map_config(self) -> Dict[str, Any]:
        """Maps standard ModelConfig fields to generic kwargs."""
        kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k
        if self.config.stop_sequences:
            kwargs["stop"] = self.config.stop_sequences
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if self.config.response_format is not None:
            kwargs["response_format"] = self.config.response_format
        if self.config.presence_penalty != 0.0:
            kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = self.config.frequency_penalty
        if self.config.timeout is not None:
            kwargs["timeout"] = self.config.timeout
        if self.config.min_p is not None:
            kwargs["min_p"] = self.config.min_p
        if self.config.repeat_penalty is not None:
            kwargs["repetition_penalty"] = self.config.repeat_penalty
        if self.config.logit_bias is not None:
            kwargs["logit_bias"] = self.config.logit_bias
            
        return kwargs

    def _parse_reasoning(self, content: str) -> tuple[str, Optional[str]]:
        """Extracts <think> blocks and returns (cleaned_content, reasoning)."""
        match = re.search(r"<(?:think|\|think\|)>(.*?)</(?:think|\|think\|)>", content, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            cleaned_content = re.sub(
                r"<(?:think|\|think\|)>.*?</(?:think|\|think\|)>\s*",
                "",
                content,
                flags=re.DOTALL,
            ).strip()
            return cleaned_content, reasoning
        return content, None

    def _format_fallback_tools(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Formats tool calls as standard assistant/user text for models lacking native tool support."""
        formatted = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""

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
            formatted.append(msg)
        return formatted

    def _convert_to_json_schema(self, func) -> Dict[str, Any]:
        if isinstance(func, dict):
            return func
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        param_docs = {}
        if docstring:
            matches = re.finditer(r"(?:- )?(?::param )?([a-zA-Z0-9_]+)(?: \(.*?\))?[:\-]\s*(.*?)(?=\n\s*(?:[a-zA-Z0-9_]+[:\-]|:param|\Z))", docstring, re.DOTALL)
            for match in matches:
                param_docs[match.group(1)] = match.group(2).strip()

        properties = {}
        required = []
        for name, param in sig.parameters.items():
            if name in ["top_k"]:
                continue
            p_type = "string"
            if param.annotation in (int, float):
                p_type = "number"
            elif param.annotation == bool:
                p_type = "boolean"
                
            desc = param_docs.get(name) or f"The argument target value for {name}."
            desc = re.sub(r"\s+", " ", desc).strip()
            
            properties[name] = {
                "type": p_type,
                "description": desc,
            }
            if param.default == inspect.Parameter.empty:
                required.append(name)
                
        main_desc = re.split(r"\n\s*(?:Args|Arguments|Parameters|:param)\s*:?", docstring)[0].strip()
        if not main_desc:
            main_desc = f"Execute function {func.__name__}"

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": main_desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _get_serialized_tools(self, tools: List[Any], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialized = []
        existing_names = set()
        for t in tools:
            t_unwrapped = t
            while hasattr(t_unwrapped, "__wrapped__"):
                t_unwrapped = t_unwrapped.__wrapped__
            if hasattr(t_unwrapped, "_dingir_schema"):
                schema = t_unwrapped._dingir_schema
                existing_names.add(schema.get("function", {}).get("name", ""))
                serialized.append(schema)
            else:
                try:
                    schema = self._convert_to_json_schema(t_unwrapped)
                    existing_names.add(schema.get("function", {}).get("name", ""))
                    serialized.append(schema)
                except Exception:
                    pass

        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    name = tc.get("name")
                    if name and name not in existing_names:
                        existing_names.add(name)
                        serialized.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "description": "Hallucinated or unavailable tool helper.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                    },
                                },
                            }
                        )
        return serialized

    def _format_fallback_prompt(self, system: Optional[str], tools: List[Any], messages: List[Dict[str, Any]]) -> str:
        if not tools:
            return system or ""
        
        tool_schemas = self._get_serialized_tools(tools, messages)
        
        tool_descriptions = []
        for schema in tool_schemas:
            func = schema.get("function", {})
            tool_descriptions.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters")
            })
            
        tool_descriptions_json = json.dumps(tool_descriptions, indent=2)
        
        tools_instruction_block = (
            "\n\n"
            "You can do function calling with the following functions:\n\n"
            f"{tool_descriptions_json}\n\n"
            "TOOL EXECUTION GUIDELINES:\n"
            "1. Analyze the user query and select the most appropriate "
            "tool from the toolset above. If no tool is needed or "
            "suitable, respond directly using your general knowledge.\n"
            "2. When executing a tool call, ensure you strictly adhere "
            "to the types, descriptions, and required constraints "
            "defined in the parameter schema.\n"
            "3. All arguments must be passed as a valid JSON object "
            "matching the defined parameter structure.\n"
            "4. Always execute the tool through the provider's native "
            "tool-calling interface. Do not simulate or mock tool "
            "responses in your text content."
        )
        return f"{system or ''}{tools_instruction_block}"
