import json
from typing import Any, Callable, Dict, List, Optional

from dingir.chat import Chat


class Agent:
    """A completely stateless orchestration single-loop execution boundary engine."""

    def __init__(
        self,
        model: Any,
        system: str,
        description: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ):
        self.model = model
        self.description = description or system[:100]
        self.tools = tools or []

        # Build tool description list to append to system prompt
        if self.tools:
            tool_descriptions = []
            for t in self.tools:
                # Extract schema information
                if hasattr(t, "_dingir_schema"):
                    func = t._dingir_schema.get("function", {})
                    name = func.get("name", getattr(t, "__name__", "unknown"))
                    desc = func.get(
                        "description",
                        getattr(t, "__doc__", "No description provided."),
                    )
                    properties = func.get("parameters", {}).get(
                        "properties", {}
                    )
                    required = func.get("parameters", {}).get("required", [])
                else:
                    name = getattr(t, "__name__", "unknown")
                    desc = getattr(t, "__doc__", "No description provided.")
                    properties = {}
                    required = []
                    try:
                        import inspect

                        sig = inspect.signature(t)
                        for param_name, param in sig.parameters.items():
                            if param_name in ["top_k"]:
                                continue
                            p_type = "string"
                            if param.annotation in (int, float):
                                p_type = "number"
                            elif param.annotation == bool:
                                p_type = "boolean"
                            properties[param_name] = {
                                "type": p_type,
                                "description": f"The target value for {param_name}.",
                            }
                            if param.default == inspect.Parameter.empty:
                                required.append(param_name)
                    except Exception:
                        pass

                clean_desc = (
                    desc.strip() if desc else "No description provided."
                )

                tool_descriptions.append(
                    {
                        "name": name,
                        "description": clean_desc,
                        "parameters": {
                            "properties": properties,
                            "required": required,
                        },
                    }
                )

            tool_descriptions_json = json.dumps(
                tool_descriptions, indent=2
            )

            # Enrich system prompt with a comprehensive instruction framework
            tools_instruction_block = (
                "\n\n"
                "You are a model that assists in resolving "
                "the user instruction. To do so, you can do function calling "
                "with the following functions:\n\n"
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
            self.system = f"{system}{tools_instruction_block}"
        else:
            self.system = system

        self.__name__ = (
            self.__class__.__name__
            + "_"
            + model.id.replace("-", "_").replace(".", "_")
        )
        self.__doc__ = self.description

    def __call__(self, instruction: str) -> str:
        """Executes subagent handoffs cleanly while preserving internal state isolation."""
        scratchpad = Chat(system=self.system)
        self.respond(scratchpad, message=instruction)
        return (
            scratchpad.last_message.content if scratchpad.last_message else ""
        )

    def _execute_tool_sync(self, name: str, args: Any) -> str:
        tool_func = next((t for t in self.tools if t.__name__ == name), None)
        if not tool_func:
            return f"ERROR: Function execution endpoint '{name}' unavailable."
        try:
            resolved_args = json.loads(args) if isinstance(args, str) else args
            if isinstance(resolved_args, dict):
                return str(tool_func(**resolved_args))
            return str(tool_func(resolved_args))
        except Exception as e:
            return f"EXECUTION FAULT: {str(e)}"

    def _get_serialized_tools(
        self, chat: Optional[Chat] = None
    ) -> List[Dict[str, Any]]:
        """
        Converts Python callables and sub-agents into clean LLM provider
        schemas, with parameter serialization and hallucination schema recovery.
        """
        serialized = []
        existing_names = set()
        for t in self.tools:
            # Check if it's a sub-agent or custom tool with an explicit predefined schema
            if hasattr(t, "_dingir_schema"):
                schema = t._dingir_schema
                existing_names.add(schema.get("function", {}).get("name", ""))
                serialized.append(schema)
            else:
                # Dynamically construct a valid OpenAI/Anthropic function schema footprint
                import inspect

                properties = {}
                required = []
                try:
                    sig = inspect.signature(t)
                    for param_name, param in sig.parameters.items():
                        if param_name in ["top_k"]:
                            continue
                        p_type = "string"
                        if param.annotation in (int, float):
                            p_type = "number"
                        elif param.annotation == bool:
                            p_type = "boolean"
                        properties[param_name] = {
                            "type": p_type,
                            "description": f"The target value for {param_name}.",
                        }
                        if param.default == inspect.Parameter.empty:
                            required.append(param_name)
                except Exception:
                    pass

                existing_names.add(t.__name__)
                serialized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.__name__,
                            "description": t.__doc__
                            or "No description provided.",
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            },
                        },
                    }
                )

        # Inject dummy definitions for hallucinated/missing tools to prevent API gateway validation errors
        if chat:
            for m in chat.messages:
                if m.tool_calls:
                    for tc in m.tool_calls:
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

    def respond(
        self,
        chat: Chat,
        message: Optional[str] = None,
        on_step_callback: Optional[Callable[[Chat], None]] = None,
    ):
        if not chat.system:
            chat.system = self.system
        elif self.system not in chat.system:
            chat.system = f"{chat.system} {self.system}"
        if message:
            chat.add_message(role="user", content=message)
        while True:
            if on_step_callback:
                on_step_callback(chat)
            serializable_messages = [m.__dict__ for m in chat.messages]

            # FIX 2: Pass the generated JSON schemas, NOT the raw Python function objects
            tool_schemas = self._get_serialized_tools(chat)
            result = self.model.request(
                self.system, serializable_messages, tool_schemas
            )

            if result.get("tool_calls"):
                chat.add_message(
                    role="assistant",
                    content=result["content"],
                    tool_calls=result["tool_calls"],
                )
                for tc in result["tool_calls"]:
                    output = self._execute_tool_sync(
                        tc["name"], tc["arguments"]
                    )
                    chat.add_message(
                        role="tool",
                        content=output,
                        tool_call_id=tc.get("id", "call_idx"),
                        name=tc["name"],
                    )
                continue

            chat.add_message(role="assistant", content=result["content"])
            break
