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
        self.system = system
        self.description = description or system[:100]
        self.tools = tools or []
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

    def _get_serialized_tools(self) -> List[Dict[str, Any]]:
        """FIX 1: Converts Python callables and sub-agents into clean LLM provider schemas."""
        serialized = []
        for t in self.tools:
            # Check if it's a sub-agent or custom tool with an explicit predefined schema
            if hasattr(t, "_dingir_schema"):
                serialized.append(t._dingir_schema)
            else:
                # Dynamically construct a valid OpenAI/Anthropic function schema footprint
                serialized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.__name__,
                            "description": t.__doc__
                            or "No description provided.",
                            "parameters": {
                                "type": "object",
                                "properties": {},  # For a simple demo, empty parameters object is valid
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
        chat.system = chat.system + " " + self.system
        if message:
            chat.add_message(role="user", content=message)
        while True:
            if on_step_callback:
                on_step_callback(chat)
            serializable_messages = [m.__dict__ for m in chat.messages]

            # FIX 2: Pass the generated JSON schemas, NOT the raw Python function objects
            tool_schemas = self._get_serialized_tools()
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
