import json
from typing import Any, Callable, Dict, List, Optional

from dingir.agents.guards import GuardError
from dingir.agents.log import Log
from dingir.agents.memory import Memory


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
                "You can do function calling with the following functions:\n\n"
                f"{tool_descriptions_json}\n\n"
                "TOOL EXECUTION GUIDELINES:\n"
                "1.Analyze the user query and select the most appropriate tool. "
                "If no tool is needed, respond directly using your general knowledge.\n"
                "2.When calling a tool, strictly adhere to the types and constraints "
                "defined in the schema.\n"
                "3.You MUST trigger a tool call by formatting it in json. "
                "Do not add any other text before or after this JSON block when\n"
                "calling a tool."
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

        # Agent-owned log and memory
        self.log = Log(
            system_prompt=self.system,
            tools=self.tools,
            agent_name=self.__name__,
            model_id=self.model.id,
            model_config=self.model.config.__dict__
            if hasattr(self.model.config, "__dict__")
            else str(self.model.config),
        )
        self.memory = Memory(system=self.system)

        # Record initial configuration
        self.log.record(
            "config",
            {
                "model_id": self.model.id,
                "config": self.model.config.__dict__
                if hasattr(self.model.config, "__dict__")
                else str(self.model.config),
            },
            agent_name=self.__name__,
        )

    def __call__(self, instruction: str) -> str:
        """Executes subagent handoffs cleanly while preserving internal state isolation."""
        # Subagent mode: use a temporary memory, run the task, return the answer.
        # The caller (parent agent) is responsible for merging self.log afterwards.
        self.memory = Memory(system=self.system)
        self.log.clear()
        self.log.system_prompt = self.system
        self.log.tools = self.tools
        self.log.agent_name = self.__name__
        self.log.model_id = self.model.id
        self.log.model_config = self.model.config.__dict__ \
            if hasattr(self.model.config, "__dict__") \
            else str(self.model.config)

        # Re-record config for this fresh run
        self.log.record(
            "config",
            {
                "model_id": self.model.id,
                "config": self.model.config.__dict__
                if hasattr(self.model.config, "__dict__")
                else str(self.model.config),
            },
            agent_name=self.__name__,
        )

        self.respond(message=instruction)
        return (
            self.memory.last.content if self.memory.last else ""
        )

    def _execute_tool_sync(self, name: str, args: Any) -> str:
        from dingir.agents.guards import _active_agent

        tool_func = next((t for t in self.tools if t.__name__ == name), None)
        if not tool_func:
            return f"ERROR: Function execution endpoint '{name}' unavailable."

        token = _active_agent.set(self)
        try:
            resolved_args = json.loads(args) if isinstance(args, str) else args
            if isinstance(resolved_args, dict):
                result = str(tool_func(**resolved_args))
            else:
                result = str(tool_func(resolved_args))

            return result
        except GuardError:
            raise
        except Exception as e:
            return f"EXECUTION FAULT: {str(e)}"
        finally:
            if isinstance(tool_func, Agent):
                self.log.merge(tool_func.log)
            _active_agent.reset(token)

    def _get_serialized_tools(self) -> List[Dict[str, Any]]:
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
        for m in self.memory.messages:
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
        message: Optional[str] = None,
        on_step_callback: Optional[Callable[["Agent"], None] | List[Callable[["Agent"], None]]] = None,
    ):
        if on_step_callback:
            cbs = on_step_callback if isinstance(on_step_callback, (list, tuple)) else [on_step_callback]
            self.log.guards = list(cbs)
        else:
            self.log.guards = []

        if message:
            self.memory.add(role="user", content=message)
            self.log.record(
                "message",
                {"role": "user", "content": message},
                agent_name=self.__name__,
            )
        while True:
            # Execute step callbacks at the start of the iteration
            if on_step_callback:
                cbs = on_step_callback if isinstance(on_step_callback, (list, tuple)) else [on_step_callback]
                for cb in cbs:
                    cb(self)

            serializable_messages = self.memory.to_messages()

            tool_schemas = self._get_serialized_tools()
            result = self.model.request(
                self.memory.system, serializable_messages, tool_schemas
            )

            if result.get("tool_calls"):
                self.memory.add(
                    role="assistant",
                    content=result["content"],
                    tool_calls=result["tool_calls"],
                )
                self.log.record(
                    "message",
                    {
                        "role": "assistant",
                        "content": result["content"],
                        "tool_calls": result["tool_calls"],
                    },
                    agent_name=self.__name__,
                )
                if on_step_callback:
                    if isinstance(on_step_callback, (list, tuple)):
                        for cb in on_step_callback:
                            cb(self)
                    else:
                        on_step_callback(self)
                for tc in result["tool_calls"]:
                    self.log.record(
                        "tool_call",
                        {"name": tc["name"], "arguments": tc["arguments"]},
                        agent_name=self.__name__,
                    )
                    try:
                        output = self._execute_tool_sync(
                            tc["name"], tc["arguments"]
                        )
                    except GuardError as e:
                        err_msg = f"GUARD ERROR: {str(e)}"
                        self.memory.add(
                            role="tool",
                            content=err_msg,
                            tool_call_id=tc.get("id", "call_idx"),
                            name=tc["name"],
                        )
                        self.log.record(
                            "tool_result",
                            {
                                "name": tc["name"],
                                "tool_call_id": tc.get("id", "call_idx"),
                                "output": err_msg,
                            },
                            agent_name=self.__name__,
                        )
                        raise

                    self.memory.add(
                        role="tool",
                        content=output,
                        tool_call_id=tc.get("id", "call_idx"),
                        name=tc["name"],
                    )
                    self.log.record(
                        "tool_result",
                        {
                            "name": tc["name"],
                            "tool_call_id": tc.get("id", "call_idx"),
                            "output": output,
                        },
                        agent_name=self.__name__,
                    )
                continue

            self.memory.add(role="assistant", content=result["content"])
            self.log.record(
                "message",
                {"role": "assistant", "content": result["content"]},
                agent_name=self.__name__,
            )
            if on_step_callback:
                if isinstance(on_step_callback, (list, tuple)):
                    for cb in on_step_callback:
                        cb(self)
                else:
                    on_step_callback(self)
            break
