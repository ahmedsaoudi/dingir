import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional

from dingir.agents.guards import GuardError
from dingir.agents.log import Log
from dingir.agents.memory import Memory


class Agent:
    """A completely stateless orchestration single-loop execution boundary engine."""

    def __init__(
        self,
        model: Any,
        system: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        guards: Optional[List[Any]] = None,
    ):
        self.model = model
        self.description = description or system[:100]
        self.tools = tools or []
        self.guards = guards or []

        self.system = system

        if name:
            self.__name__ = name
        else:
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
            guards=self.guards,
        )
        self.memory = Memory(system=self.system)

        # Step-level trackers for logging
        self._current_step_guards = []
        self._current_step_exceptions = []

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

    def record_guard_encounter(
        self,
        guard: Any,
        status: str,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        from dingir.agents.log import _serialize_guard
        serialized = _serialize_guard(guard)
        encounter = {
            "name": serialized.get("name", guard.__class__.__name__),
            "status": status,
            "tool_name": tool_name,
            "arguments": arguments,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._current_step_guards.append(encounter)

    def record_exception_encounter(self, exception: Exception, context: Optional[str] = None) -> None:
        exc_data = {
            "type": exception.__class__.__name__,
            "message": str(exception),
            "context": context,
        }
        if exc_data not in self._current_step_exceptions:
            self._current_step_exceptions.append(exc_data)

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
        self.log.guards = self.guards

        self._current_step_guards = []
        self._current_step_exceptions = []

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
        except GuardError as e:
            self.record_exception_encounter(e, f"tool_guard_execution:{name}")
            return f"GUARD ERROR: {str(e)}"
        except Exception as e:
            self.record_exception_encounter(e, f"tool_execution:{name}")
            import traceback
            tb = traceback.format_exc()
            self.log.record(
                "exception",
                {
                    "exception_type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": tb,
                    "tool_name": name,
                    "arguments": args,
                },
                agent_name=self.__name__,
            )
            return f"EXECUTION FAULT: {str(e)}"
        finally:
            unwrapped = tool_func
            while hasattr(unwrapped, "__wrapped__"):
                unwrapped = unwrapped.__wrapped__
            if isinstance(unwrapped, Agent):
                self.log.merge(unwrapped.log)
            _active_agent.reset(token)


    def respond(
        self,
        message: Optional[str] = None,
        on_step_callback: Optional[Callable[["Agent"], None] | List[Callable[["Agent"], None]]] = None,
    ):
        from dingir.agents.guards import _active_agent
        token = _active_agent.set(self)
        try:
            # Combine default guards and runtime callbacks
            cbs = list(self.guards)
            if on_step_callback:
                if isinstance(on_step_callback, (list, tuple)):
                    cbs.extend(on_step_callback)
                else:
                    cbs.append(on_step_callback)

            self.log.guards = cbs

            if message:
                self.memory.add(role="user", content=message)
                self.log.record(
                    "message",
                    {"role": "user", "content": message},
                    agent_name=self.__name__,
                )
            while True:
                # Reset step-level trackers for this iteration
                self._current_step_guards = []
                self._current_step_exceptions = []

                # Execute step callbacks at the start of the iteration
                if cbs:
                    for cb in cbs:
                        cb(self)

                serializable_messages = self.memory.to_messages()

                try:
                    result = self.model.request(
                        self.memory.system, serializable_messages, self.tools
                    )
                except Exception as e:
                    self.record_exception_encounter(e, "model_request")
                    raise

                reasoning = result.get("reasoning_content")

                if result.get("tool_calls"):
                    self.memory.add(
                        role="assistant",
                        content=result["content"],
                        tool_calls=result["tool_calls"],
                    )
                    log_content = {
                        "role": "assistant",
                        "content": result["content"],
                        "tool_calls": result["tool_calls"],
                    }
                    if reasoning:
                        log_content["reasoning_content"] = reasoning
                    msg_entry = self.log.record(
                        "message",
                        log_content,
                        agent_name=self.__name__,
                    )
                    if cbs:
                        for cb in cbs:
                            cb(self)
                    for tc in result["tool_calls"]:
                        self.log.record(
                            "tool_call",
                            {"name": tc["name"], "arguments": tc["arguments"]},
                            agent_name=self.__name__,
                        )
                        output = self._execute_tool_sync(
                            tc["name"], tc["arguments"]
                        )
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
                    
                    # Enrich metadata at the end of the iteration
                    msg_entry.metadata = {
                        "agent_name": self.__name__,
                        "driver": self.model.__class__.__name__,
                        "guards_encountered": list(self._current_step_guards),
                        "exceptions_encountered": list(self._current_step_exceptions),
                    }
                    continue

                self.memory.add(role="assistant", content=result["content"])
                log_content = {"role": "assistant", "content": result["content"]}
                if reasoning:
                    log_content["reasoning_content"] = reasoning
                msg_entry = self.log.record(
                    "message",
                    log_content,
                    agent_name=self.__name__,
                )
                if cbs:
                    for cb in cbs:
                        cb(self)
                
                # Enrich metadata at the end of the final iteration
                msg_entry.metadata = {
                    "agent_name": self.__name__,
                    "driver": self.model.__class__.__name__,
                    "guards_encountered": list(self._current_step_guards),
                    "exceptions_encountered": list(self._current_step_exceptions),
                }
                break
        except Exception as e:
            self.record_exception_encounter(e, "agent_respond")
            if not getattr(e, "_agent_logged", False):
                import traceback
                self.log.record(
                    "exception",
                    {
                        "exception_type": e.__class__.__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    agent_name=self.__name__,
                )
                try:
                    e._agent_logged = True
                except AttributeError:
                    pass
            raise
        finally:
            _active_agent.reset(token)

    def stream(
        self,
        message: Optional[str] = None,
        on_step_callback: Optional[Callable[["Agent"], None] | List[Callable[["Agent"], None]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming variant of respond(). Yields delta dicts as the model generates.

        Runs the same agentic loop as respond() (guards, tool calls, memory, logging)
        but uses model.request_stream() so content and reasoning tokens can be
        surfaced to the caller incrementally.

        Chunk protocol:
          {"type": "content_delta", "content": "..."}      — partial text token
          {"type": "reasoning_delta", "content": "..."}     — partial reasoning token
          {"type": "tool_call", "name": "...", "arguments": "..."}  — a tool was called
          {"type": "tool_result", "name": "...", "output": "..."}   — tool result
          {"type": "done", "content": "full final text"}    — final assembled response
        """
        from dingir.agents.guards import _active_agent
        token = _active_agent.set(self)
        try:
            # Combine default guards and runtime callbacks
            cbs = list(self.guards)
            if on_step_callback:
                if isinstance(on_step_callback, (list, tuple)):
                    cbs.extend(on_step_callback)
                else:
                    cbs.append(on_step_callback)

            self.log.guards = cbs

            if message:
                self.memory.add(role="user", content=message)
                self.log.record(
                    "message",
                    {"role": "user", "content": message},
                    agent_name=self.__name__,
                )
            while True:
                # Reset step-level trackers for this iteration
                self._current_step_guards = []
                self._current_step_exceptions = []

                # Execute step callbacks at the start of the iteration
                if cbs:
                    for cb in cbs:
                        cb(self)

                serializable_messages = self.memory.to_messages()

                # Stream the model response, accumulating the final result
                result = None
                try:
                    for chunk in self.model.request_stream(
                        self.memory.system, serializable_messages, self.tools
                    ):
                        if chunk["type"] in ("content_delta", "reasoning_delta"):
                            yield chunk
                        elif chunk["type"] == "done":
                            result = chunk
                except Exception as e:
                    self.record_exception_encounter(e, "model_request_stream")
                    raise

                if result is None:
                    # Shouldn't happen, but guard against it
                    result = {"content": "", "tool_calls": None, "reasoning_content": None}

                reasoning = result.get("reasoning_content")

                if result.get("tool_calls"):
                    self.memory.add(
                        role="assistant",
                        content=result["content"],
                        tool_calls=result["tool_calls"],
                    )
                    log_content = {
                        "role": "assistant",
                        "content": result["content"],
                        "tool_calls": result["tool_calls"],
                    }
                    if reasoning:
                        log_content["reasoning_content"] = reasoning
                    msg_entry = self.log.record(
                        "message",
                        log_content,
                        agent_name=self.__name__,
                    )
                    if cbs:
                        for cb in cbs:
                            cb(self)
                    for tc in result["tool_calls"]:
                        self.log.record(
                            "tool_call",
                            {"name": tc["name"], "arguments": tc["arguments"]},
                            agent_name=self.__name__,
                        )
                        yield {"type": "tool_call", "name": tc["name"], "arguments": tc["arguments"]}
                        output = self._execute_tool_sync(
                            tc["name"], tc["arguments"]
                        )
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
                        yield {"type": "tool_result", "name": tc["name"], "output": output}

                    # Enrich metadata
                    msg_entry.metadata = {
                        "agent_name": self.__name__,
                        "driver": self.model.__class__.__name__,
                        "guards_encountered": list(self._current_step_guards),
                        "exceptions_encountered": list(self._current_step_exceptions),
                    }
                    continue

                self.memory.add(role="assistant", content=result["content"])
                log_content = {"role": "assistant", "content": result["content"]}
                if reasoning:
                    log_content["reasoning_content"] = reasoning
                msg_entry = self.log.record(
                    "message",
                    log_content,
                    agent_name=self.__name__,
                )
                if cbs:
                    for cb in cbs:
                        cb(self)
                
                # Enrich metadata
                msg_entry.metadata = {
                    "agent_name": self.__name__,
                    "driver": self.model.__class__.__name__,
                    "guards_encountered": list(self._current_step_guards),
                    "exceptions_encountered": list(self._current_step_exceptions),
                }
                yield {"type": "done", "content": result["content"]}
                break
        except Exception as e:
            self.record_exception_encounter(e, "agent_stream")
            if not getattr(e, "_agent_logged", False):
                import traceback
                self.log.record(
                    "exception",
                    {
                        "exception_type": e.__class__.__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    agent_name=self.__name__,
                )
                try:
                    e._agent_logged = True
                except AttributeError:
                    pass
            raise
        finally:
            _active_agent.reset(token)
