from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


ENTRY_TYPES = {"config", "message", "tool_call", "tool_result", "subagent_log", "guard_trigger", "exception", "error"}


def _serialize_guard(g: Any) -> Dict[str, Any]:
    """Recursively serialise a guard or callback object for metadata logging."""
    if hasattr(g, "get_details"):
        constraints = g.get_details()
        serialized_constraints = {}
        for k, v in constraints.items():
            serialized_constraints[k] = _serialize_guard_value(v)
        return {
            "name": g.__class__.__name__,
            "constraints": serialized_constraints,
        }
    else:
        return {
            "name": getattr(g, "__name__", g.__class__.__name__),
        }


def _serialize_guard_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple, set)):
        return [_serialize_guard_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize_guard_value(x) for k, x in v.items()}
    if hasattr(v, "get_details"):
        return _serialize_guard(v)
    if hasattr(v, "__name__"):
        return v.__name__
    return str(v)


@dataclass
class LogEntry:
    """A single structured entry in an agent's execution log."""

    entry_type: str
    agent_name: str
    content: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.entry_type not in ENTRY_TYPES:
            raise ValueError(
                f"Invalid entry_type {self.entry_type!r}. "
                f"Must be one of {sorted(ENTRY_TYPES)}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Log:
    """Append-only structured trace of everything an agent does.

    Stores ``LogEntry`` objects that capture configuration, messages,
    tool calls / results, and nested sub-agent logs.  Designed for
    post-hoc debugging and inspection.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        agent_name: Optional[str] = None,
        model_id: Optional[str] = None,
        model_config: Optional[Any] = None,
        guards: Optional[List[Any]] = None,
    ) -> None:
        self._entries: List[LogEntry] = []
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.agent_name = agent_name
        self.model_id = model_id
        self.model_config = model_config
        self.guards = guards or []

    # ------------------------------------------------------------------
    # Core recording
    # ------------------------------------------------------------------

    def record(
        self,
        entry_type: str,
        content: Any,
        agent_name: Optional[str] = None,
        **metadata: Any,
    ) -> LogEntry:
        """Create and append a new ``LogEntry``.

        Args:
            entry_type: One of ``ENTRY_TYPES``.
            content: Arbitrary payload matching the entry type.
            agent_name: Name of the agent that produced this entry.
            **metadata: Extra key-value pairs stored on the entry.

        Returns:
            The newly created ``LogEntry``.
        """
        entry = LogEntry(
            entry_type=entry_type,
            agent_name=agent_name or "",
            content=content,
            metadata=metadata if metadata else None,
        )
        self._entries.append(entry)
        return entry

    def merge(self, child_log: "Log") -> LogEntry:
        """Fold an entire child ``Log`` into a single ``subagent_log`` entry.

        Args:
            child_log: The child log whose entries will be nested.

        Returns:
            The ``LogEntry`` wrapping the child log.
        """
        return self.record(
            entry_type="subagent_log",
            content=child_log.to_dict(),
            agent_name=child_log.agent_name,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full log to a plain dict."""
        serialized_tools = []
        for t in self.tools:
            s_tool = None
            t_unwrapped = t
            while hasattr(t_unwrapped, "__wrapped__"):
                t_unwrapped = t_unwrapped.__wrapped__

            if isinstance(t, str):
                s_tool = {"name": t}
            elif isinstance(t, dict):
                s_tool = dict(t)
            elif hasattr(t_unwrapped, "_dingir_schema"):
                func = t_unwrapped._dingir_schema.get("function", {})
                s_tool = {
                    "name": func.get("name", getattr(t_unwrapped, "__name__", "unknown")),
                    "description": func.get("description", getattr(t_unwrapped, "__doc__", "No description provided.")),
                    "parameters": func.get("parameters", {}),
                }
            elif hasattr(t_unwrapped, "__name__"):
                s_tool = {
                    "name": t_unwrapped.__name__,
                    "description": getattr(t_unwrapped, "__doc__", "") or "No description provided.",
                }
            elif hasattr(t_unwrapped, "name"):
                s_tool = {
                    "name": t_unwrapped.name,
                    "description": getattr(t_unwrapped, "description", "") or "No description provided.",
                }
            else:
                s_tool = {"name": str(t_unwrapped)}

            tool_guards = getattr(t, "_dingir_guards", None)
            if tool_guards:
                s_tool["guards"] = [_serialize_guard(g) for g in tool_guards]
            serialized_tools.append(s_tool)

        serialized_agent_guards = [_serialize_guard(g) for g in self.guards]

        return {
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "model_config": self.model_config,
            "system_prompt": self.system_prompt,
            "tools": serialized_tools,
            "guards": serialized_agent_guards,
            "entries": [entry.to_dict() for entry in self._entries],
        }

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all recorded entries."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def format(self, indent_level: int = 0) -> str:
        """Format the log hierarchy with padding/indentation."""
        return _format_log(
            system_prompt=self.system_prompt,
            tools=self.tools,
            entries=self._entries,
            agent_name=self.agent_name,
            model_id=self.model_id,
            model_config=self.model_config,
            guards=self.guards,
            indent_level=indent_level,
        )

    def __str__(self) -> str:
        return self.format()


def _indent_text(text: str, indent_level: int) -> str:
    """Indent a block of text by a specified number of levels (each level is 2 spaces)."""
    indent = "  " * indent_level
    return "\n".join(indent + line for line in text.splitlines())


def _format_model_config(config: Any) -> str:
    """Format model configurations pythonically by excluding private keys and None values."""
    if isinstance(config, dict):
        filtered = {k: v for k, v in config.items() if not k.startswith("_") and v is not None}
        return ", ".join(f"{k}={v}" for k, v in filtered.items())
    return str(config)


def _format_log(
    system_prompt: Optional[str],
    tools: List[Any],
    entries: List[Any],
    agent_name: Optional[str] = None,
    model_id: Optional[str] = None,
    model_config: Optional[Any] = None,
    guards: Optional[List[Any]] = None,
    indent_level: int = 0,
) -> str:
    indent = "  " * indent_level
    lines = []

    # 1. Agent Name & Model Info
    agent_str = f"AGENT: {agent_name}" if agent_name else "AGENT"
    lines.append(f"{indent}=== {agent_str} ===")
    if model_id:
        lines.append(f"{indent}Model: {model_id}")

    # 2. Model Config
    if model_config is not None:
        lines.append(f"{indent}Model Config: {_format_model_config(model_config)}")

    # 2.5 Guards
    if guards:
        serialized_guards = []
        for g in guards:
            if isinstance(g, dict):
                name = g.get("name", str(g))
                constraints = g.get("constraints")
                if constraints:
                    if isinstance(constraints, dict):
                        constraints_clean = {k: v for k, v in constraints.items() if k != "log"}
                        constraints_str = ", ".join(f"{k}={v}" for k, v in constraints_clean.items())
                    else:
                        constraints_str = str(constraints)
                    serialized_guards.append(f"{name}({constraints_str})")
                else:
                    serialized_guards.append(name)
            else:
                if hasattr(g, "get_details"):
                    constraints = g.get_details()
                    if isinstance(constraints, dict):
                        constraints_clean = {k: v for k, v in constraints.items() if k != "log"}
                        constraints_str = ", ".join(f"{k}={v}" for k, v in constraints_clean.items())
                    else:
                        constraints_str = str(constraints)
                    serialized_guards.append(f"{g.__class__.__name__}({constraints_str})")
                else:
                    serialized_guards.append(getattr(g, "__name__", str(g)))
        guards_str = ", ".join(serialized_guards)
        lines.append(f"{indent}Guards: {guards_str}")

    # 3. Tools
    if tools:
        serialized_tools = []
        for t in tools:
            if isinstance(t, str):
                serialized_tools.append(t)
            elif isinstance(t, dict):
                name = t.get("name", str(t))
                t_guards = t.get("guards")
                if t_guards:
                    t_guards_list = []
                    for g in t_guards:
                        g_name = g.get("name")
                        g_const = g.get("constraints")
                        if g_const and isinstance(g_const, dict):
                            g_const_clean = {k: v for k, v in g_const.items() if k != "log"}
                            g_const_str = ", ".join(f"{k}={v}" for k, v in g_const_clean.items())
                            t_guards_list.append(f"{g_name}({g_const_str})")
                        elif g_const:
                            t_guards_list.append(f"{g_name}({g_const})")
                        else:
                            t_guards_list.append(g_name)
                    t_guards_str = ", ".join(t_guards_list)
                    serialized_tools.append(f"{name} [Guards: {t_guards_str}]")
                else:
                    serialized_tools.append(name)
            else:
                t_unwrapped = t
                while hasattr(t_unwrapped, "__wrapped__"):
                    t_unwrapped = t_unwrapped.__wrapped__
                name = getattr(t_unwrapped, "__name__", str(t_unwrapped))
                t_guards = getattr(t, "_dingir_guards", [])
                if t_guards:
                    t_guards_list = []
                    for g in t_guards:
                        g_name = g.__class__.__name__
                        if hasattr(g, "get_details"):
                            g_const = g.get_details()
                            if isinstance(g_const, dict):
                                g_const_clean = {k: v for k, v in g_const.items() if k != "log"}
                                g_const_str = ", ".join(f"{k}={v}" for k, v in g_const_clean.items())
                                t_guards_list.append(f"{g_name}({g_const_str})")
                            else:
                                t_guards_list.append(f"{g_name}({g_const})")
                        else:
                            t_guards_list.append(getattr(g, "__name__", str(g)))
                    t_guards_str = ", ".join(t_guards_list)
                    serialized_tools.append(f"{name} [Guards: {t_guards_str}]")
                else:
                    serialized_tools.append(name)
        tools_str = ", ".join(serialized_tools)
        lines.append(f"{indent}Tools: {tools_str}")

    # 4. System Prompt
    if system_prompt:
        lines.append(f"{indent}System Prompt:")
        lines.append(_indent_text(system_prompt, indent_level + 1))

    # 5. Entries
    lines.append(f"{indent}Entries ({len(entries)}):")
    if not entries:
        lines.append(f"{indent}  (No entries)")
    else:
        for entry in entries:
            if isinstance(entry, LogEntry):
                e_type = entry.entry_type
                e_name = entry.agent_name
                e_content = entry.content
                e_timestamp = entry.timestamp
                e_metadata = entry.metadata
            else:
                e_type = entry.get("entry_type", "")
                e_name = entry.get("agent_name", "")
                e_content = entry.get("content")
                e_timestamp = entry.get("timestamp", "")
                e_metadata = entry.get("metadata")

            meta = f"  {e_metadata}" if e_metadata else ""

            if e_type == "config":
                lines.append(f"{indent}  [{e_timestamp}] [CONFIG]:")
                if isinstance(e_content, dict):
                    m_id = e_content.get("model_id")
                    config_val = e_content.get("config")
                    if m_id:
                        lines.append(f"{indent}    Model ID: {m_id}")
                    if config_val is not None:
                        lines.append(f"{indent}    Configuration: {_format_model_config(config_val)}")
                else:
                    lines.append(f"{indent}    {e_content}")

            elif e_type == "subagent_log" and isinstance(e_content, dict):
                lines.append(f"{indent}  [{e_timestamp}] [SUBAGENT LOG] for {e_name or 'subagent'}:")
                sub_system = e_content.get("system_prompt")
                sub_tools = e_content.get("tools", [])
                sub_entries = e_content.get("entries", [])
                sub_agent = e_content.get("agent_name") or e_name
                sub_model = e_content.get("model_id")
                sub_config = e_content.get("model_config")
                sub_guards = e_content.get("guards", [])
                sub_str = _format_log(
                    system_prompt=sub_system,
                    tools=sub_tools,
                    entries=sub_entries,
                    agent_name=sub_agent,
                    model_id=sub_model,
                    model_config=sub_config,
                    guards=sub_guards,
                    indent_level=indent_level + 4,
                )
                lines.append(sub_str)

            elif e_type == "guard_trigger" and isinstance(e_content, dict):
                g_type = e_content.get("guard_type", "Guard")
                err_msg = e_content.get("error", "")
                status = e_content.get("status", "failed")
                constraints = e_content.get("constraints", {})
                tool_msg = f" on tool '{e_content['applied_to_tool']}'" if "applied_to_tool" in e_content else ""
                agent_msg = f" on agent '{e_content['applied_to_agent']}'" if "applied_to_agent" in e_content else ""
                tool_args = e_content.get("tool_arguments")

                status_upper = status.upper()
                lines.append(
                    f"{indent}  [{e_timestamp}] [GUARD TRIGGER]: Guard '{g_type}'{tool_msg}{agent_msg} {status_upper}!"
                )
                if err_msg:
                    lines.append(f"{indent}    Error: {err_msg}")
                if constraints:
                    if isinstance(constraints, dict):
                        constraints_clean = {k: v for k, v in constraints.items() if k != "log"}
                        constraints_str = ", ".join(f"{k}={v}" for k, v in constraints_clean.items())
                    else:
                        constraints_str = str(constraints)
                    if constraints_str:
                        lines.append(f"{indent}    Constraints: {constraints_str}")
                if tool_args:
                    lines.append(f"{indent}    Arguments: {tool_args}")

            elif e_type == "message" and isinstance(e_content, dict):
                role = e_content.get("role", "")
                content = e_content.get("content", "")
                reasoning = e_content.get("reasoning_content") or e_content.get("reasoning") or ""
                tool_calls = e_content.get("tool_calls")

                if role == "user":
                    if content:
                        lines.append(f"{indent}  [{e_timestamp}] [USER MESSAGE]:")
                        lines.append(_indent_text(content, indent_level + 2))
                elif role == "assistant":
                    if reasoning:
                        lines.append(f"{indent}  [{e_timestamp}] [REASONING]:")
                        lines.append(_indent_text(reasoning, indent_level + 2))
                    if content:
                        lines.append(f"{indent}  [{e_timestamp}] [ASSISTANT MESSAGE]:")
                        lines.append(_indent_text(content, indent_level + 2))
                    if tool_calls:
                        lines.append(f"{indent}  [{e_timestamp}] [ASSISTANT TOOL CALLS]:")
                        import json
                        for tc in tool_calls:
                            tc_name = tc.get("name")
                            tc_args = tc.get("arguments")
                            if isinstance(tc_args, dict):
                                tc_args_str = ", ".join(f"{k}={v!r}" for k, v in tc_args.items())
                            elif isinstance(tc_args, str):
                                try:
                                    resolved_args = json.loads(tc_args)
                                    if isinstance(resolved_args, dict):
                                        tc_args_str = ", ".join(f"{k}={v!r}" for k, v in resolved_args.items())
                                    else:
                                        tc_args_str = str(resolved_args)
                                except Exception:
                                    tc_args_str = tc_args
                            else:
                                tc_args_str = str(tc_args)
                            lines.append(f"{indent}    - {tc_name}({tc_args_str})")
                else:
                    role_upper = role.upper()
                    if content:
                        lines.append(f"{indent}  [{e_timestamp}] [{role_upper} MESSAGE]:")
                        lines.append(_indent_text(content, indent_level + 2))

            elif e_type == "tool_call" and isinstance(e_content, dict):
                name = e_content.get("name")
                args = e_content.get("arguments")
                import json
                if isinstance(args, dict):
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                elif isinstance(args, str):
                    try:
                        resolved_args = json.loads(args)
                        if isinstance(resolved_args, dict):
                            args_str = ", ".join(f"{k}={v!r}" for k, v in resolved_args.items())
                        else:
                            args_str = str(resolved_args)
                    except Exception:
                        args_str = args
                else:
                    args_str = str(args)
                lines.append(f"{indent}  [{e_timestamp}] [TOOL CALL]: {name}({args_str})")

            elif e_type == "tool_result" and isinstance(e_content, dict):
                name = e_content.get("name")
                output = e_content.get("output", "")
                call_id = e_content.get("tool_call_id", "call_idx")

                lines.append(f"{indent}  [{e_timestamp}] [TOOL RESULT] {name} (ID: {call_id}):")
                output_str = str(output)
                if output_str:
                    lines.append(_indent_text(output_str, indent_level + 2))

            elif e_type in ("exception", "error") and isinstance(e_content, dict):
                exc_type = e_content.get("exception_type", "Exception")
                message = e_content.get("message", "")
                traceback_str = e_content.get("traceback", "")
                tool_name = e_content.get("tool_name")
                tool_msg = f" during execution of tool '{tool_name}'" if tool_name else ""

                lines.append(
                    f"{indent}  [{e_timestamp}] [EXCEPTION]: {exc_type}{tool_msg}: {message}"
                )
                if traceback_str:
                    lines.append(f"{indent}    Traceback:")
                    lines.append(_indent_text(traceback_str, indent_level + 3))

            elif e_type in ("exception", "error"):
                lines.append(f"{indent}  [{e_timestamp}] [{e_type.upper()}]: {e_content}")

            else:
                content_str = str(e_content)
                lines.append(
                    f"{indent}  [{e_timestamp}] [{e_type.upper()}]: {content_str}{meta}"
                )
    return "\n".join(lines)
