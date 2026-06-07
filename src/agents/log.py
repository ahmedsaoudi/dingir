from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


ENTRY_TYPES = {"config", "message", "tool_call", "tool_result", "subagent_log"}


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
    ) -> None:
        self._entries: List[LogEntry] = []
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.agent_name = agent_name
        self.model_id = model_id
        self.model_config = model_config

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
            if isinstance(t, str):
                serialized_tools.append({"name": t})
            elif isinstance(t, dict):
                serialized_tools.append(t)
            elif hasattr(t, "_dingir_schema"):
                func = t._dingir_schema.get("function", {})
                serialized_tools.append({
                    "name": func.get("name", getattr(t, "__name__", "unknown")),
                    "description": func.get("description", getattr(t, "__doc__", "No description provided.")),
                    "parameters": func.get("parameters", {}),
                })
            elif hasattr(t, "__name__"):
                serialized_tools.append({
                    "name": t.__name__,
                    "description": getattr(t, "__doc__", "") or "No description provided.",
                })
            elif hasattr(t, "name"):
                serialized_tools.append({
                    "name": t.name,
                    "description": getattr(t, "description", "") or "No description provided.",
                })
            else:
                serialized_tools.append({"name": str(t)})

        return {
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "model_config": self.model_config,
            "system_prompt": self.system_prompt,
            "tools": serialized_tools,
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

    def __str__(self) -> str:
        return _format_log(
            system_prompt=self.system_prompt,
            tools=self.tools,
            entries=self._entries,
            agent_name=self.agent_name,
            model_id=self.model_id,
            model_config=self.model_config,
            indent_level=0,
        )


def _format_log(
    system_prompt: Optional[str],
    tools: List[Any],
    entries: List[Any],
    agent_name: Optional[str] = None,
    model_id: Optional[str] = None,
    model_config: Optional[Any] = None,
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
        if isinstance(model_config, dict):
            try:
                import json
                config_str = json.dumps(model_config)
            except Exception:
                config_str = str(model_config)
        else:
            config_str = str(model_config)
        lines.append(f"{indent}Model Config: {config_str}")

    # 3. Tools
    if tools:
        serialized_tools = []
        for t in tools:
            if isinstance(t, str):
                serialized_tools.append(t)
            elif isinstance(t, dict):
                serialized_tools.append(t.get("name", str(t)))
            else:
                serialized_tools.append(getattr(t, "__name__", str(t)))
        tools_str = ", ".join(serialized_tools)
        lines.append(f"{indent}Tools: {tools_str}")

    # 4. System Prompt
    if system_prompt:
        indented_prompt = system_prompt.replace("\n", "\n" + indent + "  ")
        lines.append(f"{indent}System Prompt:\n{indent}  {indented_prompt}")

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

            if e_type == "subagent_log" and isinstance(e_content, dict):
                lines.append(f"{indent}  [{e_timestamp}] subagent_log for {e_name or 'subagent'}:")
                sub_system = e_content.get("system_prompt")
                sub_tools = e_content.get("tools", [])
                sub_entries = e_content.get("entries", [])
                sub_agent = e_content.get("agent_name") or e_name
                sub_model = e_content.get("model_id")
                sub_config = e_content.get("model_config")
                sub_str = _format_log(
                    system_prompt=sub_system,
                    tools=sub_tools,
                    entries=sub_entries,
                    agent_name=sub_agent,
                    model_id=sub_model,
                    model_config=sub_config,
                    indent_level=indent_level + 4,
                )
                lines.append(sub_str)
            else:
                content_str = str(e_content)
                if "\n" in content_str:
                    content_str = content_str.replace("\n", "\n" + indent + "    ")
                lines.append(
                    f"{indent}  [{e_timestamp}] {e_type}: {content_str}{meta}"
                )
    return "\n".join(lines)
