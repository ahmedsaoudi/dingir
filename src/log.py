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

    def __init__(self) -> None:
        self._entries: List[LogEntry] = []

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
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full log to a plain dict."""
        return {"entries": [entry.to_dict() for entry in self._entries]}

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
        if not self._entries:
            return "Log(empty)"

        lines: List[str] = [f"Log({len(self._entries)} entries)"]
        for entry in self._entries:
            meta = f"  {entry.metadata}" if entry.metadata else ""
            lines.append(
                f"  [{entry.timestamp}] {entry.entry_type}"
                f" ({entry.agent_name}): {entry.content}{meta}"
            )
        return "\n".join(lines)
