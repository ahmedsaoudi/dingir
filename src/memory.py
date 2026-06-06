from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant", or "tool"
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def __str__(self) -> str:
        """Format an individual message as a human-readable line."""
        header = f"{self.role.upper()}"
        if self.name:
            header += f" ({self.name})"

        lines = [f"{header}: {self.content or ''}"]

        if self.tool_calls:
            lines.append(f"  Tool Calls: {self.tool_calls}")
        if self.tool_call_id:
            lines.append(f"  Tool Call ID: {self.tool_call_id}")

        return "\n".join(lines)


class Memory:
    """Clean conversation context that gets sent to the LLM."""

    def __init__(self, system: Optional[str] = None):
        self.system: Optional[str] = system
        self.messages: List[Message] = []

    def add(self, role: str, content: Optional[str] = None, **kwargs):
        """Create and append a Message to the conversation."""
        self.messages.append(Message(role=role, content=content, **kwargs))

    @property
    def last(self) -> Optional[Message]:
        """Return the last message, or None if the conversation is empty."""
        return self.messages[-1] if self.messages else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full conversation state to a plain dict."""
        return {
            "system": self.system,
            "messages": [asdict(m) for m in self.messages],
        }

    def to_messages(self) -> List[Dict[str, Any]]:
        """Return message dicts suitable for LLM API calls."""
        return [m.__dict__ for m in self.messages]

    def clear(self):
        """Reset the messages list, keeping the system prompt."""
        self.messages = []

    def __iter__(self):
        """Yield system message first (if set), then all conversation messages."""
        if self.system:
            yield Message(role="system", content=self.system)
        yield from self.messages

    def __len__(self) -> int:
        """Number of messages (not counting system)."""
        return len(self.messages)

    def __getitem__(self, index) -> Message:
        """Index into the messages list: memory[0], memory[-1]."""
        return self.messages[index]

    def __str__(self) -> str:
        """Return a human-readable transcript of the conversation."""
        elements = []

        if self.system:
            elements.append(f"SYSTEM PROMPT:\n{self.system}")
            elements.append("-" * 40)

        if self.messages:
            elements.append("\n".join(str(m) for m in self.messages))
        else:
            elements.append("[Empty Memory]")

        return "\n".join(elements)
