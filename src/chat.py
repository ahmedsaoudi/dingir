from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Message:
    role: str  # "system", "user", "assistant", or "tool"
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def __str__(self) -> str:
        """Optional helper to format individual messages cleanly."""
        header = f"{self.role.upper()}"
        if self.name:
            header += f" ({self.name})"
        
        lines = [f"{header}: {self.content or ''}"]
        
        if self.tool_calls:
            lines.append(f"  Tool Calls: {self.tool_calls}")
        if self.tool_call_id:
            lines.append(f"  Tool Call ID: {self.tool_call_id}")
            
        return "\n".join(lines)

class Chat:
    """An explicit, mutable timeline wrapper over serializable standard Python primitives."""
    def __init__(self, system: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None):
        self.system: Optional[str] = system
        self.messages: List[Message] = []
        if messages:
            for m in messages:
                self.messages.append(Message(**m))

    def add_message(self, role: str, content: Optional[str], **kwargs):
        self.messages.append(Message(role=role, content=content, **kwargs))

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "messages": [asdict(m) for m in self.messages]
        }

    def __str__(self) -> str:
        """Returns a human-readable transcript of the entire chat history."""
        elements = []
        
        # 1. Include the system prompt if it exists
        if self.system:
            elements.append(f"SYSTEM PROMPT:\n{self.system}")
            elements.append("-" * 40)
        
        # 2. Append all messages using the Message.__str__ formatting
        if self.messages:
            elements.append("\n".join(str(m) for m in self.messages))
        else:
            elements.append("[Empty Chat Session]")
            
        return "\n".join(elements)

    def __iter__(self):
        """Allows direct iteration over the chat's messages."""
        return iter(self.messages)

    def __len__(self) -> int:
        """Returns the number of messages in the chat: len(chat)"""
        return len(self.messages)

    def __getitem__(self, index) -> Message:
        """Allows indexing: chat[0] or slicing: chat[-1]"""
        return self.messages[index]

