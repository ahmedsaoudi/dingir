from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Message:
    role: str  # "system", "user", "assistant", or "tool"
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

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
