from dataclasses import dataclass, field
from typing import Any, Dict
import uuid

@dataclass
class MCPMessage:
    sender: str
    receiver: str
    type: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "trace_id": self.trace_id,
            "payload": self.payload
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return MCPMessage(
            sender=data["sender"],
            receiver=data["receiver"],
            type=data["type"],
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            payload=data.get("payload", {})
        )
