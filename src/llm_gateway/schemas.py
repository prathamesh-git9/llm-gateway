"""Wire schemas for the gateway's OpenAI-compatible chat completions surface."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    """Deliberately a subset of the OpenAI schema: the fields a router needs."""

    model: str
    messages: list[Message]
    max_tokens: int = Field(default=1024, ge=1, le=128_000)
    stream: bool = False
    # Gateway-specific knobs. Prefixed so they never collide with upstream fields.
    x_cache: bool = True
    x_tenant: str = "default"

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v

    def prompt_text(self) -> str:
        """Flattened conversation, used as the cache key and embedding input.

        The space after the role is load-bearing: the semantic tier tokenises on
        whitespace, so `user:what` would fuse the role into the first word and
        cost every prompt one spurious term mismatch against its own paraphrase.
        """
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    # Router provenance. This is the part an SRE actually reads in an incident.
    gateway: dict[str, Any] = Field(default_factory=dict)
