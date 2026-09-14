"""Tool execution schema models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(str, Enum):
    """Status of a tool execution (contracts/v0.1's three wire values)."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class ToolExecution(BaseModel):
    """Record of a tool execution in an agent workflow."""

    model_config = ConfigDict(frozen=True)

    tool_execution_id: str = Field(..., description="Unique identifier for this tool call")
    tool_name: str = Field(..., description="Name of the tool called")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    result: Any | None = Field(None, description="Result returned by the tool")
    status: ToolStatus = Field(default=ToolStatus.SUCCESS, description="Execution status")
    started_at: str | None = Field(None, description="RFC 3339 start timestamp")
    completed_at: str | None = Field(None, description="RFC 3339 completion timestamp")
    latency_ms: float | None = Field(None, description="Execution latency in milliseconds")
    error_message: str | None = Field(None, description="Error message if status is error")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
