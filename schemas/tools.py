"""Tool execution schema models."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ToolExecution(BaseModel):
    """Record of a tool execution in an agent workflow."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique identifier for this tool call")
    tool_name: str = Field(..., description="Name of the tool called")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Result returned by the tool")
    status: ToolStatus = Field(default=ToolStatus.SUCCESS, description="Execution status")
    latency_ms: Optional[float] = Field(None, description="Execution latency in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
