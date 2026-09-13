"""Instruction schema models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class InstructionType(str, Enum):
    """Types of instructions that can be evaluated."""

    GROUNDING = "grounding"
    FORMAT = "format"
    SCOPE = "scope"
    SAFETY = "safety"
    STYLE = "style"
    TOOL_USAGE = "tool_usage"
    CITATION = "citation"
    CUSTOM = "custom"


class Priority(str, Enum):
    """Priority levels for instructions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Instruction(BaseModel):
    """Instruction for the AI system to follow."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique identifier for this instruction")
    text: str = Field(..., description="The instruction text")
    priority: Priority = Field(default=Priority.MEDIUM, description="Priority level")
    type: InstructionType = Field(default=InstructionType.CUSTOM, description="Instruction type")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
