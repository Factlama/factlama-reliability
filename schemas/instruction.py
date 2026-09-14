"""Instruction schema models."""

from enum import Enum

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
    """Priority levels for instructions (contracts/v0.1's uppercase wire values)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Instruction(BaseModel):
    """Instruction for the AI system to follow."""

    model_config = ConfigDict(frozen=True)

    instruction_id: str = Field(..., description="Unique identifier for this instruction")
    text: str = Field(..., description="The instruction text")
    priority: Priority = Field(default=Priority.MEDIUM, description="Priority level")
    type: InstructionType = Field(default=InstructionType.CUSTOM, description="Instruction type")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
