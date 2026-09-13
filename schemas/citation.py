"""Citation schema models."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Locator(BaseModel):
    """Location information within a source document."""

    model_config = ConfigDict(frozen=True)

    page: Optional[int] = Field(None, description="Page number")
    paragraph: Optional[int] = Field(None, description="Paragraph number")
    character_range: Optional[tuple[int, int]] = Field(None, description="Character range (start, end)")
    line_range: Optional[tuple[int, int]] = Field(None, description="Line range (start, end)")
    section: Optional[str] = Field(None, description="Section name/number")
    timestamp: Optional[str] = Field(None, description="Timestamp for audio/video")
    json_path: Optional[str] = Field(None, description="JSONPath for structured data")


class Citation(BaseModel):
    """Citation linking a claim to evidence."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique identifier for this citation")
    claim_id: Optional[str] = Field(None, description="ID of the claim this citation supports")
    source_id: str = Field(..., description="ID of the evidence/source being cited")
    locator: Optional[Locator] = Field(None, description="Location within the source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
