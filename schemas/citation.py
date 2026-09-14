"""Citation schema models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Locator(BaseModel):
    """Location information within a source document."""

    model_config = ConfigDict(frozen=True)

    page: int | None = Field(None, description="Page number")
    paragraph: int | None = Field(None, description="Paragraph number")
    character_range: tuple[int, int] | None = Field(
        None, description="Character range (start, end)"
    )
    line_range: tuple[int, int] | None = Field(None, description="Line range (start, end)")
    section: str | None = Field(None, description="Section name/number")
    timestamp: str | None = Field(None, description="Timestamp for audio/video")
    json_path: str | None = Field(None, description="JSONPath for structured data")


class Citation(BaseModel):
    """Citation linking a claim to evidence."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(..., description="Unique identifier for this citation")
    claim_id: str | None = Field(None, description="ID of the claim this citation supports")
    evidence_id: str = Field(..., description="ID of the evidence/source being cited")
    locator: Locator | None = Field(None, description="Location within the source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
