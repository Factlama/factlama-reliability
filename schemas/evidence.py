"""Evidence schema models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceType(str, Enum):
    """Types of evidence sources."""

    DOCUMENT = "document"
    RETRIEVAL = "retrieval"
    DATABASE = "database"
    TOOL_RESULT = "tool_result"
    USER_INPUT = "user_input"
    SYSTEM_CONTEXT = "system_context"
    CONVERSATION = "conversation"
    GENERATED_CONTEXT = "generated_context"


class Locator(BaseModel):
    """Precise location of an evidence span within its source for traceability."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., description="ID of the source (e.g., document or chunk id)")
    page: int | None = Field(None, description="Page number within the source")
    section: str | None = Field(None, description="Section or heading within the source")
    start_char: int | None = Field(
        None, description="Start character offset within the source text"
    )
    end_char: int | None = Field(None, description="End character offset within the source text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional locator metadata"
    )


class Source(BaseModel):
    """Source information for evidence."""

    model_config = ConfigDict(frozen=True)

    uri: str | None = Field(None, description="URI of the source (URL, file path, etc.)")
    title: str | None = Field(None, description="Title of the source")
    author: str | None = Field(None, description="Author of the source")
    published_at: str | None = Field(None, description="Publication date/time")
    version: str | None = Field(None, description="Version of the source (e.g., doc version)")
    updated_at: str | None = Field(None, description="Last updated date/time of the source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")


class Evidence(BaseModel):
    """Evidence is a first-class object representing available context for verification.

    Exactly one of `content` (inline text) or `reference` (a pointer without
    inline content, e.g. after content-governance redaction) is set --
    contracts/v0.1's `oneOf`.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Unique identifier for this evidence")
    type: EvidenceType = Field(default=EvidenceType.DOCUMENT, description="Type of evidence")
    content: str | None = Field(None, description="The text content extracted from the source")
    reference: dict[str, Any] | None = Field(
        None, description="Pointer to the content when it is not inlined here"
    )
    source_uri: str | None = Field(None, description="URI of the source (URL, file path, etc.)")
    title: str | None = Field(None, description="Title of the source")
    retrieval_rank: int | None = Field(None, ge=0, description="Rank in the retrieval result set")
    retrieval_score: float | None = Field(None, description="Retriever-assigned relevance score")
    timestamp: str | None = Field(None, description="RFC 3339 timestamp for this evidence")
    trust_label: str | None = Field(None, description="Caller-assigned trust label")
    source: Source | None = Field(None, description="Extended source information")
    locator: Locator | None = Field(None, description="Precise location of this evidence span")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="after")
    def validate_content_xor_reference(self) -> "Evidence":
        if (self.content is None) == (self.reference is None):
            raise ValueError("Evidence must set exactly one of content or reference")
        return self

    def __hash__(self) -> int:
        return hash(self.evidence_id)
