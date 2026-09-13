"""Telemetry schema models for observability."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TelemetryData(BaseModel):
    """Telemetry data for observability export (OpenTelemetry compatible)."""

    model_config = ConfigDict(frozen=True)

    # FactLama metadata
    factlama_version: str = Field(default="0.1", description="FactLama version")
    trace_id: str = Field(..., description="Trace ID for correlation")
    span_id: str | None = Field(None, description="Span ID within trace")

    # Service metadata
    service_name: str | None = Field(None, description="Service name")
    environment: str | None = Field(None, description="Environment (production, staging, etc.)")

    # LLM metadata
    llm_provider: str | None = Field(None, description="LLM provider")
    llm_model: str | None = Field(None, description="LLM model name")
    llm_input_tokens: int | None = Field(None, ge=0, description="Input token count")
    llm_output_tokens: int | None = Field(None, ge=0, description="Output token count")
    llm_latency_ms: float | None = Field(None, ge=0, description="LLM latency in milliseconds")

    # Verification results
    verdict: str = Field(..., description="Overall verification verdict")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Reliability score")

    # Core metrics
    groundedness: float | None = Field(None, ge=0.0, le=1.0)
    hallucination_risk: float | None = Field(None, ge=0.0, le=1.0)
    contradiction_risk: float | None = Field(None, ge=0.0, le=1.0)
    scope_breach: float | None = Field(None, ge=0.0, le=1.0)
    citation_support: float | None = Field(None, ge=0.0, le=1.0)
    instruction_adherence: float | None = Field(None, ge=0.0, le=1.0)
    tool_correctness: float | None = Field(None, ge=0.0, le=1.0)
    confidence_alignment: float | None = Field(None, ge=0.0, le=1.0)

    # Additional attributes
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional telemetry attributes"
    )

    def to_open_telemetry_attributes(self) -> dict[str, Any]:
        """Convert to OpenTelemetry attribute format."""
        attrs: dict[str, Any] = {
            "factlama.version": self.factlama_version,
            "factlama.verdict": self.verdict,
            "factlama.reliability": self.reliability,
        }

        # Add service metadata
        if self.service_name:
            attrs["service.name"] = self.service_name
        if self.environment:
            attrs["deployment.environment"] = self.environment

        # Add LLM metadata
        if self.llm_provider:
            attrs["llm.provider"] = self.llm_provider
        if self.llm_model:
            attrs["llm.model"] = self.llm_model
        if self.llm_input_tokens is not None:
            attrs["llm.input_tokens"] = self.llm_input_tokens
        if self.llm_output_tokens is not None:
            attrs["llm.output_tokens"] = self.llm_output_tokens
        if self.llm_latency_ms is not None:
            attrs["llm.latency_ms"] = self.llm_latency_ms

        # Add metrics
        if self.groundedness is not None:
            attrs["factlama.groundedness"] = self.groundedness
        if self.hallucination_risk is not None:
            attrs["factlama.hallucination_risk"] = self.hallucination_risk
        if self.contradiction_risk is not None:
            attrs["factlama.contradiction_risk"] = self.contradiction_risk
        if self.scope_breach is not None:
            attrs["factlama.scope_breach"] = self.scope_breach
        if self.citation_support is not None:
            attrs["factlama.citation_support"] = self.citation_support
        if self.instruction_adherence is not None:
            attrs["factlama.instruction_adherence"] = self.instruction_adherence
        if self.tool_correctness is not None:
            attrs["factlama.tool_correctness"] = self.tool_correctness
        if self.confidence_alignment is not None:
            attrs["factlama.confidence_alignment"] = self.confidence_alignment

        # Merge additional attributes
        attrs.update(self.attributes)

        return attrs
