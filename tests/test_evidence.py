"""REL-05: evidence engine. `SimpleEvidenceMapper`'s golden cases and
`EvidenceRetriever`'s port contract (evidence-engine.md: "Retrieval is an
optional EvidenceRetriever port ... No vector database is an MVP
dependency" -- there is deliberately no concrete implementation yet).
"""

import pytest

from core.evidence import EvidenceRetriever, SimpleEvidenceMapper
from schemas.claims import Claim, ClaimVerdict, RationaleCode
from schemas.evidence import Evidence
from schemas.tenancy import TenantContext


class TestSimpleEvidenceMapperGoldenCases:
    def test_no_evidence_is_insufficient(self) -> None:
        mapper = SimpleEvidenceMapper()
        claim = Claim(claim_id="c1", text="Paris is the capital of France.")
        result = mapper.map_evidence(claim, [])

        assert result.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
        assert result.rationale_code == RationaleCode.NO_EVIDENCE_SUPPLIED
        assert result.evidence_ids == []

    def test_directly_supporting_evidence_is_supported(self) -> None:
        mapper = SimpleEvidenceMapper()
        claim = Claim(claim_id="c1", text="Paris is the capital of France.")
        evidence = [Evidence(evidence_id="e1", content="Paris is the capital of France.")]
        result = mapper.map_evidence(claim, evidence)

        assert result.verdict == ClaimVerdict.SUPPORTED
        assert "e1" in result.evidence_ids

    def test_numerically_contradicting_evidence_is_contradicted(self) -> None:
        mapper = SimpleEvidenceMapper()
        claim = Claim(claim_id="c1", text="Product X weighs 3.4 kg.")
        evidence = [Evidence(evidence_id="e1", content="Product X weighs 2.4 kg.")]
        result = mapper.map_evidence(claim, evidence)

        assert result.verdict == ClaimVerdict.CONTRADICTED
        assert result.rationale_code == RationaleCode.NUMERICAL_CONTRADICTION

    def test_irrelevant_evidence_is_unsupported(self) -> None:
        mapper = SimpleEvidenceMapper()
        claim = Claim(claim_id="c1", text="Paris is the capital of France.")
        evidence = [Evidence(evidence_id="e1", content="Bananas are a good source of potassium.")]
        result = mapper.map_evidence(claim, evidence)

        assert result.verdict == ClaimVerdict.UNSUPPORTED
        assert result.rationale_code == RationaleCode.NO_SUPPORT

    def test_reference_only_evidence_is_not_treated_as_content(self) -> None:
        """A reference-only Evidence (no inline content, e.g. after content-
        governance redaction) must not crash the mapper, and -- since it has
        no resolvable content here -- must not be silently trusted as
        support (evidence-engine.md: "an unresolved reference is an
        unavailable evidence condition")."""
        mapper = SimpleEvidenceMapper()
        claim = Claim(claim_id="c1", text="Paris is the capital of France.")
        evidence = [Evidence(evidence_id="e1", reference={"uri": "doc://redacted"})]
        result = mapper.map_evidence(claim, evidence)

        assert result.verdict != ClaimVerdict.SUPPORTED


class TestEvidenceRetrieverPort:
    def test_cannot_instantiate_without_implementing_retrieve(self) -> None:
        with pytest.raises(TypeError):
            EvidenceRetriever()  # type: ignore[abstract]

    def test_minimal_subclass_satisfies_the_port(self) -> None:
        class _StaticRetriever(EvidenceRetriever):
            def retrieve(
                self, tenant_context: TenantContext, claim: Claim, max_results: int
            ) -> list[Evidence]:
                return [Evidence(evidence_id="retrieved_1", content="Some retrieved text.")]

        retriever = _StaticRetriever()
        results = retriever.retrieve(
            TenantContext(tenant_id="t-acme"), Claim(claim_id="c1", text="Anything."), max_results=5
        )

        assert len(results) == 1
        assert results[0].evidence_id == "retrieved_1"
