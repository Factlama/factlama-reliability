"""REL-04: claim extraction -- deterministic IDs, extractor identity, and
source offsets (claim-engine.md: "Preserve an answer offset when possible
... Use deterministic IDs derived from request ID plus ordinal for extracted
claims ... Record extractor ID/version in provenance").
"""

from core.claims import EnhancedClaimExtractor, SimpleClaimExtractor


class TestSimpleClaimExtractorOffsets:
    def test_populates_start_and_end_char(self) -> None:
        answer = "Paris is the capital of France. The Eiffel Tower is in Paris."
        extractor = SimpleClaimExtractor()
        claims = extractor.extract(answer)

        assert len(claims) == 2
        for claim in claims:
            assert claim.start_char is not None
            assert claim.end_char is not None
            assert answer[claim.start_char : claim.end_char] == claim.text

    def test_offsets_advance_monotonically_for_repeated_sentences(self) -> None:
        """A sentence repeated verbatim must resolve to its own, later
        occurrence, not the same span twice."""
        sentence = "The sky is blue during the day."
        answer = f"{sentence} {sentence}"
        extractor = SimpleClaimExtractor()
        claims = extractor.extract(answer)

        assert len(claims) == 2
        first, second = claims
        assert first.end_char <= second.start_char
        assert answer[first.start_char : first.end_char] == sentence
        assert answer[second.start_char : second.end_char] == sentence


class TestEnhancedClaimExtractorOffsets:
    def test_clause_offsets_are_literal_substrings(self) -> None:
        answer = "Company X was founded in 2018 and it is based in Paris."
        extractor = EnhancedClaimExtractor()
        claims = extractor.extract(answer)

        assert len(claims) >= 2
        for claim in claims:
            if claim.start_char is None:
                continue
            assert answer[claim.start_char : claim.end_char] == claim.text

    def test_repeated_clause_across_sentences_gets_distinct_offsets(self) -> None:
        clause = "It launched in June"
        answer = f"{clause} and it sold widely. {clause} and it sold widely."
        extractor = EnhancedClaimExtractor()
        claims = extractor.extract(answer)

        located = [c for c in claims if c.text == clause]
        assert len(located) == 2
        assert located[0].start_char < located[1].start_char
        assert answer[located[0].start_char : located[0].end_char] == clause
        assert answer[located[1].start_char : located[1].end_char] == clause


class TestExtractorIdentity:
    """Every extractor declares a stable id/version so a caller can detect a
    segmentation-algorithm change across two extractions of an edited
    answer, not just an ordinal shift."""

    def test_simple_extractor_declares_identity(self) -> None:
        extractor = SimpleClaimExtractor()
        assert extractor.extractor_id == "simple-sentence"
        assert extractor.extractor_version

    def test_enhanced_extractor_declares_identity(self) -> None:
        extractor = EnhancedClaimExtractor()
        assert extractor.extractor_id == "enhanced-clause"
        assert extractor.extractor_version

    def test_identity_is_distinct_between_extractors(self) -> None:
        assert SimpleClaimExtractor().extractor_id != EnhancedClaimExtractor().extractor_id


class TestClaimIdDeterminism:
    """claim-engine.md: IDs are deterministic ordinals for one answer against
    one extractor version -- re-running extraction on the same answer must
    reproduce the same IDs, and an edit that only changes text *after* a
    claim must not change that claim's own ID."""

    def test_same_answer_same_extractor_yields_same_ids(self) -> None:
        answer = "Paris is the capital of France. The Eiffel Tower is in Paris."
        extractor = EnhancedClaimExtractor()
        first_ids = [c.claim_id for c in extractor.extract(answer)]
        second_ids = [c.claim_id for c in extractor.extract(answer)]
        assert first_ids == second_ids

    def test_editing_a_later_sentence_preserves_earlier_claim_ids(self) -> None:
        extractor = SimpleClaimExtractor()
        original = "Paris is the capital of France. Berlin is the capital of Germany."
        edited = "Paris is the capital of France. Madrid is the capital of Spain."

        original_first = extractor.extract(original)[0]
        edited_first = extractor.extract(edited)[0]

        assert original_first.claim_id == edited_first.claim_id
        assert original_first.text == edited_first.text
