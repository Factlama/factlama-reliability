"""Claim extraction module."""

import re
from abc import ABC, abstractmethod

from schemas.claims import Claim, ClaimType


class ClaimExtractor(ABC):
    """Abstract base class for claim extraction."""

    @abstractmethod
    def extract(self, answer: str) -> list[Claim]:
        """Extract claims from an answer.

        Args:
            answer: The AI-generated answer to extract claims from.

        Returns:
            List of extracted claims.
        """
        ...


class SimpleClaimExtractor(ClaimExtractor):
    """Simple rule-based claim extractor using sentence splitting.

    This is a baseline implementation. Production implementations should use
    NLP libraries or the FactLama SLM for more accurate claim extraction.
    """

    def __init__(self, min_claim_length: int = 10) -> None:
        """Initialize the claim extractor.

        Args:
            min_claim_length: Minimum character length for a claim.
        """
        self.min_claim_length = min_claim_length
        # Patterns that typically indicate non-factual content
        self.non_factual_patterns = [
            r"^(I think|I believe|In my opinion|It seems|Perhaps|Maybe)",
            r"\?$",  # Questions
            r"^(Thank you|Thanks|Hello|Hi|Hey)",
        ]

    def extract(self, answer: str) -> list[Claim]:
        """Extract claims from an answer using sentence splitting.

        Args:
            answer: The AI-generated answer to extract claims from.

        Returns:
            List of extracted claims.
        """
        claims: list[Claim] = []

        # Split on sentence boundaries
        sentences = self._split_sentences(answer)

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < self.min_claim_length:
                continue

            # Skip non-factual content
            if self._is_non_factual(sentence):
                continue

            # Determine claim type
            claim_type = self._classify_claim_type(sentence)

            # Calculate importance (simple heuristic based on position and content)
            importance = self._calculate_importance(sentence, idx, len(sentences))

            claim = Claim(
                id=f"claim_{idx + 1:03d}",
                text=sentence,
                type=claim_type,
                importance=importance,
            )
            claims.append(claim)

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting on common boundaries
        # Production code should use proper NLP libraries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def _is_non_factual(self, sentence: str) -> bool:
        """Check if a sentence is likely non-factual.

        Args:
            sentence: Sentence to check.

        Returns:
            True if the sentence is likely non-factual.
        """
        for pattern in self.non_factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    def _classify_claim_type(self, sentence: str) -> ClaimType:
        """Classify the type of a claim.

        Args:
            sentence: The claim sentence.

        Returns:
            The classified claim type.
        """
        # Check for numerical claims
        if re.search(
            r"\d+(?:\.\d+)?(?:\s*(?:%|percent|dollars|\$|kg|lbs|miles|km|years?|months?|days?))",
            sentence,
            re.IGNORECASE,
        ):
            # Check if it's temporal
            if re.search(
                r"\b(in|on|at|during)\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", sentence
            ):
                return ClaimType.TEMPORAL
            return ClaimType.NUMERICAL

        # Check for temporal claims
        if re.search(
            r"\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            sentence,
            re.IGNORECASE,
        ):
            return ClaimType.TEMPORAL

        # Check for comparative claims
        if re.search(
            r"\b(more|less|better|worse|larger|smaller|higher|lower|faster|slower)\s+than\b",
            sentence,
            re.IGNORECASE,
        ):
            return ClaimType.COMPARATIVE

        # Check for causal claims
        if re.search(
            r"\b(because|since|therefore|thus|consequently|as a result|caused|led to)\b",
            sentence,
            re.IGNORECASE,
        ):
            return ClaimType.CAUSAL

        # Default to factual
        return ClaimType.FACTUAL

    def _calculate_importance(self, sentence: str, position: int, total: int) -> float:
        """Calculate the importance of a claim.

        Args:
            sentence: The claim sentence.
            position: Position of the claim in the answer.
            total: Total number of claims.

        Returns:
            Importance score between 0.0 and 1.0.
        """
        # Base importance on position (earlier claims tend to be more important)
        if total == 0:
            return 1.0

        position_weight = 1.0 - (position / total) * 0.3  # Max 0.3 reduction

        # Boost importance for sentences with numbers (often key facts)
        has_numbers = bool(re.search(r"\d+", sentence))
        if has_numbers:
            position_weight = min(1.0, position_weight + 0.1)

        # Boost for named entities (capitalized words that aren't sentence starts)
        has_entities = bool(re.search(r"(?<!^)(?<!\. )\b[A-Z][a-z]+\b", sentence))
        if has_entities:
            position_weight = min(1.0, position_weight + 0.1)

        return round(position_weight, 2)


class EnhancedClaimExtractor(ClaimExtractor):
    """Enhanced claim extractor with clause-level splitting.

    This extractor goes beyond sentence-level splitting to identify atomic
    claims within complex sentences. It handles:
    - Compound claims joined by conjunctions (and, but, however)
    - Relative clauses and appositives
    - Multiple independent claims within a single sentence
    """

    def __init__(self, min_claim_length: int = 10) -> None:
        """Initialize the enhanced claim extractor.

        Args:
            min_claim_length: Minimum character length for a claim.
        """
        self.min_claim_length = min_claim_length
        self.non_factual_patterns = [
            r"^(I think|I believe|In my opinion|It seems|Perhaps|Maybe)",
            r"\?$",
            r"^(Thank you|Thanks|Hello|Hi|Hey)",
        ]

        # Conjunctions that often separate independent claims
        self.claim_conjunctions = [
            r"\band\b",
            r"\bbut\b",
            r"\bhowever\b",
            r"\byet\b",
            r"\bmoreover\b",
            r"\bfurthermore\b",
            r"\balso\b",
            r"\bin addition\b",
            r"\bas well as\b",
        ]

        # Relative clause markers that often embed secondary claims
        self.relative_markers = [
            r"\bwhich\b",
            r"\bthat\b",
            r"\bwho\b",
            r"\bwhom\b",
            r"\bwhere\b",
            r"\bwhen\b",
        ]

        # Appositive markers
        self.appositive_markers = [
            r",\s+(?:a|an|the)\s+\w+",  # ", a ..."
            r"\(\s*(?:a|an|the)\s+\w+",  # "( a ..."
        ]

    def extract(self, answer: str) -> list[Claim]:
        """Extract claims from an answer using clause-level splitting.

        Args:
            answer: The AI-generated answer to extract claims from.

        Returns:
            List of extracted claims.
        """
        claims: list[Claim] = []

        # First, split into sentences
        sentences = self._split_sentences(answer)

        claim_idx = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < self.min_claim_length:
                continue

            # Skip non-factual content
            if self._is_non_factual(sentence):
                continue

            # Split sentence into clauses
            clauses = self._split_into_clauses(sentence)

            for clause in clauses:
                clause = clause.strip()
                if not clause or len(clause) < self.min_claim_length:
                    continue

                if self._is_non_factual(clause):
                    continue

                claim_type = self._classify_claim_type(clause)
                importance = self._calculate_importance(
                    clause, claim_idx, len(clauses) + len(sentences)
                )

                claim = Claim(
                    id=f"claim_{claim_idx + 1:03d}",
                    text=clause,
                    type=claim_type,
                    importance=importance,
                )
                claims.append(claim)
                claim_idx += 1

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def _split_into_clauses(self, sentence: str) -> list[str]:
        """Split a sentence into atomic clauses/claims.

        Args:
            sentence: A single sentence that may contain multiple claims.

        Returns:
            List of clause strings.
        """
        clauses = [sentence]

        # Split on claim-separating conjunctions
        for conj_pattern in self.claim_conjunctions:
            new_clauses = []
            for clause in clauses:
                # Find the conjunction position
                matches = list(re.finditer(conj_pattern, clause, re.IGNORECASE))
                if not matches:
                    new_clauses.append(clause)
                    continue

                # Split at each conjunction
                last_end = 0
                for match in matches:
                    before = clause[last_end : match.start()].strip()
                    if before:
                        new_clauses.append(before)
                    last_end = match.end()

                # Add remainder after last conjunction
                remainder = clause[last_end:].strip()
                if remainder:
                    new_clauses.append(remainder)

            clauses = new_clauses

        # Handle relative clauses (which, that, who, etc.) - extract embedded claims
        refined_clauses = []
        for clause in clauses:
            # Find relative clauses that could be separate claims
            for marker in self.relative_markers:
                pattern = rf",\s*{marker}\s+"
                matches = list(re.finditer(pattern, clause, re.IGNORECASE))
                if matches:
                    # Split the clause at the relative marker
                    last_end = 0
                    for match in matches:
                        before = clause[last_end : match.start()].strip()
                        if before:
                            refined_clauses.append(before)
                        last_end = match.end()
                    remainder = clause[last_end:].strip()
                    if remainder:
                        refined_clauses.append(remainder)
                    break
            else:
                refined_clauses.append(clause)

        # Clean up and return
        return [c for c in refined_clauses if c and len(c) >= self.min_claim_length]

    def _is_non_factual(self, sentence: str) -> bool:
        """Check if a sentence/clause is likely non-factual."""
        for pattern in self.non_factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    def _classify_claim_type(self, sentence: str) -> ClaimType:
        """Classify the type of a claim."""
        # Check for numerical claims
        if re.search(
            r"\d+(?:\.\d+)?(?:\s*(?:%|percent|dollars|\$|kg|lbs|miles|km|years?|months?|days?))",
            sentence,
            re.IGNORECASE,
        ):
            if re.search(
                r"\b(in|on|at|during)\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", sentence
            ):
                return ClaimType.TEMPORAL
            return ClaimType.NUMERICAL

        if re.search(
            r"\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            sentence,
            re.IGNORECASE,
        ):
            return ClaimType.TEMPORAL

        if re.search(
            r"\b(more|less|better|worse|larger|smaller|higher|lower|faster|slower)\s+than\b",
            sentence,
            re.IGNORECASE,
        ):
            return ClaimType.COMPARATIVE

        if re.search(
            r"\b(because|since|therefore|thus|consequently|as a result|caused|led to)\b",
            sentence,
            re.IGNORECASE,
        ):
            return ClaimType.CAUSAL

        return ClaimType.FACTUAL

    def _calculate_importance(self, sentence: str, position: int, total: int) -> float:
        """Calculate the importance of a claim."""
        if total == 0:
            return 1.0

        position_weight = 1.0 - (position / total) * 0.3

        has_numbers = bool(re.search(r"\d+", sentence))
        if has_numbers:
            position_weight = min(1.0, position_weight + 0.1)

        has_entities = bool(re.search(r"(?<!^)(?<!\. )\b[A-Z][a-z]+\b", sentence))
        if has_entities:
            position_weight = min(1.0, position_weight + 0.1)

        return round(position_weight, 2)
