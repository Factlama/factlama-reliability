"""Judge provider port and reference implementations.

`JudgeProvider.evaluate()` is the bounded port from CONTRACTS.md: it takes
exactly one claim and bounded evidence, a deadline and a cancellation token,
and returns one normalized verdict or a typed error. Claim segmentation
(`extract_claims`) is deliberately not part of this interface -- that is
`core.claims`'s job, not a judge's.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.claims import Claim, ClaimVerdict, EvidenceReference
from schemas.evidence import Evidence
from schemas.instruction import Instruction
from schemas.policy import Policy


class JudgeErrorCode(str, Enum):
    """Typed provider/dispatch error taxonomy (CONTRACTS.md)."""

    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CANCELLED = "CANCELLED"
    CONFIGURATION = "CONFIGURATION"


class CancellationToken:
    """A minimal cooperative cancellation flag for a single evaluation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class JudgeError(BaseModel):
    """A typed failure from a judge provider. Never a fabricated factual verdict."""

    model_config = ConfigDict(frozen=True)

    code: JudgeErrorCode
    message: str


class JudgeRequest(BaseModel):
    """One claim plus bounded evidence, submitted to a JudgeProvider."""

    model_config = ConfigDict(frozen=True)

    claim: Claim
    evidence: list[Evidence] = Field(default_factory=list)
    configuration_version: str = "0.1"


class JudgeResult(BaseModel):
    """A normalized judge outcome: either a verdict, or a typed error, never both."""

    model_config = ConfigDict(frozen=True)

    verdict: Optional[ClaimVerdict] = None
    evidence: list[EvidenceReference] = Field(default_factory=list)
    confidence: Optional[float] = None
    reason: Optional[str] = None
    error: Optional[JudgeError] = None


def _bounded_check(deadline: float, cancellation: CancellationToken) -> Optional[JudgeError]:
    """Shared pre-dispatch deadline/cancellation check for every provider."""
    if cancellation.is_cancelled:
        return JudgeError(code=JudgeErrorCode.CANCELLED, message="cancelled before dispatch")
    if time.monotonic() > deadline:
        return JudgeError(code=JudgeErrorCode.TIMEOUT, message="deadline exceeded before dispatch")
    return None


class JudgeProvider(ABC):
    """Abstract base class for judge providers behind the bounded JudgeProvider port.

    Providers may include enterprise/internal models, OpenAI, Anthropic,
    Azure, local open models and eventually the FactLama SLM. Reliability
    core must not branch on provider vendor.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        ...

    @abstractmethod
    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Evaluate exactly one claim against bounded evidence.

        Args:
            request: The claim and evidence to evaluate.
            deadline: A `time.monotonic()`-based absolute deadline.
            cancellation: Cooperative cancellation for this evaluation.

        Returns:
            JudgeResult with a normalized verdict, or a typed error.
        """
        ...

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, Optional[str]]:
        """Evaluate how well the answer follows an instruction.

        Not part of the formal JudgeProvider port (CONTRACTS.md's port is
        claim-evaluation only) -- an optional extra capability some
        providers offer. Defaults to assumed compliance.
        """
        return 1.0, None

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Evaluate scope compliance.

        Not part of the formal JudgeProvider port -- an optional extra
        capability. Defaults to no breach.
        """
        return 0.0, []


class MockModelProvider(JudgeProvider):
    """Mock provider for testing that returns predictable results.

    This provider is useful for testing the core verification pipeline
    without requiring an actual model. It uses deterministic logic to
    produce consistent results for the same inputs.
    """

    def __init__(
        self,
        default_verdict: str = "SUPPORTED",
        default_confidence: float = 0.95,
    ) -> None:
        """Initialize the mock provider.

        Args:
            default_verdict: Default claim verdict to return.
            default_confidence: Default confidence score.
        """
        self.default_verdict = default_verdict
        self.default_confidence = default_confidence

    @property
    def name(self) -> str:
        return "mock"

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using simple text matching."""
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        claim, evidence = request.claim, request.evidence

        if not evidence:
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                evidence=[],
                reason="No evidence provided",
            )

        # Check if claim text appears in any evidence
        claim_lower = claim.text.lower()
        evidence_refs: list[EvidenceReference] = []
        found_support = False

        for ev in evidence:
            if claim_lower in ev.extracted_text.lower() or ev.extracted_text.lower() in claim_lower:
                found_support = True
                evidence_refs.append(
                    EvidenceReference(
                        evidence_id=ev.id,
                        support=0.9,
                        relevance=0.95,
                    )
                )

        if found_support:
            return JudgeResult(
                verdict=ClaimVerdict.SUPPORTED,
                confidence=self.default_confidence,
                evidence=evidence_refs,
                reason="Claim found in evidence",
            )
        else:
            return JudgeResult(
                verdict=ClaimVerdict.UNSUPPORTED,
                confidence=0.7,
                evidence=[],
                reason="Claim not found in evidence",
            )

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, Optional[str]]:
        """Return perfect adherence for mock."""
        return 1.0, "Mock evaluation - assumed compliant"

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Return no scope breach for mock."""
        return 0.0, []


class RuleBasedProvider(JudgeProvider):
    """Rule-based verification provider using heuristics and text matching.

    This provider implements verification using deterministic rules without
    any machine learning. It's suitable for:
    - Testing and validation
    - Simple verification scenarios
    - Baseline comparisons
    - Environments where ML models cannot be deployed
    """

    def __init__(
        self,
        support_threshold: float = 0.7,
        contradiction_threshold: float = 0.8,
    ) -> None:
        """Initialize the rule-based provider.

        Args:
            support_threshold: Threshold for considering evidence as supporting.
            contradiction_threshold: Threshold for detecting contradictions.
        """
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

    @property
    def name(self) -> str:
        return "rule-based"

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using rule-based text analysis."""
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        from core.evidence import SimpleEvidenceMapper

        mapper = SimpleEvidenceMapper(
            match_threshold=self.support_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )
        verification = mapper.map_evidence(request.claim, request.evidence)
        return JudgeResult(
            verdict=verification.verdict,
            evidence=verification.evidence,
            confidence=verification.confidence,
            reason=verification.reason,
        )

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, Optional[str]]:
        """Evaluate instruction adherence using simple rules."""
        instruction_text = instruction.text.lower()

        # Check for format instructions
        if "json" in instruction_text and "return json" in instruction_text:
            # Simple check for JSON-like structure
            if answer.strip().startswith("{") and answer.strip().endswith("}"):
                return 1.0, "Answer appears to be JSON format"
            else:
                return 0.3, "Answer does not appear to be JSON format"

        # Check for grounding instructions
        if "only" in instruction_text and "context" in instruction_text:
            # This would need evidence to verify properly
            return 0.8, "Grounding instruction - cannot fully verify without evidence"

        # Default: assume compliance
        return 0.9, "Rule-based evaluation - assumed compliant"

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Evaluate scope compliance using keyword matching."""
        import re

        if not policy.scope.enabled:
            return 0.0, []

        breach_domains: list[str] = []
        answer_lower = answer.lower()

        # Check for forbidden domains using word boundary matching
        for forbidden in policy.scope.forbidden:
            forbidden_lower = forbidden.lower()
            # Check for the forbidden term as a word or phrase
            if re.search(rf"\b{re.escape(forbidden_lower)}\b", answer_lower):
                breach_domains.append(forbidden)

        # If forbidden domains found, calculate breach score
        if breach_domains:
            breach_score = min(1.0, len(breach_domains) * 0.3)
            return breach_score, breach_domains

        return 0.0, []


class EmbeddingProvider(JudgeProvider):
    """Embedding-based verification provider using sentence transformers.

    This provider uses sentence embeddings to compute semantic similarity
    between claims and evidence, enabling verification beyond exact text matching.
    It's suitable for:
    - Semantic verification where wording differs but meaning is similar
    - Paraphrase detection
    - Cross-lingual verification (with multilingual models)
    - Production use when ML models can be deployed
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        support_threshold: float = 0.7,
        contradiction_threshold: float = 0.3,
        device: str | None = None,
    ) -> None:
        """Initialize the embedding provider.

        Args:
            model_name: Name of the sentence-transformers model to use.
            support_threshold: Similarity threshold for supporting evidence.
            contradiction_threshold: Similarity threshold for detecting contradictions.
            device: Device to run the model on (e.g., "cpu", "cuda", "mps").
        """
        self.model_name = model_name
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.device = device
        self._model = None

    @property
    def name(self) -> str:
        return f"embedding:{self.model_name}"

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for EmbeddingProvider. "
                    "Install with: pip install factlama-reliability[embeddings]"
                )
        return self._model

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using semantic embedding similarity."""
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        claim, evidence = request.claim, request.evidence

        if not evidence:
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                evidence=[],
                reason="No evidence provided",
            )

        try:
            model = self._get_model()
        except RuntimeError as e:
            return JudgeResult(error=JudgeError(code=JudgeErrorCode.CONFIGURATION, message=str(e)))

        # Encode the claim
        claim_embedding = model.encode(claim.text, convert_to_tensor=True)

        evidence_refs: list[EvidenceReference] = []
        best_similarity = 0.0
        best_evidence = None

        for ev in evidence:
            # Encode evidence text
            evidence_embedding = model.encode(ev.extracted_text, convert_to_tensor=True)

            # Compute cosine similarity
            import torch.nn.functional as F
            similarity = F.cosine_similarity(
                claim_embedding.unsqueeze(0),
                evidence_embedding.unsqueeze(0)
            ).item()

            # Also check for numerical contradiction
            has_numerical_contradiction = self._check_numerical_contradiction(
                claim.text, ev.extracted_text
            )

            if has_numerical_contradiction and similarity > self.support_threshold:
                # High semantic similarity but numerical contradiction
                evidence_refs.append(
                    EvidenceReference(
                        evidence_id=ev.id,
                        support=-similarity,  # Negative support = contradiction
                        relevance=similarity,
                    )
                )
            elif similarity >= self.support_threshold:
                evidence_refs.append(
                    EvidenceReference(
                        evidence_id=ev.id,
                        support=similarity,
                        relevance=similarity,
                    )
                )

            if similarity > best_similarity:
                best_similarity = similarity
                best_evidence = ev

        # Determine verdict
        if best_evidence is not None and self._check_numerical_contradiction(claim.text, best_evidence.extracted_text) and best_similarity > self.support_threshold:
            verdict = ClaimVerdict.CONTRADICTED
            reason = "Semantic similarity but numerical contradiction detected"
        elif evidence_refs:
            best_support = max(ref.support for ref in evidence_refs)
            if best_support >= self.support_threshold:
                verdict = ClaimVerdict.SUPPORTED
                reason = f"Claim semantically supported by evidence (similarity: {best_support:.2f})"
            elif best_support <= -self.contradiction_threshold:
                verdict = ClaimVerdict.CONTRADICTED
                reason = f"Claim contradicted by evidence (similarity: {abs(best_support):.2f})"
            else:
                verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
                reason = f"Evidence exists but similarity below threshold (best: {best_support:.2f})"
        else:
            verdict = ClaimVerdict.UNSUPPORTED
            reason = "No semantically relevant evidence found"

        # Calculate confidence based on similarity
        confidence = self._calculate_confidence(evidence_refs, verdict)

        return JudgeResult(
            verdict=verdict,
            confidence=confidence,
            evidence=evidence_refs,
            reason=reason,
        )

    def _check_numerical_contradiction(self, claim: str, evidence_text: str) -> bool:
        """Check for numerical contradictions between claim and evidence."""
        import re

        claim_numbers = re.findall(r"\d+(?:\.\d+)?", claim)
        evidence_numbers = re.findall(r"\d+(?:\.\d+)?", evidence_text)

        if not claim_numbers or not evidence_numbers:
            return False

        # Check if there's high word overlap but different numbers
        claim_words = set(claim.lower().split())
        ev_words = set(evidence_text.lower().split())
        overlap = claim_words & ev_words
        similarity = len(overlap) / len(claim_words) if claim_words else 0.0

        # If texts are similar (> 60% word overlap) but numbers differ
        if similarity > 0.6 and claim_numbers != evidence_numbers:
            return True

        return False

    def _calculate_confidence(
        self,
        evidence_refs: list[EvidenceReference],
        verdict: ClaimVerdict,
    ) -> float:
        """Calculate confidence in the verdict based on embedding similarity."""
        if verdict == ClaimVerdict.UNSUPPORTED:
            return 0.5

        if not evidence_refs:
            return 0.5

        best_relevance = max(ref.relevance for ref in evidence_refs)
        best_support = max(abs(ref.support) for ref in evidence_refs)

        confidence = (best_relevance + best_support) / 2
        return min(1.0, max(0.0, confidence))

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, Optional[str]]:
        """Evaluate instruction adherence using semantic similarity."""
        model = self._get_model()

        answer_embedding = model.encode(answer, convert_to_tensor=True)
        instruction_embedding = model.encode(instruction.text, convert_to_tensor=True)

        import torch.nn.functional as F
        similarity = F.cosine_similarity(
            answer_embedding.unsqueeze(0),
            instruction_embedding.unsqueeze(0)
        ).item()

        # For instruction adherence, we check if the answer follows the instruction
        # This is a simple heuristic - in practice, more sophisticated approaches needed
        if "json" in instruction.text.lower():
            if answer.strip().startswith("{") and answer.strip().endswith("}"):
                return 1.0, "Answer is JSON format as instructed"

        return similarity, f"Semantic similarity with instruction: {similarity:.2f}"

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Evaluate scope compliance using semantic similarity."""
        if not policy.scope.enabled:
            return 0.0, []

        breach_domains: list[str] = []
        model = self._get_model()

        answer_embedding = model.encode(answer, convert_to_tensor=True)

        # Check forbidden domains using semantic similarity
        for forbidden in policy.scope.forbidden:
            forbidden_embedding = model.encode(forbidden, convert_to_tensor=True)

            import torch.nn.functional as F
            similarity = F.cosine_similarity(
                answer_embedding.unsqueeze(0),
                forbidden_embedding.unsqueeze(0)
            ).item()

            if similarity > 0.7:  # Threshold for scope breach
                breach_domains.append(forbidden)

        if breach_domains:
            breach_score = min(1.0, len(breach_domains) * 0.3)
            return breach_score, breach_domains

        return 0.0, []


class NLIProvider(JudgeProvider):
    """Natural Language Inference (NLI) based verification provider.

    This provider uses NLI models (e.g., DeBERTa, BART) to determine
    entailment/contradiction/neutral relationships between claims and evidence.
    It provides the most accurate verification but requires more compute.
    """

    def __init__(
        self,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        device: str | None = None,
    ) -> None:
        """Initialize the NLI provider.

        Args:
            model_name: Name of a sequence-classification model fine-tuned for
                3-way NLI (entailment/neutral/contradiction), e.g. an MNLI
                checkpoint. Note: a plain base checkpoint (no NLI fine-tuning,
                e.g. "microsoft/deberta-v3-base") will NOT work correctly --
                `AutoModelForSequenceClassification` would attach a randomly
                initialized classification head to it.
            device: Device to run the model on.
        """
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._label_indices: Optional[dict[str, int]] = None

    @property
    def name(self) -> str:
        return f"nli:{self.model_name}"

    def _get_model(self):
        """Lazy-load the NLI model, tokenizer, and its entailment/neutral/contradiction label mapping."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                if self.device:
                    self._model = self._model.to(self.device)
                else:
                    self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                raise RuntimeError(
                    "transformers and torch are required for NLIProvider. "
                    "Install with: pip install factlama-reliability[nli]"
                )

            # Different NLI checkpoints order their 3 output classes
            # differently (e.g. entailment=0 for some, =1 for others), so the
            # mapping must be read from the model's own config rather than
            # assumed by position.
            self._label_indices = self._resolve_label_indices(self._model.config.id2label)
        return self._tokenizer, self._model

    def _resolve_label_indices(self, id2label: dict) -> dict[str, int]:
        """Map a model's id2label config to entailment/neutral/contradiction indices.

        Args:
            id2label: The model config's index-to-label mapping.

        Returns:
            Dict mapping "entailment"/"neutral"/"contradiction" to output indices.

        Raises:
            RuntimeError: If the model's labels can't be matched to all three
                NLI classes (i.e. it isn't actually an NLI-finetuned model).
        """
        indices: dict[str, int] = {}
        for idx, label in id2label.items():
            normalized = str(label).strip().lower()
            if "entail" in normalized:
                indices["entailment"] = int(idx)
            elif "contra" in normalized:
                indices["contradiction"] = int(idx)
            elif "neutral" in normalized:
                indices["neutral"] = int(idx)

        missing = {"entailment", "neutral", "contradiction"} - indices.keys()
        if missing:
            raise RuntimeError(
                f"NLI model '{self.model_name}' does not expose recognizable "
                f"entailment/neutral/contradiction labels (found: {id2label}); "
                f"missing: {sorted(missing)}. Use a model fine-tuned for 3-way NLI."
            )
        return indices

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using NLI entailment/contradiction classification."""
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        claim, evidence = request.claim, request.evidence

        if not evidence:
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                evidence=[],
                reason="No evidence provided",
            )

        try:
            tokenizer, model = self._get_model()
        except RuntimeError as e:
            return JudgeResult(error=JudgeError(code=JudgeErrorCode.CONFIGURATION, message=str(e)))

        import torch

        evidence_refs: list[EvidenceReference] = []
        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []

        for ev in evidence:
            # Prepare premise (evidence) and hypothesis (claim) for NLI
            premise = ev.extracted_text
            hypothesis = claim.text

            inputs = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).squeeze()

                # Label order is model-specific; use the mapping resolved
                # from this model's own config, not a fixed position.
                entailment_score = probs[self._label_indices["entailment"]].item()
                neutral_score = probs[self._label_indices["neutral"]].item()
                contradiction_score = probs[self._label_indices["contradiction"]].item()

                entailment_scores.append(entailment_score)
                contradiction_scores.append(contradiction_score)
                neutral_scores.append(neutral_score)

                # Determine support based on NLI labels
                if entailment_score > 0.5:
                    evidence_refs.append(
                        EvidenceReference(
                            evidence_id=ev.id,
                            support=entailment_score,
                            relevance=entailment_score,
                        )
                    )
                elif contradiction_score > 0.5:
                    evidence_refs.append(
                        EvidenceReference(
                            evidence_id=ev.id,
                            support=-contradiction_score,
                            relevance=contradiction_score,
                        )
                    )
                else:
                    evidence_refs.append(
                        EvidenceReference(
                            evidence_id=ev.id,
                            support=0.0,
                            relevance=neutral_score,
                        )
                    )

        # Aggregate scores
        max_entailment = max(entailment_scores) if entailment_scores else 0.0
        max_contradiction = max(contradiction_scores) if contradiction_scores else 0.0

        if max_entailment > 0.5:
            verdict = ClaimVerdict.SUPPORTED
            reason = f"Evidence entails claim (entailment: {max_entailment:.2f})"
            confidence = max_entailment
        elif max_contradiction > 0.5:
            verdict = ClaimVerdict.CONTRADICTED
            reason = f"Evidence contradicts claim (contradiction: {max_contradiction:.2f})"
            confidence = max_contradiction
        else:
            verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
            reason = f"NLI classification neutral (entailment: {max_entailment:.2f}, contradiction: {max_contradiction:.2f})"
            confidence = max(neutral_scores) if neutral_scores else 0.5

        return JudgeResult(
            verdict=verdict,
            confidence=confidence,
            evidence=evidence_refs,
            reason=reason,
        )

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, Optional[str]]:
        """Evaluate instruction adherence - NLI not typically used for this."""
        # Simple fallback - could use NLI with instruction as premise
        if "json" in instruction.text.lower():
            if answer.strip().startswith("{") and answer.strip().endswith("}"):
                return 1.0, "Answer is JSON format as instructed"
        return 0.8, "NLI-based instruction evaluation not fully implemented"

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Evaluate scope compliance - NLI not typically used for this."""
        if not policy.scope.enabled:
            return 0.0, []
        return 0.0, []
