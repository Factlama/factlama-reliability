"""Vendor-coupled JudgeProvider adapters (ADR-011 T0, in-process).

`EmbeddingProvider` and `NLIProvider` need sentence-transformers or
transformers/torch respectively -- optional extras (`pip install
"factlama-reliability[embeddings]"` / `"[nli]"`). Both load their vendor SDK
lazily inside `_get_model()`, but this module is still kept separate from
`judges.port` and `judges.providers` so the dependency-boundary check
(import-linter, see pyproject.toml) can prove `schemas`/`core` never reach a
vendor model SDK even transitively.
"""

from typing import Any

from judges.port import (
    CancellationToken,
    JudgeError,
    JudgeErrorCode,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
    bounded_check,
)
from schemas.claims import ClaimVerdict, RationaleCode
from schemas.instruction import Instruction
from schemas.policy import Policy


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
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"embedding:{self.model_name}"

    def _get_model(self) -> Any:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for EmbeddingProvider. "
                    "Install with: pip install factlama-reliability[embeddings]"
                ) from exc
        return self._model

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using semantic embedding similarity."""
        error = bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        claim, evidence = request.claim, request.evidence

        if not evidence:
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                rationale_code=RationaleCode.NO_EVIDENCE_SUPPLIED,
                reason="No evidence provided",
            )

        try:
            model = self._get_model()
        except RuntimeError as e:
            return JudgeResult(error=JudgeError(code=JudgeErrorCode.CONFIGURATION, message=str(e)))

        # Encode the claim
        claim_embedding = model.encode(claim.text, convert_to_tensor=True)

        # (evidence_id, support, relevance) per candidate match.
        matches: list[tuple[str, float, float]] = []
        best_similarity = 0.0
        best_evidence = None

        for ev in evidence:
            ev_text = ev.content or ""
            # Encode evidence text
            evidence_embedding = model.encode(ev_text, convert_to_tensor=True)

            # Compute cosine similarity
            import torch.nn.functional as F

            similarity = F.cosine_similarity(
                claim_embedding.unsqueeze(0), evidence_embedding.unsqueeze(0)
            ).item()

            # Also check for numerical contradiction
            has_numerical_contradiction = self._check_numerical_contradiction(claim.text, ev_text)

            if has_numerical_contradiction and similarity > self.support_threshold:
                # High semantic similarity but numerical contradiction
                matches.append((ev.evidence_id, -similarity, similarity))
            elif similarity >= self.support_threshold:
                matches.append((ev.evidence_id, similarity, similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                best_evidence = ev

        # Determine verdict
        if (
            best_evidence is not None
            and self._check_numerical_contradiction(claim.text, best_evidence.content or "")
            and best_similarity > self.support_threshold
        ):
            verdict = ClaimVerdict.CONTRADICTED
            evidence_ids = [best_evidence.evidence_id]
            rationale_code = RationaleCode.NUMERICAL_CONTRADICTION
            reason = "Semantic similarity but numerical contradiction detected"
        elif matches:
            best_support = max(support for _, support, _ in matches)
            if best_support >= self.support_threshold:
                verdict = ClaimVerdict.SUPPORTED
                evidence_ids = [eid for eid, support, _ in matches if support >= 0]
                rationale_code = RationaleCode.PARAPHRASED_SUPPORT
                reason = (
                    f"Claim semantically supported by evidence (similarity: {best_support:.2f})"
                )
            elif best_support <= -self.contradiction_threshold:
                verdict = ClaimVerdict.CONTRADICTED
                evidence_ids = [eid for eid, support, _ in matches if support < 0]
                rationale_code = RationaleCode.NUMERICAL_CONTRADICTION
                reason = f"Claim contradicted by evidence (similarity: {abs(best_support):.2f})"
            else:
                verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
                evidence_ids = []
                rationale_code = RationaleCode.AMBIGUOUS_EVIDENCE
                reason = (
                    f"Evidence exists but similarity below threshold (best: {best_support:.2f})"
                )
        else:
            verdict = ClaimVerdict.UNSUPPORTED
            evidence_ids = []
            rationale_code = RationaleCode.NO_SUPPORT
            reason = "No semantically relevant evidence found"

        # Calculate confidence based on similarity
        confidence = self._calculate_confidence(matches, verdict)

        return JudgeResult(
            verdict=verdict,
            confidence=confidence,
            evidence_ids=evidence_ids,
            rationale_code=rationale_code,
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
        return similarity > 0.6 and claim_numbers != evidence_numbers

    def _calculate_confidence(
        self,
        matches: list[tuple[str, float, float]],
        verdict: ClaimVerdict,
    ) -> float:
        """Calculate confidence in the verdict based on embedding similarity."""
        if verdict == ClaimVerdict.UNSUPPORTED:
            return 0.5

        if not matches:
            return 0.5

        best_relevance = max(relevance for _, _, relevance in matches)
        best_support = max(abs(support) for _, support, _ in matches)

        confidence = (best_relevance + best_support) / 2
        return min(1.0, max(0.0, confidence))

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, str | None]:
        """Evaluate instruction adherence using semantic similarity."""
        model = self._get_model()

        answer_embedding = model.encode(answer, convert_to_tensor=True)
        instruction_embedding = model.encode(instruction.text, convert_to_tensor=True)

        import torch.nn.functional as F

        similarity = F.cosine_similarity(
            answer_embedding.unsqueeze(0), instruction_embedding.unsqueeze(0)
        ).item()

        # For instruction adherence, we check if the answer follows the instruction
        # This is a simple heuristic - in practice, more sophisticated approaches needed
        if (
            "json" in instruction.text.lower()
            and answer.strip().startswith("{")
            and answer.strip().endswith("}")
        ):
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
                answer_embedding.unsqueeze(0), forbidden_embedding.unsqueeze(0)
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
        self._tokenizer: Any = None
        self._model: Any = None
        self._label_indices: dict[str, int] | None = None

    @property
    def name(self) -> str:
        return f"nli:{self.model_name}"

    def _get_model(self) -> tuple[Any, Any]:
        """Lazy-load the NLI model, tokenizer, and its entailment/neutral/contradiction label mapping."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                if self.device:
                    self._model = self._model.to(self.device)
                else:
                    self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError as exc:
                raise RuntimeError(
                    "transformers and torch are required for NLIProvider. "
                    "Install with: pip install factlama-reliability[nli]"
                ) from exc

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
        error = bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        claim, evidence = request.claim, request.evidence

        if not evidence:
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                rationale_code=RationaleCode.NO_EVIDENCE_SUPPLIED,
                reason="No evidence provided",
            )

        try:
            tokenizer, model = self._get_model()
        except RuntimeError as e:
            return JudgeResult(error=JudgeError(code=JudgeErrorCode.CONFIGURATION, message=str(e)))

        # _get_model() always populates this before returning.
        assert self._label_indices is not None
        label_indices = self._label_indices

        import torch

        # (evidence_id, entailment_score, contradiction_score, neutral_score) per item.
        per_evidence: list[tuple[str, float, float, float]] = []

        for ev in evidence:
            # Prepare premise (evidence) and hypothesis (claim) for NLI
            premise = ev.content or ""
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
                entailment_score = probs[label_indices["entailment"]].item()
                neutral_score = probs[label_indices["neutral"]].item()
                contradiction_score = probs[label_indices["contradiction"]].item()

                per_evidence.append(
                    (ev.evidence_id, entailment_score, contradiction_score, neutral_score)
                )

        # Aggregate scores
        entailment_scores = [e for _, e, _, _ in per_evidence]
        contradiction_scores = [c for _, _, c, _ in per_evidence]
        neutral_scores = [n for _, _, _, n in per_evidence]
        max_entailment = max(entailment_scores) if entailment_scores else 0.0
        max_contradiction = max(contradiction_scores) if contradiction_scores else 0.0

        if max_entailment > 0.5:
            verdict = ClaimVerdict.SUPPORTED
            evidence_ids = [eid for eid, e, _, _ in per_evidence if e > 0.5]
            rationale_code = RationaleCode.DIRECT_SUPPORT
            reason = f"Evidence entails claim (entailment: {max_entailment:.2f})"
            confidence = max_entailment
        elif max_contradiction > 0.5:
            verdict = ClaimVerdict.CONTRADICTED
            evidence_ids = [eid for eid, _, c, _ in per_evidence if c > 0.5]
            rationale_code = RationaleCode.CONTRADICTION_DETECTED
            reason = f"Evidence contradicts claim (contradiction: {max_contradiction:.2f})"
            confidence = max_contradiction
        else:
            verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
            evidence_ids = []
            rationale_code = RationaleCode.AMBIGUOUS_EVIDENCE
            reason = f"NLI classification neutral (entailment: {max_entailment:.2f}, contradiction: {max_contradiction:.2f})"
            confidence = max(neutral_scores) if neutral_scores else 0.5

        return JudgeResult(
            verdict=verdict,
            confidence=confidence,
            evidence_ids=evidence_ids,
            rationale_code=rationale_code,
            reason=reason,
        )

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, str | None]:
        """Evaluate instruction adherence - NLI not typically used for this."""
        # Simple fallback - could use NLI with instruction as premise
        if (
            "json" in instruction.text.lower()
            and answer.strip().startswith("{")
            and answer.strip().endswith("}")
        ):
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
