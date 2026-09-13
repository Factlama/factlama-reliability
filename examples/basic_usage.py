"""Example usage of FactLama verification."""

from core import verify
from schemas import Evidence, Policy
from schemas.policy import GroundingPolicy, HallucinationPolicy

# Example 1: Simple verification with supporting evidence
print("=" * 60)
print("Example 1: Supported Claim")
print("=" * 60)

result = verify(
    question="When was Company X founded?",
    answer="Company X was founded in 2018.",
    evidence=[
        Evidence(
            id="doc_001",
            extracted_text="Company X was founded in 2018.",
        )
    ],
)

print(f"Verdict: {result.verdict}")
print(
    f"Groundedness: {result.scores['groundedness'].value:.2f} ({result.scores['groundedness'].status.value})"
)
print(f"Hallucination Risk: {result.scores['hallucination_risk'].value:.2f}")
print()

# Example 2: Verification with partial support
print("=" * 60)
print("Example 2: Partially Supported Claim")
print("=" * 60)

result = verify(
    question="Tell me about Company X.",
    answer="Company X was founded in 2018 by John Smith.",
    evidence=[
        Evidence(
            id="doc_001",
            extracted_text="Company X was founded in 2018.",
        )
    ],
)

print(f"Verdict: {result.verdict}")
print(f"Groundedness: {result.scores['groundedness'].value:.2f}")
print(f"Claims verified: {len(result.claims)}")
for claim in result.claims:
    print(f"  - {claim.claim_id}: {claim.verdict} (confidence: {claim.confidence:.2f})")
print()

# Example 3: Verification with contradiction
print("=" * 60)
print("Example 3: Contradicted Claim")
print("=" * 60)

result = verify(
    question="What does Product X weigh?",
    answer="Product X weighs 3.4 kg.",
    evidence=[
        Evidence(
            id="doc_001",
            extracted_text="Product X weighs 2.4 kg.",
        )
    ],
)

print(f"Verdict: {result.verdict}")
print(f"Contradiction Risk: {result.scores['contradiction_risk'].value:.2f}")
print()

# Example 4: Verification with policy constraints
print("=" * 60)
print("Example 4: Verification with Policy")
print("=" * 60)

policy = Policy(
    id="strict-policy",
    grounding=GroundingPolicy(minimum=0.90),
    hallucination=HallucinationPolicy(maximum=0.10),
)

result = verify(
    question="What are the company's products?",
    answer="The company makes widgets.",
    evidence=[
        Evidence(
            id="doc_001",
            extracted_text="The company manufactures widgets and gadgets.",
        )
    ],
    policy=policy,
)

print(f"Verdict: {result.verdict}")
print(f"Policy Action: {result.policy_action}")
print()

# Example 5: No evidence supplied -- the judge cannot confirm anything, so it abstains
print("=" * 60)
print("Example 5: No Evidence Provided")
print("=" * 60)

result = verify(
    question="When was Company X founded?",
    answer="Company X was founded in 2018.",
)

print(f"Verdict: {result.verdict}")
print(f"Status: {result.status} (abstention_reason: {result.abstention_reason})")
print(f"Groundedness: {result.scores['groundedness'].value:.2f}")
print("(Without evidence, the claim cannot be confirmed -- no optimistic default.)")
print()

print("=" * 60)
print("All examples completed successfully!")
print("=" * 60)
