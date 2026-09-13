# Claim engine

MVP accepts explicit claims or extracts atomic factual propositions from `answer`. Preserve an answer offset when possible; if extraction rewrites text, retain the original text span and a normalized proposition separately. Use deterministic IDs derived from request ID plus ordinal for extracted claims; explicit IDs are preserved. Stable IDs are scoped to a logical request, not globally unique.

Split conjunctions when each part can be independently supported ("founded in 2018 by Jane" yields founding year and founder claims). Keep qualifiers, negation, quantities and time constraints. Do not turn questions, hedges or quoted material into asserted facts without an explicit rule. If the answer contains no checkable claims, produce `ABSTAIN` with a reason code. Claim extraction errors produce an evaluator failure/abstention, not a factual `FAIL`.

Golden fixtures must cover compound statements, negation, dates, numbers, attributions, ambiguous pronouns, repeated claims, empty answers and explicit claims. Record extractor ID/version in provenance; changing segmentation may change scores and therefore needs benchmark review.
