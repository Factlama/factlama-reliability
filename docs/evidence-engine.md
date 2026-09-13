# Evidence engine

MVP evidence comes from the request. Validate unique IDs, allowed content/reference form, source metadata and payload limits. An evidence reference must be resolvable through an authorized source before judging; an unresolved reference is an unavailable evidence condition. Retrieval is an optional `EvidenceRetriever` port, called only when policy and tenant authorization explicitly allow it. No vector database is an MVP dependency.

Map each claim to candidate evidence by explicit citation first, then bounded lexical/semantic candidate selection if configured. Candidate similarity is routing only, never a support verdict. Retain source IDs, retrieval method/version, rank, source timestamp and trust label. Contradictory evidence must be passed to the judge or surfaced as a conflict; do not discard it because another item supports the claim. Missing evidence yields `INSUFFICIENT_EVIDENCE` when the corpus is absent/unavailable and `UNSUPPORTED` only when a usable, scoped corpus was examined but no support found. The judge may still abstain on ambiguous evidence.

Tests must cover empty corpus, inaccessible reference, duplicate IDs, irrelevant but similar text, conflicting sources, stale source metadata, citation mismatch, and tenant-isolated retrieval.
