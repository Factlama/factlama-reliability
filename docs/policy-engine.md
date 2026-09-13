# Policy engine

Policy evaluates a completed factual result and tenant-approved immutable policy version. Inputs are verdict, normalized violations, score statuses/values, evaluation mode, application, and optional risk tier. Rules are ordered explicitly; first matching rule wins, with a documented default `PASS` only for factual `PASS`. The default for `PARTIAL`, `FAIL`, `ABSTAIN`, or technical failure is `HUMAN_REVIEW` in advisory mode; a deployment may configure a stricter action. Policy never claims a failed provider proved hallucination.

Actions: `PASS`, `FAIL`, `REGENERATE`, `RETRIEVE_AGAIN`, `SWITCH_MODEL`, `ASK_USER`, `BLOCK`, `HUMAN_REVIEW`. `FAIL` is a policy action distinct from factual `verdict=FAIL`. Each decision records `policy_id`, version, matched rule ID, inputs used, action, timestamp and whether it is advisory or enforced. An enforced action requires explicit integration at the customer application's decision point; an async observation cannot retroactively block a completed answer.

Routing is bounded: maximum attempts, total deadline, allowed provider list, and no cyclic regenerate/retrieve loops. A fallback on judge failure records both attempts and does not conceal the initial failure. Policy configuration is tenant-scoped and auditable. Test rule precedence, missing scores, abstention, conflicting signals, fallback exhaustion and version replay.
