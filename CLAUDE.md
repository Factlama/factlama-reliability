# Claude Implementation Context

Read the architecture repository before implementation. Work one vertical slice at a time. Do not change public contracts or architecture boundaries without an ADR.

For each task use status values NOT_STARTED, IN_PROGRESS, BLOCKED, COMPLETE. Mark COMPLETE only after implementation, unit tests, integration tests, failure-path tests, tenant-isolation checks, lint/type checks and documentation are complete.

Required loop: READ -> INSPECT EXISTING CODE -> PLAN -> IMPLEMENT -> TEST -> REVIEW -> DOCUMENT.
