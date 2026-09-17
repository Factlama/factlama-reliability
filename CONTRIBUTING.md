# Contributing to FactLama Reliability

Thank you for contributing to the FactLama Reliability Engine. This repository implements claim and evidence processing, evaluator adapters, scoring, policy, provenance, and reliability APIs.

## Before starting

1. Search existing issues and pull requests to avoid duplicate work.
2. Read the local README and the relevant task and subsystem specifications in the sibling `factlama-architecture` repository. Its component `docs/implementation.md` is the task ledger.
3. For public contract or architecture changes, open an issue first and coordinate the corresponding change in `factlama-architecture`.
4. Never place provider credentials, customer content, or tenant data in source, tests, fixtures, logs, commits, or pull-request descriptions.

## Get a checkout

If you have write access, clone `https://github.com/Factlama/factlama-reliability.git` and use the branch workflow below. Otherwise, fork the repository on GitHub, replace `YOUR_USERNAME` below, and run:

```bash
git clone https://github.com/YOUR_USERNAME/factlama-reliability.git
cd factlama-reliability
git remote add upstream https://github.com/Factlama/factlama-reliability.git
git fetch upstream
git switch -c feat/short-description upstream/main
```

For a fork, push your feature branch to `origin` (your fork) and open the PR against `Factlama/factlama-reliability:main`. In the GitHub CLI command below, add `--repo Factlama/factlama-reliability`. You can also use GitHub's **Compare & pull request** button instead of installing the CLI.

Keep architecture and implementation checkouts side by side when working across repositories. Clone `https://github.com/Factlama/factlama-architecture.git` beside an implementation checkout so canonical contract tests can find it.

## Create a branch (direct collaborators)

Create a focused branch from the latest `main`:

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/short-description
```

Use a clear prefix such as `feat/`, `fix/`, `docs/`, `test/`, or `chore/`.

## Make and verify changes

Use Python 3.10–3.12 to match CI. From the repository root, create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```


Keep the implementation, relevant tests, and documentation in the same pull request. Run the checks appropriate to this repository before committing:

```bash
ruff check .
ruff format --check .
mypy schemas core judges api
lint-imports
pytest
git diff --check
```

Install the development dependencies with `pip install -e ".[dev]"` in a virtual environment when needed. Tests using optional embedding or NLI providers also require their corresponding extras. Contract-fixture tests use the sibling `factlama-architecture` checkout or `FACTLAMA_CONTRACTS_DIR`.

## Commit with sign-off

FactLama uses the [Developer Certificate of Origin](https://developercertificate.org/) sign-off process. By adding a `Signed-off-by` line, you certify that you created the contribution or have the right to submit it under this repository's license.

Configure your real identity, stage only the intended files, and create a signed-off commit:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
git add path/to/changed-file
git commit --signoff -m "feat: describe the reliability change"
```

Do not submit employer-owned work without permission, copied code, secrets, proprietary datasets or model weights, or material with an incompatible license.

## Open a pull request

Push the branch and open a pull request against `main`:

```bash
git push -u origin feat/short-description
gh pr create --base main --fill
```

The pull request description should state:

- the concrete problem and resulting behavior;
- affected REL task or execution-plan gate;
- tests and checks run;
- contract and backward-compatibility impact;
- tenant, security, privacy, failure-path, and cost impact where applicable;
- related architecture documentation changes.

Respond to review comments with additional signed-off commits. Avoid force-pushing during active review unless reviewers agree.

## Review and merge

Open a draft PR early if you need design feedback. Link the issue and any companion PR in another repository. Include the actual commands and results used for validation; identify skipped checks. Request maintainer review and wait for applicable CI checks and approval before merge. Preserve authorship and sign-off trailers when squashing commits.

Every new contribution commit must carry your DCO sign-off. Read the full [DCO 1.1](https://developercertificate.org/) before signing: it also explains that your contribution and sign-off become public records. A sign-off is a certification of your right to contribute, not a cryptographic signature or copyright transfer. Contributors retain their copyrights; no CLA is currently required.

For your latest local, unpublished commit, a missing sign-off can be added with:

```bash
git commit --amend --no-edit --signoff
```

Only certify work you have the right to submit. Coordinate with maintainers before rewriting a published branch. Sign-offs are reviewed manually unless the repository has a DCO check configured; this guide does not enable GitHub branch protection or a DCO app.

## Licensing

This repository is licensed under the Apache License 2.0. Unless you explicitly state otherwise, every contribution intentionally submitted for inclusion is provided under Apache 2.0, consistent with section 5 of the license. Third-party SDKs, model weights, datasets, and generated assets retain their own licenses and require separate review.


See [LICENSE](LICENSE), [NOTICE](NOTICE), and [LICENSE-HISTORY.md](LICENSE-HISTORY.md) for the license text, attribution, and transition from MIT.
