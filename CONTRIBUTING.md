# Contributing to citeguard

Thanks for your interest in contributing to citeguard.

## Development setup

```bash
git clone https://github.com/mmustafasenoglu/citeguard.git
cd citeguard
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality checks

Run all quality checks before opening a pull request:

```bash
ruff check .
pytest -q
python -m build
python scripts/run_benchmarks.py --suite all
```

## Zero-network test policy

- The default test suite (`pytest -q`) must be fully offline and deterministic
- Never make live API calls during tests; mock all provider responses
- Do not require paid API access for the default test suite
- Use synthetic or openly licensed excerpts in test fixtures
- Live provider smoke tests are available under `pytest -m live_llm` and require explicit opt-in with valid API keys

## Benchmark runner

The benchmark suite validates deterministic quality across releases:

```bash
python scripts/run_benchmarks.py --suite all
```

- **Calibration suite** (79 cases): internal quality gates; must pass before merging
- **Holdout suite** (59 cases): release gates; must pass before tagging a release
- Each case asserts a specific score range for a claim on a fixture document
- Results are printed to the terminal and can be compared across runs

## Code style

- Keep lines within 100 characters
- All code, comments, docstrings, CLI output, tests, issues, and documentation must be in English
- Prefer small, deterministic Python functions for parsing, normalization, matching, and scoring
- Support Python 3.10 through 3.12

## Architecture

- Core modules (`extractor`, `claims`, `linking`, `matcher`, `verification`, `scoring`) must never import from `citeguard.cli`
- Put provider-specific behavior behind the `SourceProvider` interface
- Use `ExecutionContext` for cross-cutting concerns (network usage, LLM availability)
- Preserve the distinction between source resolution, metadata match, claim support, and overall confidence

## Branch and commit conventions

- Use short, descriptive branch names: `fix/numbered-citation-overlap`, `feat/openalex-provider`
- Write commit messages in English with a clear subject line
- Keep commits focused: one logical change per commit

## Test expectations

- Every parser, matcher, scorer, cache, or provider change should include tests
- Add regression tests for fixed bugs
- Mock provider responses; never make live API calls during tests
- Use synthetic or openly licensed excerpts in test fixtures

## Privacy guidelines

- Never commit API keys, `.env` files, private papers, unpublished documents, or copyrighted full-text fixtures
- Never print secrets in CLI output, logs, screenshots, or issue reports
- Custom base URLs with credentials are sanitized before printing
- Use non-sensitive examples in bug reports

## Bug reports

Please include:

- citeguard version
- Python version and OS
- Input file type and citation style
- A minimal non-sensitive example
- Expected behavior
- Actual behavior

## Good first contribution areas

- Additional citation-regex fixtures
- Bibliography parser edge cases
- DOI/title normalization cases
- Report formatting improvements
- Documentation improvements
- Test coverage for provider normalization
