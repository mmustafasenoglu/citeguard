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

## Before opening a pull request

```bash
ruff check .
pytest -q
```

## Branch and commit conventions

- Use short, descriptive branch names: `fix/numbered-citation-overlap`, `feat/openalex-provider`
- Write commit messages in English with a clear subject line
- Keep commits focused: one logical change per commit

## Test expectations

- Every parser, matcher, scorer, cache, or provider change should include tests
- Keep the default test suite offline and deterministic
- Mock provider responses; never make live API calls during tests
- Do not require paid API access for the default test suite
- Use synthetic or openly licensed excerpts in test fixtures

## Code style

- Keep lines within 100 characters
- Keep code, comments, docstrings, CLI output, tests, issues, and documentation in English
- Prefer small, deterministic Python functions
- Put provider-specific behavior behind the `SourceProvider` interface

## Privacy guidelines

- Never commit API keys, `.env` files, private papers, unpublished documents, or copyrighted full-text fixtures
- Never print secrets in CLI output, logs, screenshots, or issue reports
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
