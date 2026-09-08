# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| <1.0    | No        |

## Reporting a vulnerability

Please report security-sensitive issues privately rather than opening a public issue. Use [GitHub's private vulnerability reporting](https://github.com/mmustafasenoglu/citeguard/security/advisories/new) when available.

We aim to acknowledge reports within 48 hours and provide a resolution timeline within 7 days.

## Secrets policy

- Never include API keys, access tokens, or credentials in issues, logs, screenshots, fixtures, or example repositories
- citeguard must not print environment-variable values or request authorization headers
- Custom base URLs containing credentials are sanitized before printing in CLI output, logs, and reports
- Reports and cache files must contain only data required for citation analysis
- API keys are never written to generated reports or cache metadata

## Document privacy

citeguard is a local CLI, but the analysis pipeline may send limited document-derived content to configured third-party APIs depending on which features are enabled:

- **Academic metadata providers**: search queries derived from citation text and bibliography entries; never full document prose
- **LLM providers** (when configured and enabled): paragraph text for claim extraction, entailment classification, and source matching
- Network usage is reported per-run via `ExecutionContext` fields (`academic_network_used`, `llm_network_used`) so you can verify whether any remote calls were made
- When running in offline/cache-only mode, no data leaves the local machine

Review the Privacy section in `README.md` before processing confidential or unpublished documents.

## Scope

This security policy applies to the citeguard CLI tool and its official GitHub repository. It does not cover third-party academic metadata providers or LLM APIs used by the tool.
