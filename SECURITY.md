# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please report security-sensitive issues privately rather than opening a public issue. Use [GitHub's private vulnerability reporting](https://github.com/mmustafasenoglu/citeguard/security/advisories/new) when available.

We aim to acknowledge reports within 48 hours and provide a resolution timeline within 7 days.

## Secrets policy

- Never include API keys, access tokens, or credentials in issues, logs, screenshots, fixtures, or example repositories
- citeguard must not print environment-variable values or request authorization headers
- Reports and cache files must contain only data required for citation analysis
- API keys are never written to generated reports or cache metadata

## Document privacy

citeguard is a local CLI, but the analysis pipeline sends limited document-derived content to configured third-party APIs:

- **Academic metadata providers**: search queries derived from citation text and bibliography entries; never full document prose
- **LLM providers** (when configured): paragraph text for claim extraction and source matching

Review the Privacy section in `README.md` before processing confidential or unpublished documents.

## Scope

This security policy applies to the citeguard CLI tool and its official GitHub repository. It does not cover third-party academic metadata providers or LLM APIs used by the tool.
