# Security Policy

## Supported versions

- `1.0.x` (MVP)
- `1.1.x` (advanced features)

## Reporting a vulnerability

Please report security issues privately to the maintainers.
Do not open a public issue with exploit details.

Include:

- Affected version
- Reproduction steps
- Impact assessment
- Suggested remediation (optional)

## Security practices in this project

- Dashboard authentication enabled by default
- API input validation and rate limiting
- Environment-based secret management (`.env`)
- Optional encryption at rest for whitelist embeddings
