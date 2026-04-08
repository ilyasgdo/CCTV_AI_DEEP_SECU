# Contributing

## Development flow

1. Read `DOCS/PLAN/00_REGLES_DEVELOPPEMENT.md`.
2. Create a branch for your change.
3. Implement one scoped change at a time.
4. Run tests:
   - `venv\\Scripts\\python.exe -m pytest tests/ -q`
5. Update docs when behavior changes.

## Commit format

Use:

- `[ETAPE-XX] feat: short description`
- `[ETAPE-XX] fix: short description`
- `[ETAPE-XX] test: short description`
- `[ETAPE-XX] docs: short description`

## Pull request checklist

- Tests pass
- No secrets committed
- Scope aligns with stage plan
- Summary doc updated if stage completed
