---
description: "Run pytest tests with optional scope (all, unit, integration, or a specific file/test name)"
---
# Run Tests

Run the test suite for the SAE pipeline.

## Scope: ${{ scope }}

```bash
.venv/bin/python -m pytest ${{ scope }} -v
```

### Common scopes:
- `tests/` — all 48 tests
- `tests/unit/` — unit tests only
- `tests/integration/` — integration tests only
- `tests/unit/test_sae_module.py` — specific file
- `tests/unit/test_sae_module.py::TestSAEManagerLoad` — specific class
