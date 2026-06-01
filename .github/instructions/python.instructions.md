---
applyTo: "**/*.py"
description: "Python conventions for the SAE pipeline: frozen dataclasses, import style, type hints, testing patterns"
---

# Python Conventions

## Imports
- Use `from __future__ import annotations` in all modules
- Within `src/`: sibling imports (`from autoencoder.sae_module import SAEManager`, `import utils`)
- In tests: direct imports work because `conftest.py` adds `src/` to `sys.path`

## Data Structures
- All configs and contracts use `@dataclass(frozen=True)` — never mutate after creation
- Validation in `__post_init__` methods
- Use `Path` objects for file paths, not raw strings

## Safety
- All tensor loading via `utils.load_tensor()` which enforces `weights_only=True`
- Seed propagation via `utils.set_global_seed()` (random, numpy, torch, cuda, cudnn)

## Testing
- CPU-only, no real model weights needed
- Class-based grouping for related tests
- Fixtures from `tests/conftest.py` — check there before creating new ones
- Deterministic seeds: 42 and 123
