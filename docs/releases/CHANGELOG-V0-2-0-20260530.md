# CHANGELOG v0.2.0 - 2026-05-30

## Summary

Refactoring of the SAE pipeline for clarity, idiomaticity. All 40 tests passing after each change.

## Changes

### src/config.py - Dataclass-based configuration

- Converted all module-level variables to frozen dataclasses:
  `PathsConfig`, `BackboneConfig`, `SAEConfig`, `TrainingConfig`,
  `ExplanationConfig`, `HardwareConfig`
- Removed backward-compat module-level aliases (scripts already migrated)
- `PathsConfig`: removed `frozen=True` + `object.__setattr__` workaround,
  replaced with simple `__post_init__` for derived paths
- Derived paths now reference parent fields (`self.embeddings_dir` instead
  of re-computing `self.project_root / "embeddings"`)

### src/sae_module.py - SAEManager facade

- Replaced manual batch generator (`_batch_generator`) with
  `torch.utils.data.DataLoader` + `TensorDataset` + infinite loop
- Replaced all `assert` statements with `ValueError` for proper runtime errors
- Trimmed verbose comments to concise, non-obvious ones
- Removed em-dashes and Unicode symbols from docstrings

### src/02a_train_sae.py - Multi-seed SAE training

- Migrated from `config.VARIABLE` to `config.sae.*`, `config.paths.*`,
  `config.training.*`, `config.hardware.*`
- Removed overly verbose educational comments

### src/02b_concept_naming.py - Concept naming via cosine similarity

- Migrated to dataclass config style
- Removed redundant comments

### src/02c_generate_explanations.py - Structured explanation generation

- Migrated to dataclass config style
- Removed redundant comments and Unicode dividers

### src/02d_stability_analysis.py - Multi-seed stability analysis

- Migrated to dataclass config style
- Removed redundant comments and Unicode dividers
- Replaced Unicode symbols (intersection/union) with plain text

## Files changed

| File | Insertions | Deletions |
|------|-----------|-----------|
| src/config.py | +65 | -35 |
| src/sae_module.py | +30 | -16 |
| src/02a_train_sae.py | +13 | -13 |
| src/02b_concept_naming.py | +13 | -15 |
| src/02c_generate_explanations.py | +13 | -14 |
| src/02d_stability_analysis.py | +17 | -16 |

## Validation

- 40 tests passing (pytest 9.0.3, 8.74s)
- No regressions in any pipeline script
