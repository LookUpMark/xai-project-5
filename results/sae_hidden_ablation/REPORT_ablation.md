# Path A — dict_size/k Ablation

_Generated: 2026-06-26_

## Summary

Path A dict_size x k sweep (steps=8000, seeds=[0, 42, 123, 456, 789]), reusing cached standard_hidden embeddings. dead% is activation-based (train); match/null is the cross-seed best-match cosine over the isotropic null.

## Presets

| preset | dict_size | k |
| --- | --- | --- |
| conservative | 1024 | 16 |
| default | 2048 | 32 |
| aggressive | 4096 | 64 |

## Results

| preset | dict_size | k | dead% | recon cos | L0 | naming mean | match cos | null cos | match/null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | 1024 | 16 | 41.7 | 0.969 | 16.0 | 0.4834 | 0.325 | 0.117 | 2.78x |
| default | 2048 | 32 | 12.9 | 0.973 | 32.0 | 0.4711 | 0.325 | 0.124 | 2.63x |
| aggressive | 4096 | 64 | 6.6 | 0.977 | 64.0 | 0.4782 | 0.260 | 0.130 | 2.00x |

## Notes

- **dead%**: activation-based (train_hidden); baseline 512-d was 40-60%.
- **naming mean**: mean top-1 cosine of live decoder features vs RadLex (random ~0.372).
- **match cos / null cos**: cross-seed decoder-cosine best-match vs isotropic null (>1x = shared subspace above chance).
- Per-preset stage REPORTs under `results/sae_hidden_ablation/{preset}/`.

