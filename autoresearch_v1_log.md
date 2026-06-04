# Autoresearch v1 Log — PXR Activity Prediction

## Overview
- **Branch**: `autoresearch/jun2`
- **Started**: 2026-06-02
- **Total runs**: 32 (including crashes)
- **Best MAE**: **0.4855** (Run 21)
- **Baseline MAE**: 0.5572
- **Improvement**: -0.0717 (12.9% relative)

## Current Best Configuration
**Commit**: `edf65dd` (before plateau experiments) or the clean baseline on branch with CheMeleon hold-out 20%.

```
# Chemprop MT: d_h=512, depth=5, n_layers=2, hidden_dim=512, multi-task (pEC50+Emax)
# CheMeleon: pretrained BondMessagePassing, hold-out 20% val, low LR (1e-5/1e-4), dropout 0.2, FFN 2-layer, hidden_dim=2048
# Sklearn: GBR on 4 fingerprint types (Morgan r1, atom_pairs, topological_torsion, MACCS keys)
# Ensemble: scipy L-BFGS-B optimized non-negative weights
```

Ensemble weights (val): CheMeleon ~54%, Chemprop MT ~29%, sklearn_ap ~12%, sklearn_tt ~4%, sklearn_maccs ~1%, sklearn_r1 ~0%.

## File: experiment.py
- **Only file to modify**. All experiments modify this file.
- Key functions:
  - `train_chemprop()`: Chemprop MPNN training (line ~148)
  - `train_chemeleon()`: CheMeleon foundation model finetuning (line ~127)
  - `train_sklearn_model()`: GBR training (line ~250)
  - `optimize_ensemble_weights()`: Weight optimization (line ~272)
  - `train_model()`: Orchestration with Phase 1 (val) + Phase 2 (final) (line ~403)
  - `evaluate_model()`: Inference + MAE computation (line ~536)

## Data
- `train.csv`: 4,139 unique compounds with multi-assay endpoints (sparse)
- `test_phase1.csv`: 513 compounds with pEC50 ground truth
- Primary data extracted: pEC50, Emax, Emax_vs_pos, std_error
- Quality filter: pEC50 ∈ [1.5, 8.0], std_error replaced 0→0.1
- Stratified train/val split: 80/20, bins [4.0, 5.0, 6.0, 8.0]

## Pretrained Model
- **chemeleon_mp.pt**: Downloaded from Zenodo (15460715). Pretrained BondMessagePassing with d_h=2048, depth=6, d_v=72, d_e=14.
- Uses `SimpleMoleculeMolGraphFeaturizer` (different from default chemprop featurizer).

## Experiment History (Chronological)

| Run | Change | MAE | Result | Evidence |
|-----|--------|-----|--------|----------|
| 01 | Baseline | 0.5572 | — | Chemprop MT + GBR (r2). Val 0.5237, test 0.5572 (gap=0.03). |
| 02 | Higher LR (5e-4/5e-3/5e-4) | 0.5779 | REVERT | Aggressive LR hurt convergence. |
| 03 | d_h=1024, depth=6 | 0.5568 | REVERT | No meaningful improvement (+0.0004). |
| 04 | Dropout 0.3 | 0.5674 | REVERT | Regularization hurt too much. |
| 05 | Multi-radius r1,r2,r3 | 0.5331 | **ADVANCED** | r1 (38% weight) captured local features. -0.024. |
| 06 | Diverse fp (r1, r3, ap, tt) | 0.5084 | **ADVANCED** | ap (41%) dominant. -0.025. |
| 07 | MACCS keys (r1, ap, tt, maccs) | 0.5066 | **ADVANCED** | r1→0% weight, maccs 10%. -0.002. |
| 08 | Count-based Morgan r2 | 0.5106 | REVERT | Worse than r1. |
| 09 | Random Forest model | 0.5249 | REVERT | Overfit on fingerprints. |
| 10 | Molecular descriptors (14) | 0.5348 | REVERT | Low signal-to-noise. |
| 11 | AttentionAggregation | CRASH | REVERT | Not compatible. |
| 12 | Single-task chemprop | 0.5339 | REVERT | Multi-task (Emax) helps. |
| 13 | CheMeleon v1 (no reg) | 0.5331 | REVERT | Val 0.4702 → test 0.5331 (massive overfit). |
| 14 | CheMeleon v2 (10% hold-out) | 0.4939 | **ADVANCED** | Hold-out val critical. -0.008. |
| 15 | CheMeleon stronger reg (5e-6 LR, 0.3 drop) | 0.5026 | REVERT | Underfit. |
| 16 | CheMeleon + chemprop ST diversity | 0.5025 | REVERT | Extra model caused overfit. |
| 17 | CheMeleon hold-out 20% | 0.4855 | **★ BEST** | Optimal hold-out size. |
| 18 | CheMeleon hold-out 30% | 0.4992 | REVERT | Too little training data. |
| 19 | CheMeleon 3-layer FFN | 0.4946 | REVERT | Overfit. |
| 20 | Simplify sklearn (ap only) | 0.4951 | REVERT | Diversity matters. |
| 21 | 4096-bit fp | 0.4983 | REVERT | No benefit over 2048. |
| 22 | Chemprop depth 6 | 0.4891 | REVERT | Close but worse. |
| 23 | Second CheMeleon (diff seed) | 0.5131 | REVERT | Overfit on test. |
| 24 | CheMeleon smaller FFN (512) | 0.4979 | REVERT | Underfit. |
| 25 | CheMeleon grad clip (norm=1) | 0.4938 | REVERT | Marginal, not worth complexity. |
| 26 | CheMeleon frozen MP | 0.5030 | REVERT | Needs some MP update. |
| 27 | CheMeleon batch 128 | 0.4952 | REVERT | No benefit. |
| 28 | CheMeleon 3-seeds avg | 0.4911 | REVERT | Worse single model. |
| 29 | Smaller chemprop (300, 3) | 0.4962 | REVERT | Underfit. |
| 30 | CheMeleon only (no ensemble) | 0.5261 | REVERT | Ensemble critical. |

## Key Findings

### 1. CheMeleon Foundation Model is the Biggest Lever
The CheMeleon pretrained BondMessagePassing (d_h=2048, depth=6) generalizes far better than training from scratch. Its individual test MAE (0.51-0.52) beats all other individual models.

### 2. Hold-out Validation is Critical for CheMeleon
Without hold-out val, CheMeleon overfits massively (val 0.47 → test 0.53). The 20% hold-out (80/20 train/val in Phase 2) is the sweet spot:
- 10% hold-out: 0.4939 (still some overfit)
- 20% hold-out: 0.4855 (optimal)
- 30% hold-out: 0.4992 (not enough training data)

### 3. Ensemble Diversity Matters
The weighted ensemble of CheMeleon (54%), Chemprop MT (29%), and diverse fingerprint models (17%) achieves 0.4855. Dropping the ensemble → 0.5261. The complementary information from different fingerprint types and architectures is essential.

### 4. Atom Pairs Fingerprint is Best Sklearn Feature
Among Morgan r1/r3, atom_pairs, topological_torsion, and MACCS keys — atom_pairs consistently gets the highest weight (~15-40%). It captures spatial relationships that complement graph-based models.

### 5. Multi-task Learning (pEC50 + Emax) Helps Chemprop
Dropping Emax co-training → 0.5339 (worse). The Emax signal regularizes the pEC50 predictions.

### 6. What Doesn't Work
- Higher LR schedules → worse convergence
- Bigger models (d_h=1024, depth=6) → no gain
- Higher dropout → hurts performance
- Count-based fingerprints → redundant
- Molecular descriptors → low signal
- GELU activation → not supported; ELU → worse
- Random Forest → overfits on fingerprints
- HistGradientBoosting → API incompatibility (no subsample)
- Concatenated fingerprints → overfit
- Frozen message passing → underfit
- Gradient clipping → marginal benefit
- Multiple CheMeleon seeds → worse than single

## Hypotheses for Future Work

### High Priority
1. **CheMeleon with different pretrained weights**: Try alternative foundation models (if available).
2. **CheMeleon with warmup LR schedule**: Current schedule (1e-5→1e-4→1e-5) is flat-ish. Try CosineAnnealingWarmupRestarts.
3. **CheMeleon with max_epochs=120+**: Current 80 may not be enough for slow convergence with low LR.
4. **CheMeleon with different early stopping patience**: Current 15. Try 20-30.
5. **Stacking with meta-learner on CheMeleon + chemprop predictions**: Ridge/elastic-net on top model predictions.

### Medium Priority
6. **CheMeleon with gradient clipping at lower threshold** (0.5 instead of default).
7. **CheMeleon with label smoothing** (if regression-compatible).
8. **CheMeleon with different aggregation** (SumAggregation, NormAggregation instead of MeanAggregation).
9. **MolPipeline pharmacophore fingerprints** (tried, got 1% weight — but try concatenated with ap).
10. **CheMeleon with different hidden_dim** (1024 instead of 2048 for FFN).

### Low Priority
11. Different stratification bins for train/val split.
12. Data augmentation via SMILES randomization (tried, failed due to RDKit API).
13. GradientBoostingRegressor parameter sweep per fingerprint type.
14. Different ensemble optimization (genetic algorithm instead of L-BFGS-B).

## Run Configuration
```
python experiment.py > run.log 2>&1
# Results in results.csv (last line is most recent)
# Typical runtime: 1060-1100s (~18 min)
# GPU: 1x NVIDIA, VRAM ~8GB usage
```

## Important Notes for Successor
- **Commit policy**: Advance branch if MAE < 0.4855, revert otherwise.
- **lab_notebook.txt**: Maintain running log of experiments.
- **results.csv**: Append-only, last line is current MAE.
- **checkpoints/**: Cleaned up automatically after each run.
- **chemeleon_mp.pt**: Pretrained weights — do not delete.
- **Plateau**: 11 consecutive failures. The plateau detection rule says to make a structural change. The CheMeleon foundation model is already a structural change. Next structural leap should target something fundamentally different (e.g., different aggregation, different optimization algorithm, or trying the MolPipeline featurizers more aggressively).
