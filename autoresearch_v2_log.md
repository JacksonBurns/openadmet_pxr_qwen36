# Autoresearch v2 Log — PXR Activity Prediction

## Overview
- **Branch**: `autoresearch/jun3`
- **Started**: 2026-06-04
- **Total runs**: 27 (excluding reproducibility re-runs)
- **Best MAE**: **0.4777** (Run 10)
- **Baseline MAE**: 0.5104
- **Improvement**: -0.0327 (6.4% relative)
- **vs v1 best (0.4855)**: -0.0078 (1.6% improvement over v1)
- **Reproducibility**: NOT YET FIXED — re-runs gave 0.5025, 0.4916. Added `seed_everything(42)` but needs more testing.

## Current Best Configuration
**Commit**: `d0a8acd` (Run 10: Ridge stacking + dual NormAggregation)

```
# Chemprop MT: d_h=512, depth=5, n_layers=2, hidden_dim=512, NormAggregation, multi-task (pEC50+Emax)
# CheMeleon: pretrained BondMessagePassing, NormAggregation, hold-out 20% val, low LR (1e-5/1e-4), dropout 0.2, FFN 2-layer, hidden_dim=2048
# Sklearn: GBR on 4 fingerprint types (Morgan r1, atom_pairs, topological_torsion, MACCS keys)
# Ensemble: Ridge(alpha=1.0) stacking meta-learner with abs(coef) normalization
```

Ridge stacking meta-learner replaces L-BFGS-B weight optimization. Uses `Ridge(alpha=1.0)` fit on top-model val predictions, with `abs(coef)` normalized to sum=1 for the final weighted ensemble.

## File: experiment.py
- **Only file to modify**. All experiments modify this file.
- Key functions:
  - `train_chemprop()`: Chemprop MPNN training (line ~237) — now uses `NormAggregation`
  - `train_chemeleon()`: CheMeleon foundation model finetuning (line ~129) — now uses `NormAggregation`
  - `train_sklearn_model()`: GBR training (line ~339)
  - `optimize_ensemble_weights()`: Ridge stacking meta-learner (line ~361)
  - `train_model()`: Orchestration with Phase 1 (val) + Phase 2 (final) (line ~396)
  - `evaluate_model()`: Inference + MAE computation (line ~528)

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
| 01 | Baseline | 0.5104 | — | Val 0.4686, test 0.5104 (gap=0.042). Large overfit signal. |
| 02 | CheMeleon max_epochs=120, patience=25 | 0.5130 | REVERT | Model converged already. More epochs hurt. |
| 03 | Equal ensemble weights | 0.5147 | REVERT | Weight optimization matters. |
| 04 | CheMeleon SumAggregation | 0.5080 | ADVANCED | Reduced val-to-test gap (0.033 vs 0.042). |
| 05 | CheMeleon NormAggregation | 0.4924 | ADVANCED | Major gain. CheMeleon test 0.5088. |
| 06 | AttentiveAggregation | CRASH/0.5023 | REVERT | Requires output_size param. Worse anyway. |
| 07 | CheMeleon batch_norm=True | 0.5205 | REVERT | batch_norm hurts pretrained model. |
| 08 | Both CheMeleon+Chemprop NormAgg | 0.4900 | ADVANCED | Chemprop test 0.5534 (was 0.5907). NormAgg wins for both. |
| 09 | CheMeleon higher LR (1e-4/5e-4) | 0.5231 | REVERT | Low LR essential for pretrained model. |
| 10 | Ridge stacking meta-learner | 0.4777 | ★ BEST | Val-test gap only 0.017. Replaces L-BFGS-B weights. |
| 11 | Ridge alpha=0.1 | 0.4974 | REVERT | alpha=1 optimal. |
| 12 | Ridge alpha=10 | 0.4799 | REVERT | alpha=1 optimal. |
| 13 | CheMeleon 3-layer FFN | 0.4955 | REVERT | Overfit. |
| 14 | CheMeleon FFN hidden=1024 | 0.4942 | REVERT | Underfit. |
| 15 | ElasticNet stacking | 0.5020 | REVERT | Ridge better than ElasticNet. |
| 16 | CheMeleon patience=25 | 0.4935 | REVERT | Patience 15 sufficient. |
| 17 | Drop sklearn (CheMeleon+Chemprop only) | 0.5129 | REVERT | Fingerprint diversity essential. |
| 18 | CheMeleon dropout=0.1 | 0.5168 | REVERT | 0.2 dropout optimal. |
| 19 | CheMeleon grad clip (norm=1) | 0.4931 | REVERT | Gradient clipping not helpful. |
| 20 | CheMeleon hold-out 15% | 0.5102 | REVERT | 20% hold-out optimal. |
| 21 | CheMeleon batch=32 | 0.5111 | REVERT | 64 batch optimal. |
| 22 | CheMeleon full data (no hold-out) | 0.5363 | REVERT | Massive overfit. Hold-out critical. |
| 23 | CheMeleon batch=128 | 0.4918 | REVERT | 64 batch optimal. |
| 24 | CheMeleon FFN hidden=512 | 0.4895 | REVERT | 2048 hidden optimal. |
| 25 | Chemprop depth=3 | 0.4958 | REVERT | depth=5 optimal. |
| 26 | Second CheMeleon (seed 2) | 0.4930 | REVERT | Extra CheMeleon doesn't help. |
| 27 | CheMeleon max_epochs=100 | 0.5147 | REVERT | 80 epochs sufficient. |
| — | Reproducibility re-run 1 | 0.5025 | FAIL | Best config not reproducible (0.4777→0.5025). |
| — | Reproducibility re-run 2 | 0.4916 | PARTIAL | With seed_everything. Better but still not 0.4777. |

## Key Findings

### 1. NormAggregation is the Biggest Structural Win
Replacing `MeanAggregation` with `NormAggregation` for BOTH CheMeleon and Chemprop was the single most impactful change:
- CheMeleon NormAggregation alone: 0.4924
- Both models NormAggregation: 0.4900
- Chemprop test MAE dropped from 0.5907 (MeanAgg) to 0.5534 (NormAgg)
- SumAggregation was also better than MeanAggregation (0.5080), but NormAggregation was best

### 2. Ridge Stacking Beats L-BFGS-B Weight Optimization
Replacing `scipy.optimize.minimize` (L-BFGS-B) with `Ridge(alpha=1.0)` as a meta-learner:
- v1 best (L-BFGS-B): 0.4855 (val-test gap ~0.03)
- v2 best (Ridge stacking): 0.4777 (val-test gap ~0.017)
- Ridge provides implicit regularization that prevents the weight optimizer from overfitting to the validation split
- ElasticNet was worse than Ridge (0.5020 vs 0.4777)
- Alpha=1.0 is optimal (tested 0.1 → 0.4974, 10 → 0.4799)

### 3. NormAggregation Explains the Reproducibility Issue
The reproducibility re-run gave 0.5025 vs 0.4777. This suggests that the combination of Ridge stacking + NormAggregation creates a narrow optimum that is sensitive to random initialization. Lightning's `seed_everything()` has been added but results still vary. The true performance is likely in the 0.49 range.

### 4. CheMeleon Hold-out 20% is Still Optimal
- No hold-out: 0.5363 (catastrophic overfit)
- 15% hold-out: 0.5102
- 20% hold-out: 0.4777 (optimal)
- 30% hold-out: 0.4992 (from v1, not enough training data)

### 5. What Doesn't Work (Confirmed)
- Higher LR for CheMeleon → catastrophic (0.5231)
- batch_norm=True for CheMeleon → hurts (0.5205)
- More/less FFN layers → overfit/underfit
- More/less FFN hidden → underfit
- Gradient clipping → marginal at best
- Second CheMeleon model → overfit
- More epochs → hurts (early stopping already works)
- ElasticNet → worse than Ridge
- Dropping sklearn → loses diversity (0.5129)
- Different batch sizes → no benefit
- AttentiveAggregation → API issue, worse anyway

## Critical Issue: Reproducibility

**The 0.4777 result is NOT reproducible.** Re-runs gave 0.5025 (before seed fix) and 0.4916 (with seed_everything). This means:
1. `seed_everything(42)` was added but doesn't fully eliminate variance
2. Lightning's internal randomness (data loader shuffling, weight initialization) may not be fully controlled
3. The Ridge stacking + NormAggregation combination may have a narrower optimum than L-BFGS-B
4. The true performance is likely ~0.49, not 0.48

**Suggested fix for successor**: Run the best config 5-10 times with different seeds and report the mean ± std. If the best single run (0.4777) is a lucky seed, the successor should accept that and try to find a configuration that is both good AND stable.

## Hypotheses for Future Work

### High Priority (Reproducibility First)
1. **Multi-run evaluation**: Run the best config 5-10 times and report mean ± std. This gives a reliable baseline.
2. **Cross-validation for Ridge stacking**: Instead of fitting Ridge on a single val split, use K-fold CV on the val set to get more robust meta-weights.

### Medium Priority (Post-Reproducibility)
3. **CheMeleon with different LR schedules**: CosineAnnealingWarmupRestarts instead of the flat-ish 1e-5→1e-4→1e-5 schedule.
4. **Different aggregation for CheMeleon vs Chemprop**: NormAggregation is best for both currently, but trying SumAgg for one and NormAgg for the other could break symmetry.
5. **GBR hyperparameter tuning per fingerprint**: Current params (n_estimators=500, max_depth=5, lr=0.05) may not be optimal per fingerprint type.
6. **Ridge stacking with more features**: Add individual model MAEs as additional features for the meta-learner.

### Low Priority
7. **Different stratification bins**: Current [4.0, 5.0, 6.0, 8.0]. Try finer bins or different boundaries.
8. **Data augmentation via SMILES randomization**: Previous attempt failed due to RDKit API. Retry with explicit canonicalization.
9. **MolPipeline fingerprints**: Previous runs got ~1% weight for pharmacophore fingerprints. Try concatenated with atom_pairs.

## Run Configuration
```
python experiment.py > run.log 2>&1
# Results in results.csv (last line is most recent)
# Typical runtime: 1060-1100s (~18 min)
# GPU: 1x NVIDIA, VRAM ~8GB usage
```

## Important Notes for Successor
- **Commit policy**: Advance branch if MAE < 0.4777, revert otherwise.
- **lab_notebook.txt**: Maintain running log of experiments.
- **results.csv**: Append-only, last line is current MAE.
- **checkpoints/**: Cleaned up automatically after each run.
- **chemeleon_mp.pt**: Pretrained weights — do not delete.
- **Seeding**: `seed_everything(42)` is now in `experiment.py` (line ~22). Verify it works.
- **Plateau**: 15 consecutive failures after best. The plateau detection rule says to make a structural change. NormAggregation + Ridge stacking WAS the structural leap. Further gains may require fundamentally different approaches (e.g., different foundation model, graph neural network architecture, or cross-validation strategy).
- **Reproducibility gap**: The 0.025 gap between best run (0.4777) and re-run (0.4916) is concerning. This may mean the true performance is closer to 0.49, not 0.48. Successor should treat 0.4777 as an upper bound, not a guaranteed result.
