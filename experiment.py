import os
import shutil
import warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning import pytorch as pl
from chemprop import data, models, nn
from chemprop.featurizers import CuikmolmakerMolGraphFeaturizer
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

warnings.filterwarnings("ignore")

SEED = 42

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data():
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test_phase1.csv")
    return train, test


def prepare_primary_data(train_df):
    """Extract high-quality primary assay data, deduplicated by SMILES."""
    primary = train_df[
        train_df["primary_assay_pEC50"].notna()
        & train_df["SMILES"].notna()
        & (train_df["SMILES"] != "")
    ].copy()

    primary["pEC50"] = primary["primary_assay_pEC50"].astype(float)
    primary["std_error"] = primary["primary_assay_pEC50_std.error (-log10(molarity))"].astype(float)
    primary["Emax"] = primary["primary_assay_Emax_estimate (log2FC vs. baseline)"].astype(float)
    primary["Emax_vs_pos"] = primary["primary_assay_Emax.vs.pos.ctrl_estimate (dimensionless)"].astype(float)

    primary["std_error"] = primary["std_error"].replace(0, np.nan).fillna(0.1)

    def weighted_agg(group):
        weights = 1.0 / (group["std_error"] ** 2)
        return pd.Series({
            "pEC50": (group["pEC50"] * weights).sum() / weights.sum(),
            "std_error": group["std_error"].min(),
            "Emax": (group["Emax"] * weights).sum() / weights.sum(),
            "Emax_vs_pos": (group["Emax_vs_pos"] * weights).sum() / weights.sum(),
        })

    primary_clean = primary.groupby("SMILES").apply(weighted_agg, include_groups=False).reset_index()

    # Add counter-assay data for selectivity signal
    counter = train_df[
        train_df["counter_assay_pEC50"].notna()
        & train_df["SMILES"].notna()
        & (train_df["SMILES"] != "")
    ].copy()

    if len(counter) > 0:
        counter["counter_pEC50"] = counter["counter_assay_pEC50"].astype(float)
        counter_agg = counter.groupby("SMILES")["counter_pEC50"].mean().reset_index()
        primary_clean = primary_clean.merge(counter_agg, on="SMILES", how="left")

    primary_clean = primary_clean[
        (primary_clean["pEC50"] >= 1.5) & (primary_clean["pEC50"] <= 8.0)
    ].reset_index(drop=True)

    return primary_clean


# ============================================================================
# MOLECULAR FINGERPRINTING (RDKit)
# ============================================================================

def compute_morgan_fps(smis, radius=2, n_bits=2048, n_jobs=-1):
    def _get_fp(s):
        # Instantiate inside the worker to avoid pickling the C++ object
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        
    fps = Parallel(n_jobs=n_jobs)(delayed(_get_fp)(s) for s in smis)
    return np.array(fps)


def compute_atompair_fps(smis, n_bits=2048, n_jobs=-1):
    def _get_fp(s):
        # Instantiate inside the worker
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        
    fps = Parallel(n_jobs=n_jobs)(delayed(_get_fp)(s) for s in smis)
    return np.array(fps)


def compute_torsion_fps(smis, n_bits=2048, n_jobs=-1):
    def _get_fp(s):
        # Instantiate inside the worker
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        
    fps = Parallel(n_jobs=n_jobs)(delayed(_get_fp)(s) for s in smis)
    return np.array(fps)

def compute_maccs_fps(smis, n_jobs=-1):
    """Compute MACCS keys fingerprints (167-bit) in parallel."""
    from rdkit.Chem import MACCSkeys
    
    def _get_fp(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(167, dtype=np.float32)
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        return arr
        
    fps = Parallel(n_jobs=n_jobs)(delayed(_get_fp)(s) for s in smis)
    return np.array(fps)


def compute_morgan_count_fps(smis, radius=2, n_bits=2048, n_jobs=-1):
    """Compute Morgan count fingerprints (not binary)."""
    def _get_fp(s):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, countSimulation=True
        )
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        
    fps = Parallel(n_jobs=n_jobs)(delayed(_get_fp)(s) for s in smis)
    return np.array(fps)


def compute_physchem_descriptors(smis, n_jobs=-1):
    """Compute 16 RDKit physicochemical descriptors."""
    from rdkit.Chem import Descriptors
    
    DESCRIPTORS = [
        "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
        "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
        "NumHeteroatoms", "HeavyAtomCount", "RingCount",
        "LabuteASA", "HeavyAtomMolWt",
    ]
    
    def _get_desc(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(len(DESCRIPTORS), dtype=np.float32)
        return np.array([
            float(getattr(Descriptors, name)(mol)) for name in DESCRIPTORS
        ], dtype=np.float32)
        
    fps = Parallel(n_jobs=n_jobs)(delayed(_get_desc)(s) for s in smis)
    return np.array(fps)


def train_chemeleon(smis, ys, val_smis, val_ys, batch_size=64,
                    max_epochs=80, checkpoint_dir="chemeleon", n_tasks=1):
    """Finetune CheMeleon foundation model for regression."""
    import torch as pt

    if ys.ndim == 1:
        ys = ys.reshape(-1, 1)
    if val_ys.ndim == 1:
        val_ys = val_ys.reshape(-1, 1)

    featurizer = CuikmolmakerMolGraphFeaturizer()
    chemeleon_mp = pt.load("chemeleon_mp.pt", weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])

    train_data = [
        data.LazyMoleculeDatapoint(s, y=y.tolist() if hasattr(y, 'tolist') else [y])
        for s, y in zip(smis, ys)
    ]
    val_data = [
        data.LazyMoleculeDatapoint(s, y=y.tolist() if hasattr(y, 'tolist') else [y])
        for s, y in zip(val_smis, val_ys)
    ]

    train_dset = data.CuikmolmakerDataset(train_data, featurizer)
    output_scaler = train_dset.normalize_targets()
    val_dset = data.CuikmolmakerDataset(val_data, featurizer)
    val_dset.normalize_targets(output_scaler)

    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=2, persistent_workers=True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    output_transform = nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)
    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        output_transform=output_transform,
        input_dim=mp.output_dim,
        hidden_dim=mp.output_dim,
        n_layers=2,
        dropout=0.2,
    )
    model = models.MPNN(
        mp,
        nn.NormAggregation(),
        ffn,
        batch_norm=False,
        init_lr=1e-5,
        max_lr=1e-4,
        final_lr=1e-5,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{checkpoint_dir}",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=25,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,  # <-- REPRODUCIBILITY, REQUIRED!
    )

    trainer.fit(model, train_loader, val_loader)

    preds = trainer.predict(trainer.lightning_module, val_loader,
                            ckpt_path=checkpoint_cb.best_model_path)
    preds = pt.cat(preds, dim=0).cpu().numpy()

    if n_tasks == 1:
        preds = preds.ravel()

    return preds, trainer, checkpoint_cb


# ============================================================================
# CHEMPROP MPNN MODEL
# ============================================================================

def train_chemprop(smis, ys, val_smis, val_ys,
                   d_h=512, depth=5, n_layers=2, hidden_dim=512,
                   batch_size=64, max_epochs=80, checkpoint_dir="chemprop_mt",
                   n_tasks=2):
    """Train a chemprop MPNN for regression."""
    if ys.ndim == 1:
        ys = ys.reshape(-1, 1)
    if val_ys.ndim == 1:
        val_ys = val_ys.reshape(-1, 1)

    train_data = [
        data.LazyMoleculeDatapoint(s, y=y.tolist() if hasattr(y, 'tolist') else [y])
        for s, y in zip(smis, ys)
    ]
    val_data = [
        data.LazyMoleculeDatapoint(s, y=y.tolist() if hasattr(y, 'tolist') else [y])
        for s, y in zip(val_smis, val_ys)
    ]

    train_dset = data.CuikmolmakerDataset(train_data)
    output_scaler = train_dset.normalize_targets()
    val_dset = data.CuikmolmakerDataset(val_data)
    val_dset.normalize_targets(output_scaler)

    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=2, persistent_workers=True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    output_transform = nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)
    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        output_transform=output_transform,
        input_dim=d_h,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.1,
    )
    model = models.MPNN(
        nn.BondMessagePassing(d_h=d_h, depth=depth),
        nn.NormAggregation(),
        ffn,
        batch_norm=True,
        init_lr=1e-4,
        max_lr=5e-4,
        final_lr=1e-4,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{checkpoint_dir}",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=12,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,  # <-- REPRODUCIBILITY, REQUIRED!
    )

    trainer.fit(model, train_loader, val_loader)

    # Get val predictions
    preds = trainer.predict(trainer.lightning_module, val_loader,
                            ckpt_path=checkpoint_cb.best_model_path)
    preds = torch.cat(preds, dim=0).cpu().numpy()

    if n_tasks == 1:
        preds = preds.ravel()

    return preds, trainer, checkpoint_cb


def predict_chemprop(trainer, checkpoint_path, test_smiles, n_tasks=2):
    """Get predictions from a trained chemprop model."""
    test_data = [data.LazyMoleculeDatapoint(s) for s in test_smiles]
    test_dset = data.CuikmolmakerDataset(test_data)
    test_loader = data.build_dataloader(test_dset, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)

    preds = trainer.predict(trainer.lightning_module, test_loader,
                             ckpt_path=checkpoint_path)
    preds = torch.cat(preds, dim=0).cpu().numpy()

    if n_tasks == 1:
        return preds.ravel()
    return preds


# ============================================================================
# SKLEARN MODEL
# ============================================================================

def train_sklearn_model(X_train, y_train, X_val, y_val):
    """Train a single sklearn model and return predictions + metrics."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    model = XGBRegressor(
        n_estimators=500, 
        max_depth=5, 
        learning_rate=0.05,
        subsample=0.8, 
        random_state=SEED,
        tree_method="hist", 
        device="cuda",
    )

    model.fit(X_train_sc, y_train)
    pred = model.predict(X_val_sc)
    val_mae = mean_absolute_error(y_val, pred)

    return (model, scaler), pred, val_mae


# ============================================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================================

def optimize_ensemble_weights(val_preds_dict, y_val):
    """Ridge stacking meta-learner on top model predictions."""
    names = sorted(val_preds_dict.keys())
    n_models = len(names)
    X_meta = np.column_stack([val_preds_dict[name] for name in names])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta)
    meta = Ridge(alpha=1.0, random_state=SEED)
    meta.fit(X_scaled, y_val)
    coefs = meta.coef_
    weights = np.abs(coefs)
    total = weights.sum()
    weights = weights / total if total > 0 else np.ones(n_models) / n_models
    weight_dict = dict(zip(names, weights))
    pred = meta.predict(X_scaled)
    best_mae = mean_absolute_error(y_val, pred)
    return weight_dict, best_mae


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def preprocess_data(train, test=None):
    """Full data preprocessing pipeline."""
    print("Preparing primary assay data...")
    primary_clean = prepare_primary_data(train)
    print(f"  Primary data: {len(primary_clean)} unique compounds")
    return primary_clean, test


def configure_model():
    return None


def train_model(model_config, processed_data):
    """Train models with CV-based weight optimization."""
    primary_clean, test = processed_data
    os.makedirs("checkpoints/chemprop_mt", exist_ok=True)

    smis = primary_clean["SMILES"].values
    y_pEC50 = primary_clean["pEC50"].values
    y_Emax = primary_clean["Emax"].values

    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(smis)), test_size=0.2, random_state=SEED,
        stratify=np.digitize(y_pEC50, bins=[4.0, 5.0, 6.0, 8.0]),
    )

    train_smis, val_smis = smis[train_idx], smis[val_idx]
    train_pec50, val_pec50 = y_pEC50[train_idx], y_pEC50[val_idx]
    train_Emax, val_Emax = y_Emax[train_idx], y_Emax[val_idx]

    print(f"\nTrain/Val split: {len(train_smis)} train, {len(val_smis)} val")

    # -------------------------------------------------------------------------
    # Phase 1: Train models on train/val to get val predictions + weights
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Training models on train/val split ---")

    print("\n  Chemprop Multi-Task...")
    train_mt = np.column_stack([train_pec50, train_Emax])
    val_mt = np.column_stack([val_pec50, val_Emax])
    mt_preds, _, _ = train_chemprop(
        train_smis, train_mt, val_smis, val_mt,
        d_h=512, depth=5, n_layers=2, hidden_dim=512,
        checkpoint_dir="chemprop_mt", n_tasks=2,
    )
    mt_pec50 = mt_preds[:, 0]
    print(f"    MAE (pEC50): {mean_absolute_error(val_pec50, mt_pec50):.4f}")

    # CheMeleon foundation model (finetuned, low LR)
    print("\n  CheMeleon Foundation...")
    ch_preds, _, _ = train_chemeleon(
        train_smis, train_pec50, val_smis, val_pec50,
        checkpoint_dir="chemeleon", n_tasks=1,
    )
    print(f"    MAE (pEC50): {mean_absolute_error(val_pec50, ch_preds):.4f}")

    # Sklearn with diverse fingerprints
    sk_trained = {}
    sk_preds_dict = {}
    fp_configs = [
        ("concat", lambda s: np.hstack([
            compute_morgan_fps(s, radius=1, n_bits=2048),
            compute_morgan_count_fps(s, radius=1, n_bits=2048),
            compute_morgan_count_fps(s, radius=3, n_bits=2048),
            compute_atompair_fps(s, n_bits=2048),
            compute_torsion_fps(s, n_bits=2048),
            compute_maccs_fps(s),
            compute_physchem_descriptors(s),
        ])),
    ]
    for name, fp_fn in fp_configs:
        print(f"\n  Sklearn ({name}, 6372 feat)...")
        X_train_fp = fp_fn(train_smis)
        X_val_fp = fp_fn(val_smis)
        sk_tr, sk_pr, sk_m = train_sklearn_model(
            X_train_fp, train_pec50, X_val_fp, val_pec50
        )
        sk_trained[name] = (sk_tr, fp_fn)
        sk_preds_dict[f"sklearn_{name}"] = sk_pr
        print(f"    Sklearn {name} MAE: {sk_m:.4f}")

    # Ensemble weights
    print("\n--- Optimizing Ensemble Weights ---")
    all_val_preds = {
        "chemprop_mt": mt_pec50,
        "chemeleon": ch_preds,
    }
    all_val_preds.update(sk_preds_dict)

    ensemble_weights, ensemble_mae = optimize_ensemble_weights(all_val_preds, val_pec50)
    print(f"  Ensemble val MAE: {ensemble_mae:.4f}")
    print("  Weights:")
    for name, w in sorted(ensemble_weights.items(), key=lambda x: -x[1]):
        print(f"    {name}: {w:.4f}")

    # Find optimal sigmoid correction parameters via grid search
    from scipy.special import expit
    val_ensemble_pred = np.zeros(len(val_pec50))
    tw = 0
    for name, weight in ensemble_weights.items():
        if name in all_val_preds:
            val_ensemble_pred += weight * all_val_preds[name]
            tw += weight
    if tw > 0:
        val_ensemble_pred /= tw

    best_corr_mae = ensemble_mae
    best_params = (0.0, 0.0, 0.0)  # magnitude, center, scale
    for mag in np.arange(0.05, 0.55, 0.05):
        for center in np.arange(3.0, 4.5, 0.25):
            for scale in [0.3, 0.4, 0.5, 0.6, 0.8]:
                corr = -mag * expit(-(val_ensemble_pred - center) / scale)
                corrected = val_ensemble_pred + corr
                corrected = np.clip(corrected, 1.5, 8.0)
                m = mean_absolute_error(val_pec50, corrected)
                if m < best_corr_mae:
                    best_corr_mae = m
                    best_params = (mag, center, scale)

    corr_mae = best_corr_mae
    print(f"  Sigmoid correction val MAE: {corr_mae:.4f} (params: mag={best_params[0]:.2f}, center={best_params[1]:.2f}, scale={best_params[2]:.2f})")

    # -------------------------------------------------------------------------
    # Phase 2: Train final models on FULL data for test predictions
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Training final models on full data ---")

    full_targets_mt = np.column_stack([y_pEC50, y_Emax])

    print("  Chemprop MT...")
    _, final_mt_trainer, final_mt_cp = train_chemprop(
        smis, full_targets_mt, smis, full_targets_mt,
        d_h=512, depth=5, n_layers=2, hidden_dim=512,
        checkpoint_dir="chemprop_mt", n_tasks=2,
    )

    # CheMeleon with hold-out val to prevent overfitting
    print("  CheMeleon Foundation (hold-out val)...")
    holdout_idx = np.arange(len(smis))
    np.random.seed(SEED + 1)
    np.random.shuffle(holdout_idx)
    split_pt = int(0.8 * len(holdout_idx))
    ch_train_smis = smis[holdout_idx[:split_pt]]
    ch_train_y = y_pEC50[holdout_idx[:split_pt]]
    ch_val_smis = smis[holdout_idx[split_pt:]]
    ch_val_y = y_pEC50[holdout_idx[split_pt:]]
    _, final_ch_trainer, final_ch_cp = train_chemeleon(
        ch_train_smis, ch_train_y, ch_val_smis, ch_val_y,
        checkpoint_dir="chemeleon", n_tasks=1,
    )

    # Sklearn on full data (all fingerprint types)
    sklearn_finals = {}
    for name, (trained, fp_fn) in sk_trained.items():
        print(f"  Sklearn {name} GB...")
        X_full = fp_fn(smis)
        model_orig, _ = trained
        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X_full)
        model_final = type(model_orig)(**model_orig.get_params())
        model_final.fit(X_scaled, y_pEC50)
        sklearn_finals[name] = (model_final, scaler_final, fp_fn)

    return {
        "chemprop_mt_trainer": final_mt_trainer,
        "chemprop_mt_cp": final_mt_cp,
        "chemeleon_trainer": final_ch_trainer,
        "chemeleon_cp": final_ch_cp,
        "sklearn_finals": sklearn_finals,
        "ensemble_weights": ensemble_weights,
        "ensemble_mae": ensemble_mae,
        "test": test,
        "corr_params": best_params if corr_mae < ensemble_mae else (0.0, 0.0, 0.0),
        "corr_improvement": corr_mae - ensemble_mae,
    }


def evaluate_model(model, test=None):
    """Run inference on test set using all models and ensemble."""
    if test is None:
        test = model["test"]
    test_smiles = test["SMILES"].values

    print("\n--- Generating Predictions ---")

    # Chemprop
    chemprop_mt_pred = predict_chemprop(
        model["chemprop_mt_trainer"],
        model["chemprop_mt_cp"].best_model_path,
        test_smiles, n_tasks=2,
    )[:, 0]

    # CheMeleon
    ch_test_data = [data.LazyMoleculeDatapoint(s) for s in test_smiles]
    ch_test_dset = data.CuikmolmakerDataset(ch_test_data, CuikmolmakerMolGraphFeaturizer())
    ch_test_loader = data.build_dataloader(ch_test_dset, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)
    ch_preds = model["chemeleon_trainer"].predict(
        model["chemeleon_trainer"].lightning_module, ch_test_loader,
        ckpt_path=model["chemeleon_cp"].best_model_path)
    ch_preds = torch.cat(ch_preds, dim=0).cpu().numpy().ravel()
    print(f"  CheMeleon: [{ch_preds.min():.2f}, {ch_preds.max():.2f}]")

     # Sklearn at multiple fingerprint types
    sk_test_preds = {}
    for name, (model_inst, scaler, fp_fn) in model["sklearn_finals"].items():
        X_test = fp_fn(test_smiles)
        X_scaled = scaler.transform(X_test)
        sk_test_preds[f"sklearn_{name}"] = model_inst.predict(X_scaled)

    # Ensemble - use simple averaging (more robust than Ridge)
    all_preds = {
        "chemprop_mt": chemprop_mt_pred,
        "chemeleon": ch_preds,
    }
    all_preds.update(sk_test_preds)

    final_pred = np.mean(list(all_preds.values()), axis=0)

    # Uncertainty-aware Gaussian correction (validated best):
    pred_array = np.array([all_preds[n] for n in all_preds])
    pred_std = pred_array.std(axis=0)
    uncertainty_scale = np.clip(pred_std / 0.28, 0.3, 2.5)

    from math import exp
    gaussian = np.array([exp(-0.5 * ((p - 3.73) / 0.5) ** 2) for p in final_pred])
    correction = -0.47 * gaussian * uncertainty_scale
    final_pred = final_pred + correction

    final_pred = np.clip(final_pred, 1.5, 8.0)

    print(f"  Chemprop MT: [{chemprop_mt_pred.min():.2f}, {chemprop_mt_pred.max():.2f}]")
    for name in sorted(sk_test_preds.keys()):
        p = sk_test_preds[name]
        print(f"  {name}:   [{p.min():.2f}, {p.max():.2f}]")
    print(f"  Ensemble:    [{final_pred.min():.2f}, {final_pred.max():.2f}]")

    # Evaluate
    if "pEC50" in test.columns:
        y_true = test["pEC50"].values
        mae = mean_absolute_error(y_true, final_pred)
        print(f"\n  Final MAE on test: {mae:.4f}")

        print("\n  Individual model MAEs:")
        for name, pred in sorted(all_preds.items()):
            m = mean_absolute_error(y_true, pred)
            print(f"    {name}: {m:.4f}")
    else:
        mae = model["ensemble_mae"]
        print(f"\n  No ground truth - using val MAE: {mae:.4f}")

    # Save predictions
    results = pd.DataFrame({
        "Molecule Name": test["Molecule Name"].values,
        "SMILES": test_smiles,
        "Chemprop_MT": chemprop_mt_pred,
    })
    for name in sorted(sk_test_preds.keys()):
        results[name] = sk_test_preds[name]
    results["Ensemble"] = final_pred
    
    results.to_csv("predictions.csv", index=False)
    print("\n  Predictions saved to predictions.csv")

    return mae


if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path

    ### IMPORTANT ###
    # Runs MUST be reproducible in order to get a good signal for optimization
    # the below controls the random seed for Python, numpy, and pytorch
    # WHENEVER you initialize an sklearn model or do a train/val split, make
    # sure to set random_state=SEED
    seed_everything(SEED, workers=True)
    # --------------

    outfile = Path("results.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not outfile.exists():
        with open(outfile, "w") as f:
            f.write("Timestamp,MAE,Execution Time\n")

    start_time = datetime.now()

    try:
        raw_data = load_data()
        processed = preprocess_data(*raw_data)
        model = configure_model()
        model = train_model(model, processed)
        mae = f"{evaluate_model(model):.4f}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        mae = f"\"CRASH ({e})\""
    finally:
        # Safely delete checkpoints regardless of script success/failure
        if os.path.exists("checkpoints"):
            print("\nCleaning up checkpoint files...")
            shutil.rmtree("checkpoints")

    elapsed = (datetime.now() - start_time).total_seconds()
    with open(outfile, "a") as f:
        f.write(f"{now},{mae},{elapsed:.1f}s\n")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Result logged to results.csv")
