import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch import seed_everything
from lightning import pytorch as pl

from chemprop import data, models, nn
from chemprop.featurizers import CuikmolmakerMolGraphFeaturizer
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys

warnings.filterwarnings("ignore")
SEED = 42

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_data():
    return pd.read_csv("train.csv", low_memory=False), pd.read_csv("test_phase1.csv")

def prepare_primary_data(train_df):
    primary = train_df[train_df["primary_assay_pEC50"].notna() & train_df["SMILES"].notna() & (train_df["SMILES"] != "")].copy()
    primary["pEC50"] = primary["primary_assay_pEC50"].astype(float)
    primary["std_error"] = primary["primary_assay_pEC50_std.error (-log10(molarity))"].astype(float).replace(0, np.nan).fillna(0.1)
    primary["Emax"] = primary["primary_assay_Emax_estimate (log2FC vs. baseline)"].astype(float)
    primary["Emax_vs_pos"] = primary["primary_assay_Emax.vs.pos.ctrl_estimate (dimensionless)"].astype(float)

    def weighted_agg(group):
        weights = 1.0 / (group["std_error"] ** 2)
        return pd.Series({
            "pEC50": (group["pEC50"] * weights).sum() / weights.sum(),
            "std_error": group["std_error"].min(),
            "Emax": (group["Emax"] * weights).sum() / weights.sum(),
            "Emax_vs_pos": (group["Emax_vs_pos"] * weights).sum() / weights.sum(),
        })

    primary_clean = primary.groupby("SMILES").apply(weighted_agg, include_groups=False).reset_index()

    counter = train_df[train_df["counter_assay_pEC50"].notna() & train_df["SMILES"].notna() & (train_df["SMILES"] != "")].copy()
    if len(counter) > 0:
        counter["counter_pEC50"] = counter["counter_assay_pEC50"].astype(float)
        primary_clean = primary_clean.merge(counter.groupby("SMILES")["counter_pEC50"].mean().reset_index(), on="SMILES", how="left")

    return primary_clean[(primary_clean["pEC50"] >= 1.5) & (primary_clean["pEC50"] <= 8.0)].reset_index(drop=True)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_all_rdkit_features(smis, n_jobs=-1):
    def _get_features(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None: return np.zeros(10420, dtype=np.float32)

        m1 = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048)
        m1c = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048, countSimulation=True)
        m3c = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, countSimulation=True)
        ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)

        fps = [m1.GetFingerprintAsNumPy(mol).astype(np.float32), m1c.GetFingerprintAsNumPy(mol).astype(np.float32),
               m3c.GetFingerprintAsNumPy(mol).astype(np.float32), ap.GetFingerprintAsNumPy(mol).astype(np.float32),
               tt.GetFingerprintAsNumPy(mol).astype(np.float32)]

        maccs_arr = np.zeros(167, dtype=np.float32)
        for bit in MACCSkeys.GenMACCSKeys(mol).GetOnBits(): maccs_arr[bit] = 1.0
        fps.append(maccs_arr)

        physchem_keys = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings", "NumHeteroatoms", "HeavyAtomCount", "RingCount", "LabuteASA", "HeavyAtomMolWt"]
        fps.append(np.array([float(getattr(Descriptors, name)(mol)) for name in physchem_keys], dtype=np.float32))

        return np.concatenate(fps)

    return np.array(Parallel(n_jobs=n_jobs)(delayed(_get_features)(s) for s in smis))

def get_raw_osmordred(smis, osmordred_df):
    """Safely extracts raw arrays. reindex() prevents KeyErrors if test SMILES are missing."""
    features = osmordred_df.set_index("SMILES").reindex(smis).drop(columns=["SMILES"], errors="ignore").values.astype(np.float32)
    return np.where(np.isinf(features), np.nan, features)

# ============================================================================
# MODELING (PYTORCH LIGHTNING / SKLEARN)
# ============================================================================

def build_and_train_pl(train_smis, train_ys, val_smis=None, val_ys=None, model_type='chemprop', 
                       max_epochs=80, patience=12, checkpoint_dir="", n_tasks=1, mode='search'):
    """Unified wrapper handling both 'search' (find best epoch) and 'retrain' (100% data + SWA) modes."""
    train_ys = np.atleast_2d(train_ys)
    if train_ys.shape[0] == 1: train_ys = train_ys.T
    
    featurizer = CuikmolmakerMolGraphFeaturizer()

    train_data = [data.LazyMoleculeDatapoint(s, y=y.tolist()) for s, y in zip(train_smis, train_ys)]
    train_dset = data.CuikmolmakerDataset(train_data, featurizer)
    out_scaler = train_dset.normalize_targets()
    train_loader = data.build_dataloader(train_dset, batch_size=64, num_workers=2, persistent_workers=True)
    out_transform = nn.transforms.UnscaleTransform.from_standard_scaler(out_scaler)

    from chemprop.nn.metrics import MAE
    if model_type == 'chemeleon':
        ckpt = torch.load("chemeleon_mp.pt", weights_only=True)
        mp = nn.BondMessagePassing(**ckpt['hyper_parameters'])
        mp.load_state_dict(ckpt['state_dict'])
        ffn = nn.RegressionFFN(n_tasks=n_tasks, output_transform=out_transform, input_dim=mp.output_dim, hidden_dim=mp.output_dim, n_layers=2, dropout=0.2, criterion=MAE())
        model = models.MPNN(mp, nn.NormAggregation(), ffn, batch_norm=False, init_lr=1e-5, max_lr=1e-4, final_lr=1e-5)
        swa_lr = 1e-5
    else:
        ffn = nn.RegressionFFN(n_tasks=n_tasks, output_transform=out_transform, input_dim=512, hidden_dim=512, n_layers=2, dropout=0.1, criterion=MAE())
        model = models.MPNN(nn.BondMessagePassing(d_h=512, depth=5), nn.NormAggregation(), ffn, batch_norm=True, init_lr=1e-4, max_lr=5e-4, final_lr=1e-4)
        swa_lr = 1e-4

    if mode == 'search':
        val_ys = np.atleast_2d(val_ys)
        if val_ys.shape[0] == 1: val_ys = val_ys.T
        val_data = [data.LazyMoleculeDatapoint(s, y=y.tolist()) for s, y in zip(val_smis, val_ys)]
        val_dset = data.CuikmolmakerDataset(val_data, featurizer)
        val_dset.normalize_targets(out_scaler)
        val_loader = data.build_dataloader(val_dset, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)

        ckpt_cb = ModelCheckpoint(dirpath=f"checkpoints/{checkpoint_dir}", filename="best-epoch={epoch:02d}-val_loss={val_loss:.4f}", monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, callbacks=[ckpt_cb, EarlyStopping(monitor="val_loss", patience=patience, mode="min")], logger=False, enable_progress_bar=False, enable_model_summary=False, deterministic=True)
        trainer.fit(model, train_loader, val_loader)
        
        preds = torch.cat(trainer.predict(trainer.lightning_module, val_loader, ckpt_path=ckpt_cb.best_model_path), dim=0).cpu().numpy()
        preds = preds.ravel() if n_tasks == 1 else preds
        
        best_epoch = int(re.search(r'epoch=(\d+)', ckpt_cb.best_model_path).group(1))
        return preds, best_epoch

    elif mode == 'retrain':
        swa_cb = StochasticWeightAveraging(swa_lrs=swa_lr)
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, callbacks=[swa_cb], logger=False, enable_progress_bar=False, enable_model_summary=False, deterministic=True)
        trainer.fit(model, train_loader)
        
        return trainer, model

def predict_pl_model(trainer, model, test_smiles, n_tasks=1):
    featurizer = CuikmolmakerMolGraphFeaturizer()
    dset = data.CuikmolmakerDataset([data.LazyMoleculeDatapoint(s) for s in test_smiles], featurizer)
    loader = data.build_dataloader(dset, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)
    preds = torch.cat(trainer.predict(model, loader, ckpt_path=None), dim=0).cpu().numpy() 
    return preds.ravel() if n_tasks == 1 else preds

def train_sklearn_model(X_train, y_train, X_val, y_val):
    """Uses a pipeline to safeguard against NaNs during featurization."""
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'), 
        StandardScaler(), 
        XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=SEED, tree_method="hist", device="cuda", objective='reg:absoluteerror')
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)
    return pipeline, pred, mean_absolute_error(y_val, pred)

def optimize_ensemble_weights(val_preds_dict, y_val):
    names = sorted(val_preds_dict.keys())
    X_scaled = StandardScaler().fit_transform(np.column_stack([val_preds_dict[name] for name in names]))
    meta = LinearRegression(positive=True).fit(X_scaled, y_val)
    weights = np.abs(meta.coef_) / np.sum(np.abs(meta.coef_)) if np.sum(np.abs(meta.coef_)) > 0 else np.ones(len(names)) / len(names)
    return dict(zip(names, weights)), mean_absolute_error(y_val, meta.predict(X_scaled))

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_data(train, test=None):
    print("Preparing primary assay data...")
    primary_clean = prepare_primary_data(train)
    print(f"  Primary data: {len(primary_clean)} unique compounds")
    return primary_clean, test

def train_model(processed_data):
    primary_clean, test = processed_data
    os.makedirs("checkpoints/chemprop_mt", exist_ok=True)

    smis, y_pEC50, y_Emax = primary_clean["SMILES"].values, primary_clean["pEC50"].values, primary_clean["Emax"].values
    train_idx, val_idx = train_test_split(np.arange(len(smis)), test_size=0.2, random_state=SEED, stratify=np.digitize(y_pEC50, bins=[4.0, 5.0, 6.0, 8.0]))
    train_smis, val_smis = smis[train_idx], smis[val_idx]
    train_pec50, val_pec50 = y_pEC50[train_idx], y_pEC50[val_idx]

    osmordred_train = pd.read_parquet("train_osmordred_features.parquet").groupby("SMILES", as_index=False).mean(numeric_only=True)

    # -------------------------------------------------------------------------
    # Phase 1: Search (Find Best Epochs and Weights on 80/20 Split)
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Search Phase (80/20 Split) ---")
    
    print("\n  Chemprop Multi-Task Search...")
    mt_preds, mt_best_epoch = build_and_train_pl(
        train_smis, np.column_stack([train_pec50, y_Emax[train_idx]]), 
        val_smis, np.column_stack([val_pec50, y_Emax[val_idx]]), 
        'chemprop', max_epochs=80, patience=12, checkpoint_dir="chemprop_mt", n_tasks=2, mode='search'
    )
    print(f"    MAE (pEC50): {mean_absolute_error(val_pec50, mt_preds[:, 0]):.4f} | Best Epoch: {mt_best_epoch}")

    print("\n  CheMeleon Foundation Search...")
    ch_preds, ch_best_epoch = build_and_train_pl(
        train_smis, train_pec50, val_smis, val_pec50, 
        'chemeleon', max_epochs=80, patience=25, checkpoint_dir="chemeleon", n_tasks=1, mode='search'
    )
    print(f"    MAE (pEC50): {mean_absolute_error(val_pec50, ch_preds):.4f} | Best Epoch: {ch_best_epoch}")

    print("\n  Sklearn (fingerprints)...")
    X_train_fp, X_val_fp = compute_all_rdkit_features(train_smis), compute_all_rdkit_features(val_smis)
    sk_trained, sk_preds_dict = {}, {}
    sk_trained["concat"], sk_preds_dict["sklearn_concat"], sk_m_fp = train_sklearn_model(X_train_fp, train_pec50, X_val_fp, val_pec50)
    print(f"    Sklearn concat MAE: {sk_m_fp:.4f}")

    print("\n  Ridge (osmordred)...")
    X_train_osm, X_val_osm = get_raw_osmordred(train_smis, osmordred_train), get_raw_osmordred(val_smis, osmordred_train)
    
    best_ridge_m, best_alpha, best_ridge_pr = 999, 10.0, None
    for alpha in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        ridge_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), Ridge(alpha=alpha, random_state=SEED))
        ridge_pipe.fit(X_train_osm, train_pec50)
        pr = ridge_pipe.predict(X_val_osm)
        
        if (m := mean_absolute_error(val_pec50, pr)) < best_ridge_m:
            best_ridge_m, best_alpha, best_ridge_pr = m, alpha, pr
            
    sk_preds_dict["ridge_osmordred"] = best_ridge_pr
    print(f"    Ridge osmordred MAE: {best_ridge_m:.4f} (alpha={best_alpha})")

    print("\n--- Optimizing Ensemble Weights ---")
    all_val_preds = {"chemprop_mt": mt_preds[:, 0], "chemeleon": ch_preds, **sk_preds_dict}
    ensemble_weights, ensemble_mae = optimize_ensemble_weights(all_val_preds, val_pec50)
    print(f"  Ensemble val MAE: {ensemble_mae:.4f}\n  Weights:")
    for name, w in sorted(ensemble_weights.items(), key=lambda x: -x[1]): print(f"    {name}: {w:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2: Retrain (100% Data, calculated epochs + 10%, SWA)
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Retrain Phase (100% Data + SWA) ---")
    
    target_mt_epochs = max(int(mt_best_epoch * 1.1), 1)
    print(f"  Chemprop MT (Retraining blindly for {target_mt_epochs} epochs with SWA)...")
    final_mt_trainer, final_mt_model = build_and_train_pl(
        smis, np.column_stack([y_pEC50, y_Emax]), 
        model_type='chemprop', max_epochs=target_mt_epochs, n_tasks=2, mode='retrain'
    )

    target_ch_epochs = max(int(ch_best_epoch * 1.1), 1)
    print(f"  CheMeleon Foundation (Retraining blindly for {target_ch_epochs} epochs with SWA)...")
    final_ch_trainer, final_ch_model = build_and_train_pl(
        smis, y_pEC50, 
        model_type='chemeleon', max_epochs=target_ch_epochs, n_tasks=1, mode='retrain'
    )

    sklearn_finals, X_full_fp = {}, compute_all_rdkit_features(smis)
    for name, pipeline_orig in sk_trained.items():
        print(f"  Sklearn {name} GB (Retraining on 100% data)...")
        xgb_orig = pipeline_orig.named_steps['xgbregressor']
        pipeline_new = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(), 
            type(xgb_orig)(**xgb_orig.get_params())
        )
        pipeline_new.fit(X_full_fp, y_pEC50)
        sklearn_finals[name] = pipeline_new

    print("  Ridge osmordred GB (Retraining on 100% data)...")
    ridge_final = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), Ridge(alpha=best_alpha, random_state=SEED))
    ridge_final.fit(get_raw_osmordred(smis, osmordred_train), y_pEC50)

    osmordred_test = pd.read_parquet("test_phase1_osmordred_features.parquet").groupby("SMILES", as_index=False).mean(numeric_only=True)

    return {
        "chemprop_mt_trainer": final_mt_trainer, "chemprop_mt_model": final_mt_model,
        "chemeleon_trainer": final_ch_trainer, "chemeleon_model": final_ch_model,
        "sklearn_finals": sklearn_finals, "ridge_final": ridge_final,
        "osmordred_test": osmordred_test, "ensemble_weights": ensemble_weights, "ensemble_mae": ensemble_mae,
        "test": test,
    }

def evaluate_model(model, test=None):
    if test is None: test = model["test"]
    test_smiles = test["SMILES"].values
    print("\n--- Generating Predictions ---")

    chemprop_mt_pred = predict_pl_model(model["chemprop_mt_trainer"], model["chemprop_mt_model"], test_smiles, n_tasks=2)[:, 0]
    ch_preds = predict_pl_model(model["chemeleon_trainer"], model["chemeleon_model"], test_smiles, n_tasks=1)
    print(f"  CheMeleon: [{ch_preds.min():.2f}, {ch_preds.max():.2f}]")

    sk_test_preds, X_test_fp = {}, compute_all_rdkit_features(test_smiles)
    for name, pipeline in model["sklearn_finals"].items():
        sk_test_preds[f"sklearn_{name}"] = pipeline.predict(X_test_fp)

    all_preds = {
        "chemprop_mt": chemprop_mt_pred, 
        "chemeleon": ch_preds, 
        "ridge_osmordred": model["ridge_final"].predict(get_raw_osmordred(test_smiles, model["osmordred_test"])),
        **sk_test_preds
    }
    
    final_pred = np.zeros(len(test_smiles))
    total_weight = 0
    for name, pred in all_preds.items():
        weight = model["ensemble_weights"].get(name, 0)
        final_pred += weight * pred
        total_weight += weight
        
    final_pred = (final_pred / total_weight) if total_weight > 0 else np.mean(list(all_preds.values()), axis=0)
    
    # Clip predictions to plausible target range
    final_pred = np.clip(final_pred, 1.5, 8.0)

    print(f"  Chemprop MT: [{chemprop_mt_pred.min():.2f}, {chemprop_mt_pred.max():.2f}]")
    for name, p in sorted(sk_test_preds.items()): print(f"  {name}:   [{p.min():.2f}, {p.max():.2f}]")
    print(f"  Ensemble:    [{final_pred.min():.2f}, {final_pred.max():.2f}]")

    if "pEC50" in test.columns:
        mae = mean_absolute_error(test["pEC50"].values, final_pred)
        print(f"\n  Final MAE on test: {mae:.4f}\n  Individual model MAEs:")
        for name, pred in sorted(all_preds.items()): print(f"    {name}: {mean_absolute_error(test['pEC50'].values, pred):.4f}")
    else:
        mae = model["ensemble_mae"]
        print(f"\n  No ground truth - using val MAE: {mae:.4f}")

    results = pd.DataFrame({"Molecule Name": test["Molecule Name"].values, "SMILES": test_smiles, "Chemprop_MT": chemprop_mt_pred})
    for name, pred in sorted(sk_test_preds.items()): results[name] = pred
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
        with open(outfile, "w") as f: f.write("Timestamp,MAE,Execution Time\n")

    start_time = datetime.now()
    try:
        raw_data = load_data()
        mae = f"{evaluate_model(train_model(preprocess_data(*raw_data))):.4f}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        mae = f"\"CRASH ({e})\""
    finally:
        if os.path.exists("checkpoints"):
            print("\nCleaning up checkpoint files...")
            shutil.rmtree("checkpoints")

    elapsed = (datetime.now() - start_time).total_seconds()
    with open(outfile, "a") as f: f.write(f"{now},{mae},{elapsed:.1f}s\n")
    print(f"\nTotal time: {elapsed:.1f}s\nResult logged to results.csv")
