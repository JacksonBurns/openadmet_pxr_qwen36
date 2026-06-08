import pandas as pd

if __name__ == "__main__":
    # read each predicton file, rename the columns, write
    for dset in ("test", "test_phase1", "train"):
        df = pd.read_csv(f"{dset}_chemeleon_pampa_preds_individual.csv")
        df = df.rename(columns={"Y_model_0": "model_0_pampa_prob", "Y_model_1": "model_1_pampa_prob", "Y_model_2": "model_2_pampa_prob", "Y_model_3": "model_3_pampa_prob"})
        df[["SMILES", "model_0_pampa_prob", "model_1_pampa_prob", "model_2_pampa_prob", "model_3_pampa_prob"]].to_csv(f"{dset}_chemeleon_pampa_preds_individual.csv", index=False)
    