if __name__ == "__main__":
    # download the data
    import pandas as pd

    train          = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
    test           = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")
    train_counter  = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv")
    train_single   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")
    train_crudes   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_htchem-libraries_TRAIN.csv")
    train_semi_pure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
    test_phase1    = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")

    test.to_csv("test.csv", index=False)
    test_phase1.to_csv("test_phase1.csv", index=False)

    # combine all the other dataframe into one for training, matching SMILES when they overlap
    from functools import reduce

    named_dfs = {
        "primary_assay": train,
        "counter_assay": train_counter,
        "single_concentration": train_single,
        "crudes": train_crudes,
        "semi_pure": train_semi_pure,
    }

    renamed_dfs = []

    for name, df in named_dfs.items():
        renamed = df.rename(
            columns={
                c: f"{name}_{c}"
                for c in df.columns
                if c != "SMILES"
            }
        )
        renamed_dfs.append(renamed)

    merged = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on="SMILES",
            how="outer"
        ),
        renamed_dfs
    )

    merged.sample(frac=0.1, random_state=42).to_csv("train_sample.csv", index=False)
    merged.to_csv("train_full.csv", index=False)
