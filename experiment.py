import pandas as pd


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_phase1.csv")

    return train, test

def preprocess_data(train, test):
    # do some preprocessing here
    return train, test

def configure_model():
    # configure the model here
    return None

def train_model(model, train):
    # train the model here
    return model

def evaluate_model(model, test):
    # run inference
    predictions = ...
    # calculate the MAE
    mae = test["pEC50"] - predictions
    mae = mae.abs().mean()
    return mae

if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path

    outfile = Path("results.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not outfile.exists():
        with open(outfile, "w") as f:
            f.write("Timestamp,MAE\n")

    start_time = datetime.now()

    try:
        data = load_data()
        train, test = preprocess_data(*data)
        model = configure_model()
        model = train_model(model, train)
        mae = evaluate_model(model, test)
    except Exception as e:
        mae = f"CRASH ({e})"
    
    with open(outfile, "a") as f:
        f.write(f"{now},{mae}\n")
