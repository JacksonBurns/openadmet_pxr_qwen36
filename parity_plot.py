import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

preds = pd.read_csv("predictions.csv")
test = pd.read_csv("test_phase1.csv")

y_true = test["pEC50"].values
y_pred = preds["Ensemble"].values

mae = mean_absolute_error(y_true, y_pred)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=15)
xlim = ax.get_xlim()
ax.plot(xlim, xlim, "r-", linewidth=1)
ax.set_xlabel("True pEC50", fontsize=12)
ax.set_ylabel("Predicted pEC50", fontsize=12)
ax.set_title(f"Parity Plot\nMAE = {mae:.4f}", fontsize=14)
plt.tight_layout()
plt.savefig("parity_plot.png", dpi=150)
print("Saved parity_plot.png")
