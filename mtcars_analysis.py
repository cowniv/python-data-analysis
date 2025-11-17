"""mtcars: correlations, predicting mpg, and grouping cars with k-means."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from utils import actual_vs_predicted, load_csv, regression_report, save_fig, section

df = load_csv("mtcars.csv").set_index("model")

# ---------------- Correlations ----------------
section("Correlation with mpg")
corr = df.corr()
print(corr["mpg"].drop("mpg").sort_values().round(2))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("mtcars correlation matrix")
save_fig(fig, "mtcars_corr_heatmap.png")

# ---------------- mpg regression ----------------
section("Predicting mpg")
