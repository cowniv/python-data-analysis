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
y = df["mpg"]

# simple: weight only
simple = LinearRegression().fit(df[["wt"]], y)
print(f"  mpg = {simple.intercept_:.2f} + ({simple.coef_[0]:.2f} x wt)")
print(f"  -> every extra 1000 lbs costs about {abs(simple.coef_[0]):.1f} mpg")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["wt"], y)
xs = np.linspace(df["wt"].min(), df["wt"].max(), 50)
ax.plot(xs, simple.predict(pd.DataFrame({"wt": xs})), "r-", label="fitted line")
ax.set_xlabel("Weight (1000 lbs)")
ax.set_ylabel("MPG")
ax.set_title("MPG vs weight")
ax.legend()
