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
save_fig(fig, "mtcars_mpg_vs_wt.png")

# only 32 cars, so use 5-fold cross-validation instead of one small test split
for features in (["wt"], ["wt", "hp"], ["wt", "hp", "cyl", "am"]):
    pred = cross_val_predict(LinearRegression(), df[features], y, cv=5)
    regression_report(y, pred, f"{'+'.join(features):<14} (5-fold CV)")

best = ["wt", "hp"]
pred = cross_val_predict(LinearRegression(), df[best], y, cv=5)
actual_vs_predicted(y, pred, "MPG: actual vs predicted (wt + hp)", "mtcars_actual_vs_pred.png")

# ---------------- K-means ----------------
section("K-means clustering")
features = ["mpg", "hp", "wt", "qsec"]
X = StandardScaler().fit_transform(df[features])  # scale so hp doesn't dominate

# elbow method
inertias = []
ks = range(1, 9)
for k in ks:
    inertias.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(ks), inertias, "o-")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
ax.set_title("Elbow method")
save_fig(fig, "mtcars_elbow.png")

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
df["cluster"] = kmeans.labels_
print(df.groupby("cluster")[features + ["cyl"]].mean().round(1))
for c in sorted(df["cluster"].unique()):
    names = ", ".join(df[df.cluster == c].index[:5])
    print(f"  cluster {c}: {names}...")

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(df["hp"], df["mpg"], c=df["cluster"], cmap="viridis", s=60)
for name, row in df.iterrows():
    ax.annotate(name, (row["hp"], row["mpg"]), fontsize=6, alpha=0.7)
ax.set_xlabel("Horsepower")
ax.set_ylabel("MPG")
ax.set_title("K-means clusters (k=3)")
ax.legend(*scatter.legend_elements(), title="cluster")
save_fig(fig, "mtcars_kmeans.png")
