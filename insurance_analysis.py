"""Medical insurance charges: EDA + linear regression.

Questions:
  1. How much more do smokers pay?
  2. How do BMI and age relate to charges?
  3. How well can a linear model predict charges?
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils import actual_vs_predicted, load_csv, regression_report, save_fig, section

df = load_csv("insurance.csv")

# ---------------- EDA ----------------
section("Summary statistics")
print(df.describe().round(2))

section("Average charges by group")
for col in ["smoker", "sex", "region", "children"]:
    print(df.groupby(col)["charges"].agg(["count", "mean", "median"]).round(0), "\n")

smoker_mean = df.groupby("smoker")["charges"].mean()
print(f"Smokers pay about {smoker_mean['yes'] / smoker_mean['no']:.1f}x more on average.")

# BMI categories (WHO cut-offs)
df["bmi_group"] = pd.cut(
    df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=["underweight", "normal", "overweight", "obese"]
)
print("\nMean charges by smoker and BMI group:")
print(df.pivot_table(values="charges", index="bmi_group", columns="smoker", aggfunc="mean", observed=True).round(0))

# ---------------- Figures ----------------
section("Figures")
colors = {"no": "tab:blue", "yes": "tab:red"}

# 1. distribution of charges
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["charges"], bins=40, color="tab:purple", edgecolor="white")
ax.set_title("Distribution of insurance charges")
ax.set_xlabel("Charges")
ax.set_ylabel("Count")
save_fig(fig, "insurance_charges_hist.png")

# 2. smoker vs non-smoker
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([df[df.smoker == "no"]["charges"], df[df.smoker == "yes"]["charges"]])
ax.set_xticks([1, 2], ["Non-smoker", "Smoker"])
ax.set_title("Charges by smoking status")
ax.set_ylabel("Charges")
save_fig(fig, "insurance_smoker_box.png")

# 3. BMI vs charges, coloured by smoker
fig, ax = plt.subplots(figsize=(7, 5))
for status, group in df.groupby("smoker"):
    ax.scatter(group["bmi"], group["charges"], s=12, alpha=0.6, c=colors[status], label=f"smoker = {status}")
ax.axvline(30, color="gray", linestyle="--", linewidth=1)
ax.text(30.5, df["charges"].max() * 0.95, "BMI 30 (obese)", color="gray")
ax.set_title("BMI vs charges")
ax.set_xlabel("BMI")
ax.set_ylabel("Charges")
ax.legend()
save_fig(fig, "insurance_bmi_scatter.png")

# 4. age vs charges
fig, ax = plt.subplots(figsize=(7, 5))
for status, group in df.groupby("smoker"):
    ax.scatter(group["age"], group["charges"], s=12, alpha=0.6, c=colors[status], label=f"smoker = {status}")
ax.set_title("Age vs charges")
ax.set_xlabel("Age")
ax.set_ylabel("Charges")
ax.legend()
save_fig(fig, "insurance_age_scatter.png")

# ---------------- Linear regression ----------------
section("Linear regression")
X = pd.get_dummies(df[["age", "sex", "bmi", "children", "smoker", "region"]], drop_first=True, dtype=int)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model A: all features
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
