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

