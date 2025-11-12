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
