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
