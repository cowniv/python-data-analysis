"""Shared helpers for the two analysis scripts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def load_csv(name):
    """Read a CSV from data/ and print a quick overview."""
    df = pd.read_csv(DATA_DIR / name)
    print(f"\n=== {name}: {df.shape[0]} rows x {df.shape[1]} columns ===")
    print(df.head())
