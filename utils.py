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
    missing = df.isna().sum().sum()
    print(f"Missing values: {missing}")
    return df


def section(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def save_fig(fig, filename):
    """Save a figure into figures/ and close it."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def regression_report(y_true, y_pred, label="model"):
    """Print and return the usual regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"  {label}: R^2 = {r2:.3f} | MAE = {mae:,.2f} | RMSE = {rmse:,.2f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}


