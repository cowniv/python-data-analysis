# 22 · Python Data Analysis (Insurance + mtcars)

**Roadmap stage:** Phase 7 – Backend + Data · November 2025
**Inspired by:** my Pattern Recognition coursework (pandas, matplotlib, linear regression and k-means on the same two datasets)

In Pattern Recognition we worked through these datasets in Jupyter notebooks. I wanted to redo them as clean, rerunnable scripts instead of scattered cells: one script per dataset, shared helpers in `utils.py`, and every chart saved to `figures/`.

### Features
- `insurance_analysis.py`: summary stats, average charges by smoker / sex / region / children, BMI groups, 4 charts, and linear regression (with and without a smoker × BMI term)
- `mtcars_analysis.py`: correlation heatmap, mpg vs weight line, multiple regression with 5-fold cross-validation, elbow method + k-means (k = 3)
- `utils.py`: CSV loading, figure saving, R² / MAE / RMSE report, actual vs predicted plot

### Run
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python insurance_analysis.py
python mtcars_analysis.py
```
Charts are written to `figures/` (ignored by git, just rerun the scripts).

### Findings
- **Smoking is the biggest factor.** Smokers pay about **3.8×** more on average, and the regression gives smoking a coefficient of about **+23,600** while everything else is in the hundreds.
- **BMI only matters a lot for smokers.** Obese non-smokers average ~8.9k, obese smokers ~41.7k. Adding a smoker × BMI term raised test R² from **0.78 to 0.87** and cut MAE from ~4.2k to ~2.8k.
- **Age raises charges steadily**, about +257 per year, in three visible "bands" on the age scatter (non-smokers, smokers, obese smokers).
- **Weight predicts mpg best** (r = −0.87): roughly **−5.3 mpg per 1000 lbs**. Weight + horsepower gets a cross-validated R² of about **0.75**. Adding cylinders and transmission barely helps.
- **K-means found 3 sensible groups:** heavy V8 muscle cars (~15 mpg, ~207 hp), light economy cars (~30 mpg), and mid-size cars in between.

### What I practiced
- pandas: `groupby`, `pivot_table`, `pd.cut`, `get_dummies`
- matplotlib: histograms, box plots, scatter plots, heatmaps with `imshow`
- scikit-learn: `LinearRegression`, `train_test_split`, `cross_val_predict`, `StandardScaler`, `KMeans`
- Reading regression metrics (R², MAE, RMSE) and coefficients
- Turning notebook code into reusable modules
