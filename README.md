# 22 · Python Data Analysis (Insurance + mtcars)

**Roadmap stage:** Phase 7 – Backend + Data · November 2025
**Inspired by:** my Pattern Recognition coursework (pandas, matplotlib, linear regression and k-means on the same two datasets)

In Pattern Recognition we worked through these datasets in Jupyter notebooks. I wanted to redo them as clean, rerunnable scripts instead of scattered cells: one script per dataset, shared helpers in `utils.py`, and every chart saved to `figures/`.

### Features
- `insurance_analysis.py`: summary stats, average charges by smoker / sex / region / children, BMI groups, 4 charts, and linear regression (with and without a smoker × BMI term)
- `mtcars_analysis.py`: correlation heatmap, mpg vs weight line, multiple regression with 5-fold cross-validation, elbow method + k-means (k = 3)
- `utils.py`: CSV loading, figure saving, R² / MAE / RMSE report, actual vs predicted plot

