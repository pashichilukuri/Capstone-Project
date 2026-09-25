# Titanic Survival Analysis & Prediction

An end-to-end data science project on the classic **Titanic** dataset (loaded via `seaborn.load_dataset('titanic')`). The notebook/script covers data cleaning, exploratory data analysis (EDA), feature engineering, classification modeling, class-imbalance handling, hyperparameter tuning, and a bonus regression task predicting `fare`.

## Project Structure

```
.
├── titanic_analysis.py / .ipynb   # main script/notebook (your code)
├── titanic.csv                    # raw dataset exported from seaborn
├── univariate_age_fare.png        # histogram + boxplot of age & fare
├── correlation_heatmap.png        # correlation matrix heatmap
├── requirements.txt
└── README.md
```

## What the Code Does

### 1. Data Loading & Profiling
- Loads the Titanic dataset from `seaborn` and exports it to `titanic.csv`.
- Inspects shape, dtypes, and summary statistics with `df.info()` and `df.describe()`.
- Computes the percentage of missing values per column.

### 2. Data Cleaning
- Fills missing `age` values with the median.
- Drops rows with missing `embarked` / `embark_town`.
- Fills missing `deck` values with a new `"Missing"` category.
- Re-checks the missing-value report after cleaning.

### 3. Univariate Analysis
- Histograms and boxplots for `age` and `fare`.
- IQR-based outlier detection (`Q1 − 1.5×IQR`, `Q3 + 1.5×IQR`) for both columns.
- Mean/median/mode comparison for `fare` to assess skewness (found to be **right-skewed**: mean > median > mode).

### 4. Survival Rate Analysis (Boolean Masking)
- Survival rate by **sex**.
- Survival rate by **passenger class**.
- Survival rate by **sex × passenger class** combined.

### 5. Correlation & Multivariate Visualization
- Correlation matrix for `survived, pclass, age, sibsp, parch, fare` with a heatmap.
- Four charts building a narrative around survival:
  1. Survival rate by sex (bar)
  2. Survival rate by pclass (bar)
  3. Survival rate by pclass & sex (grouped bar)
  4. Age vs. Fare scatter, colored by survival and styled by sex

### 6. Preprocessing for Modeling
- Z-score standardization of `age` and `fare` via `StandardScaler`.
- Stratified train/test split (80/20) on `survived`.
- A `ColumnTransformer` pipeline:
  - Categorical (`sex`, `embarked`): most-frequent imputation + one-hot encoding.
  - Numeric (`age`, `pclass`, `sibsp`, `parch`, `fare`): median imputation + standard scaling.
- All preprocessing is **fit only on the training split** and applied transform-only to the test split.

### 7. Classification Models
Three classifiers trained on the same split, each wrapped in the shared preprocessing pipeline:
- Logistic Regression
- Decision Tree (`max_depth=4`, `min_samples_leaf=10`, visualized with `plot_tree`)
- Random Forest (`n_estimators=200`, `max_depth=4`)

Evaluated with confusion matrix, accuracy, precision, recall, F1, and ROC-AUC, summarized in a side-by-side comparison table.

### 8. Class Imbalance Handling
Compares three strategies on Logistic Regression:
- Baseline (no handling)
- `class_weight='balanced'`
- SMOTE oversampling (via `imblearn.pipeline.Pipeline`)

### 9. Hyperparameter Tuning
- `RandomForestClassifier` with `oob_score=True` wrapped in the preprocessing pipeline.
- `GridSearchCV` over `n_estimators`, `max_depth`, and `max_features`, optimizing F1 with 5-fold CV.

### 10. Regression Side-Task: Predicting Fare
- Multivariate Linear Regression predicting `fare` from the other features.
- Metrics: MAE, RMSE, R², and Adjusted R².
- Residual plot (Predicted Fare vs. Residuals) to visually assess homoscedasticity/heteroscedasticity.

### 11. Final Model Comparison Table
A single table combining classification metrics (Accuracy, Precision, Recall, F1, AUC) for the three classifiers and regression metrics (MAE, RMSE, R², Adjusted R²) for the linear regression model, side by side.

## Setup

```bash
pip install -r requirements.txt
```

## Notes / Known Issues to Fix Before Running

A few lines in the pasted code will raise errors as-is and should be corrected:

1. **Syntax error** — missing comma in Random Forest constructor:
   ```python
   RandomForestClassifier(max_depth=4, n_estimators=200 random_state=42)
   ```
   should be:
   ```python
   RandomForestClassifier(max_depth=4, n_estimators=200, random_state=42)
   ```

2. **Wrong variable in F1 score** for Decision Tree and Random Forest — uses `y_pred` (undefined at that point) instead of `y_pred_DT` / `y_pred_RF`:
   ```python
   F1_DT = f1_score(y_test, y_pred)   # should be y_pred_DT
   F1_RF = f1_score(y_test, y_pred)   # should be y_pred_RF
   ```

3. **`linear_model` is used but never defined** before `linear_model.fit(...)`. It needs to be built as a pipeline, e.g.:
   ```python
   preprocessor_reg = ColumnTransformer([
       ('categorical', categorical_pipeline, categorical_columns_reg),
       ('numeric', numeric_pipeline, numerical_features_reg)
   ])
   linear_model = Pipeline([
       ('preprocessor', preprocessor_reg),
       ('regressor', LinearRegression())
   ])
   ```

4. **`param_grid` has a string `"0.5"` instead of a float** in `max_features`:
   ```python
   "max_features": ["sqrt", "log2", "0.5"]   # should be 0.5 (float), not a string
   ```

5. **`GridSearchCV` object (`grid_search`) is defined but `.fit()` is never called** on it — add `grid_search.fit(X_train, y_train)` if you want the tuned model.

6. `display()` is a Jupyter-only function — if running as a plain `.py` script, replace with `print()`.

## Key Findings

- **Sex** is the strongest single predictor of survival (women survived at a much higher rate than men).
- **Passenger class** also strongly predicts survival, with 1st class > 2nd class > 3rd class.
- The effect of sex holds **within** every passenger class, suggesting both factors matter independently (consistent with "women and children first" combined with unequal access to lifeboats by class).
- `fare` is right-skewed with significant high-value outliers, and correlates with `pclass` (inversely) and `survived`.

## License

For educational/portfolio use with the public Titanic dataset (via seaborn).
