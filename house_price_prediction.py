import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================================================
# STEP 1 — LOAD DATA
# ======================================================
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset Loaded Successfully\n")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ======================================================
# STEP 2 — FEATURES & TARGET
# ======================================================
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# ======================================================
# STEP 3 — TRAIN / TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# STEP 4 — BUILD PIPELINE
# ======================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# ======================================================
# STEP 5 — CROSS VALIDATION
# ======================================================
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\n================ CROSS VALIDATION RESULTS ================\n")
print("R2 Scores:", cv_scores)
print("Average R2:", cv_scores.mean())

# ======================================================
# STEP 6 — TRAIN BASELINE MODEL
# ======================================================
pipeline.fit(X_train, y_train)

baseline_pred = pipeline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = baseline_mse ** 0.5
baseline_r2 = r2_score(y_test, baseline_pred)

print("\n================ BASELINE MODEL RESULTS ================\n")
print("MAE:", baseline_mae)
print("MSE:", baseline_mse)
print("RMSE:", baseline_rmse)
print("R2 Score:", baseline_r2)

# ======================================================
# STEP 7 — HYPERPARAMETER TUNING
# ======================================================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n================ GRID SEARCH RESULTS ================\n")
print("Best Parameters:", grid.best_params_)
print("Best Cross Validation R2:", grid.best_score_)

# ======================================================
# STEP 8 — BEST MODEL EVALUATION
# ======================================================
best_model = grid.best_estimator_

best_pred = best_model.predict(X_test)

best_mae = mean_absolute_error(y_test, best_pred)
best_mse = mean_squared_error(y_test, best_pred)
best_rmse = best_mse ** 0.5
best_r2 = r2_score(y_test, best_pred)

print("\n================ BEST MODEL RESULTS ================\n")
print("MAE:", best_mae)
print("MSE:", best_mse)
print("RMSE:", best_rmse)
print("R2 Score:", best_r2)

# ======================================================
# STEP 9 — FEATURE IMPORTANCE
# ======================================================
feature_importance = best_model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\n================ FEATURE IMPORTANCE ================\n")
print(importance_df)

# ======================================================
# STEP 10 — VISUALIZATION
# ======================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.5)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

# ======================================================
# STEP 11 — SAVE MODEL
# ======================================================
joblib.dump(best_model, "house_price_model.pkl")

print("\nModel Saved Successfully")
