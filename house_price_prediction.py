import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# -----------------------------
# 1. Load Dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 3. Build ML Pipeline
# -----------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# -----------------------------
# 4. Train Model
# -----------------------------
pipeline.fit(X_train, y_train)
print("✅ Model trained successfully")


# -----------------------------
# 5. Make Predictions
# -----------------------------
y_pred = pipeline.predict(X_test)


# -----------------------------
# 6. Evaluate Model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance")
print(f"MAE      : {mae:.3f}")
print(f"MSE      : {mse:.3f}")
print(f"RMSE     : {rmse:.3f}")
print(f"R2 Score : {r2:.3f}")


# -----------------------------
# 7. Feature Importance
# -----------------------------
feature_importance = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\n📌 Feature Importance")
print(importance_df)


# -----------------------------
# 8. Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()


# -----------------------------
# 9. Save Model
# -----------------------------
joblib.dump(pipeline, "house_price_model.pkl")
print("\n💾 Model saved successfully")
