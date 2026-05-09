# 🏠 House Price Prediction using Machine Learning

## 🚀 Project Overview

This project predicts California house prices using Machine Learning regression models.

The goal of this project was to understand the difference between **classification** and **regression**, build a baseline regression model, improve performance using an advanced algorithm, and analyze the most important features affecting house prices.

---

## 📘 Classification vs Regression

### Classification

Classification predicts categories or labels.

Examples:

- Customer Churn → Yes / No
- Spam Detection → Spam / Not Spam
- Disease Prediction → Positive / Negative

Output:

```text
Discrete classes
```

---

### Regression

Regression predicts continuous numerical values.

Examples:

- House Price → ₹45,00,000
- Salary Prediction → ₹12,50,000
- Temperature Forecast → 32.5°C

Output:

```text
Continuous numeric value
```

---

## 📂 Dataset

Dataset used:
**California Housing Dataset**

Loaded directly from Scikit-learn:

```python
fetch_california_housing(as_frame=True)
```

Dataset details:

- 20,640 rows
- 8 input features
- 1 target column

Target variable:

```text
MedHouseVal
```

Features:

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

---

## 🧠 Machine Learning Workflow

### 1. Data Loading

Loaded California housing dataset into a Pandas DataFrame.

---

### 2. Data Preparation

Separated:

- Features (`X`)
- Target (`y`)

```python
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
```

---

### 3. Train-Test Split

Split data into:

- 80% training
- 20% testing

```python
train_test_split(test_size=0.2, random_state=42)
```

---

### 4. Baseline Model

Started with:

```python
LinearRegression()
```

Baseline performance:

| Metric   | Score |
| -------- | ----: |
| MAE      | 0.533 |
| MSE      | 0.556 |
| RMSE     | 0.746 |
| R² Score | 0.576 |

---

### 5. Model Improvement

Improved performance using:

```python
RandomForestRegressor()
```

Pipeline used:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor())
])
```

---

## 📊 Final Model Performance

| Metric   | Score |
| -------- | ----: |
| MAE      | 0.327 |
| MSE      | 0.255 |
| RMSE     | 0.505 |
| R² Score | 0.805 |

### Interpretation

**MAE**
Average prediction error:

```text
~$32,700
```

**RMSE**
Typical prediction error:

```text
~$50,500
```

**R² Score**
Model explains:

```text
80.5% of house price variation
```

---

## 📌 Feature Importance

Random Forest feature importance revealed:

| Feature    | Importance |
| ---------- | ---------: |
| MedInc     |   0.524871 |
| AveOccup   |   0.138443 |
| Latitude   |   0.088936 |
| Longitude  |   0.088629 |
| HouseAge   |   0.054593 |
| AveRooms   |   0.044272 |
| Population |   0.030650 |
| AveBedrms  |   0.029606 |

### Key Insights

- Median income was the strongest predictor
- Location (latitude/longitude) significantly impacted pricing
- Occupancy also contributed meaningfully

---

## 📈 Visualization

### Actual vs Predicted Plot

Scatter plot used to compare actual house prices vs predicted house prices.

Observation:

- Strong diagonal trend
- Predictions closely aligned with actual values
- Some spread at higher values due to dataset complexity

---

## 💾 Model Saving

Saved trained model using Joblib:

```python
joblib.dump(pipeline, "house_price_model.pkl")
```

This allows reusing the trained model without retraining.

---

## 📁 Project Structure

```text
HousePricePrediction/
│
├── house_price_prediction.py
├── house_price_model.pkl
├── README.md
```

---

## 🎯 Learning Outcomes

This project helped me understand:

✅ Regression vs Classification  
✅ Train/Test Split  
✅ Linear Regression  
✅ Random Forest Regression  
✅ Model Evaluation Metrics  
✅ MAE / MSE / RMSE / R²  
✅ Feature Importance  
✅ Data Visualization  
✅ ML Pipelines  
✅ Model Persistence

---

## 🚀 Future Improvements

Possible enhancements:

- Hyperparameter tuning
- Cross-validation
- Custom user input prediction
- Flask/FastAPI deployment
- Streamlit web app
- XGBoost comparison

---

## Conclusion

This project demonstrates an end-to-end machine learning regression workflow, from data preparation to model training, evaluation, visualization, and model persistence.

Random Forest significantly outperformed Linear Regression, improving prediction quality and explaining over 80% of the variance in housing prices.
