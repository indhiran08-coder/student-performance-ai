import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ===============================
# STEP 1: MULTI-MODEL COMPARISON
# ===============================

# 1️⃣ Load dataset
data = pd.read_csv("data/students.csv")

X = data[["Attendance", "StudyHours", "InternalMarks"]]
y = data["FinalMarks"]

# 2️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4️⃣ Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

# 5️⃣ Train, Evaluate, Compare
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    cv_score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2"
    ).mean()

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 2),
        "CV Score": round(cv_score, 2)
    })

# 6️⃣ Display Comparison Table
results_df = pd.DataFrame(results)
print("\n📊 MODEL COMPARISON RESULTS")
print(results_df)

# 7️⃣ Auto-select Best Model
best_model = results_df.sort_values(
    by=["R2 Score", "MAE"],
    ascending=[False, True]
).iloc[0]

print("\n🏆 BEST MODEL SELECTED")
print(best_model)
