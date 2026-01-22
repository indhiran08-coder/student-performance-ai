import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# ===============================
# STEP 4: SAVE & LOAD AI MODEL
# ===============================

# 1️⃣ Load dataset
data = pd.read_csv("data/students.csv")

X = data[["Attendance", "StudyHours", "InternalMarks"]]
y = data["FinalMarks"]

# 2️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4️⃣ Train BEST model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 5️⃣ Save model & scaler
joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Best AI model saved successfully")
print("✅ Scaler saved successfully")
