import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# ===============================
# STEP 2: EXPLAINABLE AI (XAI)
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
X_train_scaled = scaler.fit_transform(X_train)

# 4️⃣ Train BEST model (from Step 1)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 5️⃣ Feature Importance
importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\n🔍 FEATURE IMPORTANCE (Explainable AI)")
print(importance_df)

# 6️⃣ Visualization
plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance - Explainable AI")
plt.gca().invert_yaxis()
plt.show()
