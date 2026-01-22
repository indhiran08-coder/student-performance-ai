import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# ===============================
# STEP 3: RISK & RECOMMENDATION
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

# 4️⃣ Train best model (Gradient Boosting)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 5️⃣ Take user input
attendance = int(input("Enter Attendance (%): "))
study_hours = int(input("Enter Study Hours per day: "))
internal_marks = int(input("Enter Internal Marks: "))

# 6️⃣ Create input dataframe
new_student = pd.DataFrame(
    [[attendance, study_hours, internal_marks]],
    columns=["Attendance", "StudyHours", "InternalMarks"]
)

new_student_scaled = scaler.transform(new_student)

# 7️⃣ Predict marks
predicted_marks = model.predict(new_student_scaled)[0]

# 8️⃣ Risk Classification
if predicted_marks >= 70:
    risk = "LOW RISK ✅"
elif predicted_marks >= 50:
    risk = "MEDIUM RISK ⚠️"
else:
    risk = "HIGH RISK 🚨"

# 9️⃣ Identify weak features
feature_means = X.mean()
weak_factors = []

if attendance < feature_means["Attendance"]:
    weak_factors.append("Attendance")
if study_hours < feature_means["StudyHours"]:
    weak_factors.append("Study Hours")
if internal_marks < feature_means["InternalMarks"]:
    weak_factors.append("Internal Marks")

# 🔟 Recommendation Engine
recommendations = []

if "Attendance" in weak_factors:
    recommendations.append("Increase attendance by attending all classes regularly.")
if "Study Hours" in weak_factors:
    recommendations.append("Increase daily study time by at least 1 hour.")
if "Internal Marks" in weak_factors:
    recommendations.append("Improve internal assessment performance through practice tests.")

if not recommendations:
    recommendations.append("Keep up the good work! Maintain current performance.")

# 1️⃣1️⃣ Display Output
print("\n🎓 AI STUDENT PERFORMANCE REPORT")
print("--------------------------------")
print(f"Predicted Final Marks : {round(predicted_marks, 2)}")
print(f"Risk Level            : {risk}")

print("\n🔍 Identified Weak Areas:")
for wf in weak_factors:
    print("-", wf)

print("\n💡 Personalized Recommendations:")
for rec in recommendations:
    print("-", rec)
