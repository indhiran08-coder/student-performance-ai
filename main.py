import pandas as pd
import joblib

# ===============================
# FINAL AI DECISION SUPPORT SYSTEM
# ===============================

# Load saved model & scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Take user input
attendance = int(input("Enter Attendance (%): "))
study_hours = int(input("Enter Study Hours per day: "))
internal_marks = int(input("Enter Internal Marks: "))

# Create input dataframe
new_student = pd.DataFrame(
    [[attendance, study_hours, internal_marks]],
    columns=["Attendance", "StudyHours", "InternalMarks"]
)

new_student_scaled = scaler.transform(new_student)

# Predict final marks
predicted_marks = model.predict(new_student_scaled)[0]

# Risk classification
if predicted_marks >= 70:
    risk = "LOW RISK ✅"
elif predicted_marks >= 50:
    risk = "MEDIUM RISK ⚠️"
else:
    risk = "HIGH RISK 🚨"

# Recommendation logic
recommendations = []

if attendance < 75:
    recommendations.append("Increase attendance to improve performance.")
if study_hours < 2:
    recommendations.append("Increase daily study hours.")
if internal_marks < 20:
    recommendations.append("Focus on internal assessments.")

if not recommendations:
    recommendations.append("Excellent performance. Keep it up!")

# Display final output
print("\n🎓 AI STUDENT PERFORMANCE DECISION SUPPORT")
print("------------------------------------------")
print(f"Predicted Final Marks : {round(predicted_marks, 2)}")
print(f"Risk Level            : {risk}")

print("\n💡 Recommendations:")
for rec in recommendations:
    print("-", rec)
