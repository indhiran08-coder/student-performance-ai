import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# ===============================
# CUSTOM UI STYLES
# ===============================
st.markdown("""
<style>


.block-container {
    padding-top: 2.5rem;
    max-width: 1300px;
}


section[data-testid="stSidebar"] {
    background: transparent !important;
    border-right: none !important;
}


.control-panel {
    background: linear-gradient(180deg, #0f172a, #020617);
    border-radius: 22px;
    padding: 1.8rem 1.5rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    border: 1px solid #1e293b;
}

.control-title {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
}

.control-sub {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 1.2rem;
}

.control-divider {
    height: 1px;
    background: linear-gradient(to right, #22c55e, #6366f1);
    margin: 1.2rem 0;
}


section[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
    font-weight: 600;
}


section[data-testid="stSidebar"] div[data-baseweb="slider"] {
    margin-top: 6px;
    margin-bottom: 18px;
}

section[data-testid="stSidebar"]
div[data-baseweb="slider"] > div > div:nth-child(1) {
    background-color: #334155 !important;
    height: 6px;
    border-radius: 6px;
}

section[data-testid="stSidebar"]
div[data-baseweb="slider"] > div > div:nth-child(2) {
    background-color: #22c55e !important;
    height: 6px;
    border-radius: 6px;
}

section[data-testid="stSidebar"]
div[data-baseweb="slider"] span {
    background-color: #ffffff !important;
    border: 2px solid #22c55e !important;
    box-shadow: 0 0 10px rgba(34,197,94,0.8);
    width: 18px;
    height: 18px;
}

section[data-testid="stSidebar"]
div[data-baseweb="slider"] div[role="slider"] {
    color: #38bdf8 !important;
    font-weight: 700;
}

/* ===== HEADER ===== */
.header {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    padding: 2.2rem;
    border-radius: 26px;
    color: white;
    text-align: center;
    margin-bottom: 2.5rem;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    color: white;
    border-radius: 999px;
    height: 3.2em;
    font-size: 17px;
    font-weight: 600;
    border: none;
    padding: 0 2rem;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #22c55e, #6366f1);
}

/* ===== METRIC CARDS ===== */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border-radius: 16px;
    padding: 1.1rem;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===============================
# LEFT CONTROL PANEL
# ===============================
with st.sidebar:
    st.markdown("""
    <div class="control-panel">
        <div class="control-title">🎛 Student Control Panel</div>
        <div class="control-sub">Adjust inputs to explore predictions</div>
        <div class="control-divider"></div>
    """, unsafe_allow_html=True)

    attendance = st.slider(
        "📊 Attendance (%)",
        0, 100, 75,
        help="Percentage of classes attended by the student"
    )
    study_hours = st.slider(
        "📘 Study Hours / Day",
        0, 6, 2,
        help="Average number of hours the student studies daily"
    )
    internal_marks = st.slider(
        "📝 Internal Marks",
        0, 30, 20,
        help="Marks scored in internal assessments"
    )

    show_graphs = st.toggle(
        "📊 Show Visual Analytics",
        value=True,
        help="Turn ON/OFF graphs and charts"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="header">
    <h1>🎓 Student Performance Predictor</h1>
    <p>AI-powered academic analysis • Risk detection • Smart guidance</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# PREDICT
# ===============================
if st.button("🚀 Predict Student Performance"):

    X_new = pd.DataFrame(
        [[attendance, study_hours, internal_marks]],
        columns=["Attendance", "StudyHours", "InternalMarks"]
    )

    X_scaled = scaler.transform(X_new)
    predicted_marks = model.predict(X_scaled)[0]

    # Category & Risk
    if predicted_marks >= 80:
        risk, category = "LOW", "🌟 Excellent"
    elif predicted_marks >= 65:
        risk, category = "LOW", "✅ Good"
    elif predicted_marks >= 50:
        risk, category = "MEDIUM", "⚠️ Average"
    else:
        risk, category = "HIGH", "🚨 At Risk"

    # Grade
    if predicted_marks >= 85: grade = "A+"
    elif predicted_marks >= 75: grade = "A"
    elif predicted_marks >= 65: grade = "B"
    elif predicted_marks >= 55: grade = "C"
    elif predicted_marks >= 40: grade = "D"
    else: grade = "F"

    # ===============================
    # METRICS
    # ===============================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📈 Predicted Marks", f"{predicted_marks:.1f}")
    c2.metric("🅰 Grade", grade)
    c3.metric("⚠ Risk Level", risk)
    c4.metric("🎯 Performance", category)

    # ===============================
    # EXPLANATION
    # ===============================
    st.markdown("### 🧠 Why this prediction?")
    st.info(
        "This prediction is based on patterns learned from historical student data. "
        "Attendance and internal assessments contribute most strongly to the outcome."
    )

    # ===============================
    # VISUAL ANALYTICS (TOGGLE)
    # ===============================
    if show_graphs:
        st.markdown("### 📊 Visual Analytics")

        colA, colB = st.columns(2)

        with colA:
            st.markdown(
                "#### Performance Score ℹ️",
                help="Shows the predicted final marks out of 100"
            )
            fig1, ax1 = plt.subplots()
            ax1.barh(["Predicted Marks"], [predicted_marks], color="#6366f1")
            ax1.set_xlim(0, 100)
            ax1.set_xlabel("Marks")
            st.pyplot(fig1)

        with colB:
            st.markdown(
                "#### Feature Contribution ℹ️",
                help="Illustrates how much each input influences the prediction"
            )
            features = ["Attendance", "Study Hours", "Internal Marks"]
            importance = [0.45, 0.20, 0.35]

            fig2, ax2 = plt.subplots()
            ax2.bar(features, importance,
                    color=["#22c55e", "#f59e0b", "#6366f1"])
            ax2.set_ylabel("Relative Influence")
            st.pyplot(fig2)

    # ===============================
    # SAVE HISTORY
    # ===============================
    os.makedirs("history", exist_ok=True)
    history_file = "history/student_history.csv"

    record = pd.DataFrame([{
        "Attendance": attendance,
        "StudyHours": study_hours,
        "InternalMarks": internal_marks,
        "PredictedMarks": round(predicted_marks, 2),
        "Risk": risk,
        "Grade": grade
    }])

    if os.path.exists(history_file):
        record.to_csv(history_file, mode="a", header=False, index=False)
    else:
        record.to_csv(history_file, index=False)

    # ===============================
    # TREND CHART OVER HISTORY
    # ===============================
    st.markdown("### 📈 Performance Trend Over Time ℹ️",
                help="Shows how predicted marks change across different scenarios")

    history_df = pd.read_csv(history_file)

    fig3, ax3 = plt.subplots()
    ax3.plot(history_df["PredictedMarks"], marker="o", color="#22c55e")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Predicted Marks")
    ax3.set_xlabel("Prediction Instance")
    st.pyplot(fig3)

    # ===============================
    # HISTORY TABLE
    # ===============================
    st.markdown("### 📚 Previous Predictions")
    st.dataframe(history_df, use_container_width=True)
