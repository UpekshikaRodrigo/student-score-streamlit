# app/app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================
# Config
# ============================
st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")

# ✅ Deployment-safe absolute paths (relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_best_pipeline.joblib"

ID_COL = "id"
TARGET = "exam_score"

# Put your best CV RMSE from Task E here
CV_RMSE = 8.80

# =============================
# UI Styling (dropdown readability + hide number steppers)
# =============================
st.markdown(
    """
<style>
div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 8px !important;
}
ul[role="listbox"] {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 8px !important;
}
li[role="option"]:hover {
    background-color: #2563eb !important;
    color: white !important;
}
li[aria-selected="true"] {
    background-color: #1d4ed8 !important;
    color: white !important;
}
.stSelectbox, .stNumberInput {
    margin-bottom: 12px !important;
}
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
input[type="number"] {
    -moz-appearance: textfield;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Category options (from dataset)
# -----------------------------
GENDER_OPTS = ["female", "male", "other"]
COURSE_OPTS = ["b.com", "b.sc", "b.tech", "ba", "bba", "bca", "diploma"]
INTERNET_OPTS = ["no", "yes"]
SLEEP_QUALITY_OPTS = ["average", "good", "poor"]
STUDY_METHOD_OPTS = ["coaching", "group study", "mixed", "online videos", "self-study"]
FACILITY_OPTS = ["high", "low", "medium"]
DIFFICULTY_OPTS = ["easy", "hard", "moderate"]

PH = "— Select —"
GENDER_UI = [PH] + GENDER_OPTS
COURSE_UI = [PH] + COURSE_OPTS
INTERNET_UI = [PH] + INTERNET_OPTS
SLEEP_QUALITY_UI = [PH] + SLEEP_QUALITY_OPTS
STUDY_METHOD_UI = [PH] + STUDY_METHOD_OPTS
FACILITY_UI = [PH] + FACILITY_OPTS
DIFFICULTY_UI = [PH] + DIFFICULTY_OPTS

# ============================
# Load model
# ============================
@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(str(model_path))

# ============================
# Feature engineering (MUST match Task E)
# ============================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "study_hours" in df.columns and "class_attendance" in df.columns:
        df["study_effort"] = df["study_hours"] * df["class_attendance"]
    if "sleep_hours" in df.columns and "study_hours" in df.columns:
        df["sleep_study_ratio"] = df["sleep_hours"] / (df["study_hours"] + 1e-6)
    return df

# ============================
# Grade + Interpretation + Recommendations
# ============================
def score_to_grade(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Very Good"
    elif score >= 60:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Improvement"

def interpretation_text(score: float) -> str:
    if score >= 90:
        return "Outstanding performance expected. Maintain current routine and continue strategic revision."
    elif score >= 75:
        return "Strong performance expected. Keep effort consistent and revise key topics to sustain results."
    elif score >= 60:
        return "Stable performance expected. Improve consistency and focus on weak areas to move up a band."
    elif score >= 50:
        return "Borderline performance. Improve attendance and study consistency to reduce risk of underperformance."
    else:
        return "At-risk performance. A structured study plan and support (coaching/group study) is recommended."

def build_recommendations(
    study_hours: float,
    class_attendance_pct: float,
    sleep_hours: float,
    sleep_quality: str,
    study_method: str,
    internet_access: str,
):
    recs = []

    if study_hours < 4:
        recs.append("Increase study hours gradually (e.g., +1 hour/day or +5 hours/week) to improve predicted performance.")
    elif study_hours < 6:
        recs.append("Consider adding 1–2 extra study hours per week to reach the next performance band.")

    if class_attendance_pct < 70:
        recs.append("Increase attendance (aim 80–90%). Attendance is a consistent predictor of higher scores.")
    elif class_attendance_pct < 85:
        recs.append("If possible, lift attendance closer to 90% for more stable outcomes.")

    if sleep_hours < 6:
        recs.append("Increase sleep duration (target ~7–8 hours). Low sleep often reduces performance stability.")
    elif sleep_hours > 9:
        recs.append("Ensure sleep is consistent and balanced; extremely high sleep may signal fatigue/inefficient routine.")

    if sleep_quality in ["poor", "average"]:
        recs.append("Improve sleep quality (fixed bedtime, reduced screen time, lighter evening meals).")

    if study_method == "self-study":
        recs.append("If progress stalls, try 'mixed' or 'coaching' for structured practice and feedback.")
    if study_method == "online videos":
        recs.append("Combine videos with timed practice questions to convert learning into exam performance.")

    if internet_access == "no":
        recs.append("If possible, access offline resources (downloaded notes/past papers) to support consistent revision.")

    if not recs:
        recs.append("Maintain current routine. Focus on targeted revision and past-paper practice to maximise score.")

    return recs

# ============================
# Prediction helpers (robust to pipelines)
# ============================
def predict_row(model, df_row: pd.DataFrame, expected_cols: list | None):
    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in df_row.columns]
        if missing:
            raise ValueError(f"Missing columns required by pipeline: {missing}")
        df_row = df_row[expected_cols]
    pred = float(model.predict(df_row)[0])
    return float(np.clip(pred, 0, 100))

def predict_with_changes(model, base_row: dict, expected_cols: list | None, changes: dict):
    tmp = base_row.copy()
    tmp.update(changes)
    df = pd.DataFrame([tmp])
    df = add_features(df)
    return predict_row(model, df, expected_cols)

# ============================
# Reset / Clear form support
# ============================
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

DEFAULTS = {
    "age": 0,
    "study_hours": 0.0,
    "class_attendance": 0.0,  # % in UI
    "sleep_hours": 0.0,
    "gender": PH,
    "course": PH,
    "internet_access": PH,
    "sleep_quality": PH,
    "study_method": PH,
    "facility_rating": PH,
    "exam_difficulty": PH,
}

def reset_form():
    st.session_state.reset_counter += 1
    k = st.session_state.reset_counter

    st.session_state[f"age_{k}"] = DEFAULTS["age"]
    st.session_state[f"study_hours_{k}"] = DEFAULTS["study_hours"]
    st.session_state[f"class_attendance_{k}"] = DEFAULTS["class_attendance"]
    st.session_state[f"sleep_hours_{k}"] = DEFAULTS["sleep_hours"]

    st.session_state[f"gender_{k}"] = DEFAULTS["gender"]
    st.session_state[f"course_{k}"] = DEFAULTS["course"]
    st.session_state[f"internet_{k}"] = DEFAULTS["internet_access"]
    st.session_state[f"sleep_quality_{k}"] = DEFAULTS["sleep_quality"]
    st.session_state[f"study_method_{k}"] = DEFAULTS["study_method"]
    st.session_state[f"facility_rating_{k}"] = DEFAULTS["facility_rating"]
    st.session_state[f"exam_difficulty_{k}"] = DEFAULTS["exam_difficulty"]

# ============================
# Header
# ============================
st.title("🎓 Student Exam Score Predictor")
st.write("Decision-support app: predicts exam score and provides guidance to improve performance.")

# ✅ Robust model existence check
if not MODEL_PATH.exists():
    st.error(
        f"❌ Model not found at: `{MODEL_PATH}`\n\n"
        "Fix:\n"
        "1) Ensure Task E saved the model as `models/final_best_pipeline.joblib`\n"
        "2) Re-run the save-model cell in Task E notebook."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Try to read expected feature names (works for many sklearn estimators/pipelines)
expected_cols = None
try:
    expected_cols = list(model.feature_names_in_)
except Exception:
    expected_cols = None

# ============================
# Single prediction form
# ============================
st.markdown("---")
st.subheader("📋 Student Profile")

k = st.session_state.reset_counter

with st.form("predict_form"):
    st.subheader("📝 Input Student Details")

    age = st.number_input("Age", min_value=0, max_value=80, step=1, key=f"age_{k}")
    study_hours = st.number_input("Study hours", min_value=0.0, max_value=10.0, step=0.1, key=f"study_hours_{k}")
    class_attendance_pct = st.number_input("Class attendance (%)", min_value=0.0, max_value=100.0, step=1.0, key=f"class_attendance_{k}")
    sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=10.0, step=0.1, key=f"sleep_hours_{k}")

    gender = st.selectbox("Gender", GENDER_UI, index=0, key=f"gender_{k}")
    course = st.selectbox("Course", COURSE_UI, index=0, key=f"course_{k}")
    internet_access = st.selectbox("Internet access", INTERNET_UI, index=0, key=f"internet_{k}")
    sleep_quality = st.selectbox("Sleep quality", SLEEP_QUALITY_UI, index=0, key=f"sleep_quality_{k}")
    study_method = st.selectbox("Study method", STUDY_METHOD_UI, index=0, key=f"study_method_{k}")
    facility_rating = st.selectbox("Facility rating", FACILITY_UI, index=0, key=f"facility_rating_{k}")
    exam_difficulty = st.selectbox("Exam difficulty", DIFFICULTY_UI, index=0, key=f"exam_difficulty_{k}")

    col1, col2 = st.columns([1, 1])
    with col1:
        submitted = st.form_submit_button("🎯 Predict")
    with col2:
        clear_clicked = st.form_submit_button("🧹 Clear Inputs")

if clear_clicked:
    reset_form()
    st.rerun()

if submitted:
    if (gender == PH or course == PH or internet_access == PH or sleep_quality == PH or
        study_method == PH or facility_rating == PH or exam_difficulty == PH):
        st.warning("⚠️ Please select all dropdown fields (not '— Select —') before predicting.")
        st.stop()

    # ✅ IMPORTANT: Attendance is already percentage scale (40–100) in training data
    class_attendance = float(class_attendance_pct)

    base_row = {
        "age": int(age),
        "gender": gender,
        "course": course,
        "study_hours": float(study_hours),
        "class_attendance": class_attendance,
        "sleep_hours": float(sleep_hours),
        "internet_access": internet_access,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty,
    }

    input_df = pd.DataFrame([base_row])
    input_df = add_features(input_df)

    try:
        pred = predict_row(model, input_df, expected_cols)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    grade = score_to_grade(pred)
    msg = interpretation_text(pred)

    if pred >= 75:
        st.success(f"✅ Predicted Exam Score: **{pred:.2f} / 100**")
    elif pred >= 50:
        st.warning(f"⚠️ Predicted Exam Score: **{pred:.2f} / 100**")
    else:
        st.error(f"❌ Predicted Exam Score: **{pred:.2f} / 100**")

    st.info(f"📌 Predicted Grade: **{grade}**")
    st.write(f"**Interpretation:** {msg}")

    low = max(0.0, pred - CV_RMSE)
    high = min(100.0, pred + CV_RMSE)
    st.write(f"**Expected variation:** ~ **{pred:.2f} ± {CV_RMSE:.2f}** → **[{low:.1f}, {high:.1f}]**")

    st.subheader("✅ Recommendations to Improve Score")
    recs = build_recommendations(
        study_hours=float(study_hours),
        class_attendance_pct=float(class_attendance_pct),
        sleep_hours=float(sleep_hours),
        sleep_quality=sleep_quality,
        study_method=study_method,
        internet_access=internet_access,
    )
    for r in recs:
        st.write(f"• {r}")

    st.subheader("🔁 What-if Analysis (Simulation)")
    c1, c2, c3 = st.columns(3)

    with c1:
        sim1 = predict_with_changes(
            model, base_row, expected_cols,
            {"study_hours": min(float(study_hours) + 1.0, 10.0)}
        )
        st.metric("Study hours +1", f"{sim1:.2f}")

    with c2:
        sim2 = predict_with_changes(
            model, base_row, expected_cols,
            {"class_attendance": 95.0}
        )
        st.metric("Attendance = 95%", f"{sim2:.2f}")

    with c3:
        sim3 = predict_with_changes(
            model, base_row, expected_cols,
            {"sleep_hours": 8.0}
        )
        st.metric("Sleep = 8 hours", f"{sim3:.2f}")

    with st.expander("Show technical details (for report/viva)"):
        st.write("Engineered features:")
        if "study_effort" in input_df.columns:
            st.write(f"- study_effort = {float(input_df['study_effort'].iloc[0]):.2f}")
        if "sleep_study_ratio" in input_df.columns:
            st.write(f"- sleep_study_ratio = {float(input_df['sleep_study_ratio'].iloc[0]):.2f}")
        st.write(f"CV RMSE used for uncertainty: {CV_RMSE:.2f}")
        st.write(f"Using expected_cols alignment: {expected_cols is not None}")

# ============================
# Batch prediction (CSV upload)
# ============================
st.divider()
st.subheader("📂 Batch Prediction (Upload CSV)")
st.write("Upload a CSV with the same columns as test.csv (id optional).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    ids = df[ID_COL] if ID_COL in df.columns else None
    X = df.drop(columns=[ID_COL], errors="ignore").copy()

    # ✅ IMPORTANT: class_attendance already in percentage scale (40–100) — no conversion
    X = add_features(X)

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in X.columns]
        if missing:
            st.error(f"❌ Missing columns in uploaded CSV (after feature engineering): {missing}")
            st.stop()
        X_for_pred = X[expected_cols]
    else:
        X_for_pred = X

    try:
        preds = model.predict(X_for_pred)
        preds = np.clip(preds, 0, 100)
    except Exception as e:
        st.error(f"❌ Batch prediction failed: {e}")
        st.stop()

    if ids is not None:
        out = pd.DataFrame({ID_COL: ids, TARGET: preds})
    else:
        out = df.copy()
        out[TARGET] = preds

    st.write("✅ Preview predictions:")
    st.dataframe(out.head(10))

    st.write(f"**Pred summary:** min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Submission CSV",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )
