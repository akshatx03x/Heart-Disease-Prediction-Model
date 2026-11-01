import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress version warnings from pickle
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Human-Centered Clinical Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Risk Assessment Tool",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refined, deliberate styling — no neon gradients, no emoji spam, no AI tropes
st.markdown("""
<style>
    /* Typography & Canvas */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top application header */
    .app-header {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 18px;
        margin-bottom: 24px;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 4px;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Section group containers */
    .clinical-group {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 18px 20px 14px 20px;
        margin-bottom: 18px;
        background-color: rgba(128, 128, 128, 0.02);
    }
    .clinical-group-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 14px;
    }

    /* Helper labels under inputs */
    .field-hint {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: -6px;
        margin-bottom: 12px;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-optimal {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-borderline {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-alert {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Assessment Output Cards */
    .assessment-card {
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }
    .assessment-card-low {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #14532d;
    }
    .assessment-card-high {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #7f1d1d;
    }
    .assessment-headline {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .assessment-subline {
        font-size: 0.92rem;
        line-height: 1.5;
        opacity: 0.95;
    }

    /* Primary button refinement */
    div.stButton > button:first-child {
        font-weight: 600;
        border-radius: 6px;
        padding: 0.55rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Asset Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

# -----------------------------------------------------------------------------
# Clinical Test Cases (Human Data Presets)
# -----------------------------------------------------------------------------
SAMPLE_CASES = {
    "default": {
        "age": 45, "sex": "M", "chest_pain": "NAP", "resting_bp": 125,
        "cholesterol": 210, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 155, "exercise_angina": "N", "oldpeak": 0.5, "st_slope": "Up"
    },
    "case_healthy": {
        "age": 34, "sex": "F", "chest_pain": "ATA", "resting_bp": 112,
        "cholesterol": 175, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 178, "exercise_angina": "N", "oldpeak": 0.0, "st_slope": "Up"
    },
    "case_risk": {
        "age": 61, "sex": "M", "chest_pain": "ASY", "resting_bp": 158,
        "cholesterol": 275, "fasting_bs": 1, "resting_ecg": "ST",
        "max_hr": 115, "exercise_angina": "Y", "oldpeak": 2.4, "st_slope": "Flat"
    }
}

for k, v in SAMPLE_CASES["default"].items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_case(case_name):
    target = SAMPLE_CASES[case_name]
    for k, v in target.items():
        st.session_state[k] = v

# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">Cardiovascular Clinical Risk Evaluator</div>
    <div class="app-subtitle">
        Physician-referenced decision support tool assessing probability of coronary artery disease using a 15-variable k-Nearest Neighbors model.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Case Switcher
# -----------------------------------------------------------------------------
case_col1, case_col2, case_col3, case_col4 = st.columns([1.5, 1.5, 1.2, 3.8])
with case_col1:
    if st.button("Load Case 1: Low-Risk Baseline", use_container_width=True):
        load_case("case_healthy")
        st.rerun()
with case_col2:
    if st.button("Load Case 2: Elevated-Risk Case", use_container_width=True):
        load_case("case_risk")
        st.rerun()
with case_col3:
    if st.button("Reset Form", use_container_width=True):
        load_case("default")
        st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Two-Column Clinical Interface
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    # 1. Patient Demographics
    st.markdown('<div class="clinical-group">', unsafe_allow_html=True)
    st.markdown('<div class="clinical-group-title">1. Patient Profile</div>', unsafe_allow_html=True)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        age = st.slider("Patient Age (Years)", min_value=18, max_value=95, key="age")
        st.markdown(f'<div class="field-hint">Predicted maximum exertion heart rate: ~{220 - age} bpm</div>', unsafe_allow_html=True)
    with sub_col2:
        sex_selection = st.radio(
            "Biological Sex",
            options=["Male (M)", "Female (F)"],
            index=0 if st.session_state["sex"] == "M" else 1,
            horizontal=True
        )
        sex = "M" if "Male" in sex_selection else "F"
        st.markdown('<div class="field-hint">Accounting for sex-differentiated epidemiological baselines</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Vitals & Laboratory Findings
    st.markdown('<div class="clinical-group">', unsafe_allow_html=True)
    st.markdown('<div class="clinical-group-title">2. Hemodynamics & Blood Chemistry</div>', unsafe_allow_html=True)

    vitals_c1, vitals_c2 = st.columns(2)
    with vitals_c1:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=75,
            max_value=220,
            key="resting_bp",
            help="Arterial blood pressure measured at rest upon initial clinical examination."
        )
        if resting_bp < 120:
            bp_tag = '<span class="status-badge badge-optimal">Normal (< 120)</span>'
        elif resting_bp <= 129:
            bp_tag = '<span class="status-badge badge-borderline">Elevated (120–129)</span>'
        elif resting_bp <= 139:
            bp_tag = '<span class="status-badge badge-alert">Stage 1 HTN (130–139)</span>'
        else:
            bp_tag = '<span class="status-badge badge-alert">Stage 2 HTN (≥ 140)</span>'
        st.markdown(f'<div class="field-hint">ACC/AHA Guideline: {bp_tag}</div>', unsafe_allow_html=True)

        fasting_bs_choice = st.selectbox(
            "Fasting Blood Glucose",
            options=["Normal fasting glucose (≤ 120 mg/dL)", "Elevated fasting glucose (> 120 mg/dL)"],
            index=st.session_state["fasting_bs"],
            help="Venous blood glucose sample taken after an 8-hour fasting period."
        )
        fasting_bs = 1 if "> 120" in fasting_bs_choice else 0

    with vitals_c2:
        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=80,
            max_value=550,
            key="cholesterol",
            help="Total fasting serum cholesterol measurement."
        )
        if cholesterol < 200:
            chol_tag = '<span class="status-badge badge-optimal">Desirable (< 200)</span>'
        elif cholesterol <= 239:
            chol_tag = '<span class="status-badge badge-borderline">Borderline High (200–239)</span>'
        else:
            chol_tag = '<span class="status-badge badge-alert">High Risk (≥ 240)</span>'
        st.markdown(f'<div class="field-hint">NCEP Adult Treatment Panel: {chol_tag}</div>', unsafe_allow_html=True)

        resting_ecg_map = {
            "Normal": "Normal (No pathological deviations)",
            "ST": "ST-T Wave Abnormality (T-wave inversion / ST displacement > 0.05 mV)",
            "LVH": "Left Ventricular Hypertrophy (Estes criteria met)"
        }
        ecg_keys = list(resting_ecg_map.keys())
        current_ecg_idx = ecg_keys.index(st.session_state["resting_ecg"]) if st.session_state["resting_ecg"] in ecg_keys else 0
        resting_ecg = st.selectbox(
            "Resting Electrocardiogram (ECG)",
            options=ecg_keys,
            format_func=lambda x: resting_ecg_map[x],
            index=current_ecg_idx
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Stress Test & Symptomatology
    st.markdown('<div class="clinical-group">', unsafe_allow_html=True)
    st.markdown('<div class="clinical-group-title">3. Stress Response & Electrocardiography</div>', unsafe_allow_html=True)

    stress_c1, stress_c2 = st.columns(2)
    with stress_c1:
        chest_pain_map = {
            "ATA": "Atypical Angina (Non-classic presentation)",
            "NAP": "Non-Anginal Discomfort (Musculoskeletal or gastrointestinal)",
            "ASY": "Asymptomatic (Silent ischemia / no overt chest pain)",
            "TA": "Typical Angina (Classic substernal pressure with exertion)"
        }
        cp_keys = list(chest_pain_map.keys())
        current_cp_idx = cp_keys.index(st.session_state["chest_pain"]) if st.session_state["chest_pain"] in cp_keys else 0
        chest_pain = st.selectbox(
            "Chest Pain Classification",
            options=cp_keys,
            format_func=lambda x: chest_pain_map[x],
            index=current_cp_idx
        )

        angina_choice = st.radio(
            "Exercise-Induced Angina",
            options=["Negative (No angina on exertion)", "Positive (Angina provoked by exertion)"],
            index=0 if st.session_state["exercise_angina"] == "N" else 1,
            horizontal=True
        )
        exercise_angina = "Y" if "Positive" in angina_choice else "N"

    with stress_c2:
        max_hr = st.slider(
            "Max Achieved Heart Rate (bpm)",
            min_value=60,
            max_value=220,
            key="max_hr",
            help="Peak chronotropic response reached during standardized treadmill stress protocol."
        )
        hr_pct = int((max_hr / (220 - age)) * 100)
        st.markdown(f'<div class="field-hint">{hr_pct}% of theoretical age-predicted maximum</div>', unsafe_allow_html=True)

        slope_map = {
            "Up": "Upsloping (Physiologically typical)",
            "Flat": "Flat (Common indicator of exercise-induced ischemia)",
            "Down": "Downsloping (High positive predictive value for CAD)"
        }
        slope_keys = list(slope_map.keys())
        current_slope_idx = slope_keys.index(st.session_state["st_slope"]) if st.session_state["st_slope"] in slope_keys else 0
        st_slope = st.selectbox(
            "ST Segment Slope at Peak Exercise",
            options=slope_keys,
            format_func=lambda x: slope_map[x],
            index=current_slope_idx
        )

    oldpeak = st.slider(
        "Oldpeak: ST Depression (mm relative to rest)",
        min_value=0.0,
        max_value=5.5,
        step=0.1,
        key="oldpeak",
        help="Magnitude of horizontal or downsloping ST depression measured 80ms past the J-point."
    )
    if oldpeak < 1.0:
        st_tag = '<span class="status-badge badge-optimal">Baseline (< 1.0 mm)</span>'
    elif oldpeak <= 2.0:
        st_tag = '<span class="status-badge badge-borderline">Moderate (1.0–2.0 mm)</span>'
    else:
        st_tag = '<span class="status-badge badge-alert">Marked Depression (> 2.0 mm)</span>'
    st.markdown(f'<div class="field-hint">Clinical Interpretation: {st_tag}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Right Column: Evaluation & Diagnostic Summary Panel
# -----------------------------------------------------------------------------
with right_col:
    st.markdown('<div class="clinical-group">', unsafe_allow_html=True)
    st.markdown('<div class="clinical-group-title">Assessment Execution</div>', unsafe_allow_html=True)

    evaluate_button = st.button("Evaluate Patient Risk", type="primary", use_container_width=True)

    # Prepare model input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == 'M' else 0,
        'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
        'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
        'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Up': 1 if st_slope == 'Up' else 0,
    }

    input_df = pd.DataFrame([raw_input])[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    # Calculate inference
    pred_label = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    risk_prob = float(probabilities[1]) * 100.0

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if evaluate_button or "evaluated" in st.session_state:
        st.session_state["evaluated"] = True

        if pred_label == 1:
            st.markdown(f"""
            <div class="assessment-card assessment-card-high">
                <div class="assessment-headline">Elevated Cardiovascular Risk</div>
                <div class="assessment-subline">
                    Model evaluation indicates an estimated <b>{risk_prob:.1f}% risk probability</b> of significant coronary heart disease based on feature proximity to historical clinical cases.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assessment-card assessment-card-low">
                <div class="assessment-headline">Low Cardiovascular Risk</div>
                <div class="assessment-subline">
                    The patient's current profile aligns predominantly with non-disease cohorts, yielding a <b>{100.0 - risk_prob:.1f}% healthy probability</b> ({risk_prob:.1f}% risk estimate).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tabular Diagnostic Summary
        st.markdown("**Biomarker Diagnostic Breakdown**")
        
        breakdown_rows = [
            ("Resting Blood Pressure", f"{resting_bp} mm Hg", "< 120 mm Hg", "Normal" if resting_bp < 120 else ("Elevated" if resting_bp <= 129 else "Hypertensive")),
            ("Serum Cholesterol", f"{cholesterol} mg/dL", "< 200 mg/dL", "Desirable" if cholesterol < 200 else ("Borderline" if cholesterol <= 239 else "High")),
            ("Fasting Blood Sugar", "> 120 mg/dL" if fasting_bs == 1 else "≤ 120 mg/dL", "≤ 120 mg/dL", "Elevated" if fasting_bs == 1 else "Normal"),
            ("Peak Heart Rate", f"{max_hr} bpm", f"~{220 - age} bpm", f"{hr_pct}% of Max"),
            ("ST Depression (Oldpeak)", f"{oldpeak:.1f} mm", "< 1.0 mm", "Normal" if oldpeak < 1.0 else ("Moderate" if oldpeak <= 2.0 else "Marked")),
            ("Exertional Angina", "Present" if exercise_angina == "Y" else "Absent", "Absent", "Flagged" if exercise_angina == "Y" else "Normal")
        ]

        df_summary = pd.DataFrame(breakdown_rows, columns=["Clinical Metric", "Patient Value", "Reference Range", "Status"])
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Objective Clinical Guidance
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Clinical Observations**")
        
        observations = []
        if resting_bp >= 130:
            observations.append(f"• Resting blood pressure ({resting_bp} mm Hg) exceeds normal thresholds; recommend ambulatory BP monitoring.")
        if cholesterol >= 200:
            observations.append(f"• Total cholesterol ({cholesterol} mg/dL) is elevated; a complete lipid sub-fraction panel is advised.")
        if exercise_angina == "Y":
            observations.append("• Presence of exertional angina is a documented high-sensitivity marker for coronary stenosis.")
        if oldpeak >= 1.5:
            observations.append(f"• Exercise ST depression of {oldpeak:.1f} mm indicates potential subendocardial ischemia under physical load.")
        if not observations:
            observations.append("• All measured hemodynamic and electrocardiographic parameters fall within acceptable clinical ranges.")

        for obs in observations:
            st.markdown(obs)

    else:
        st.info("Select patient values or click a sample case above, then press **Evaluate Patient Risk** to generate an assessment.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Transparency Drawer
    with st.expander("Technical Model Documentation"):
        st.markdown("""
        **Methodology & Training Specifications**:
        - **Classifier**: $k$-Nearest Neighbors ($k=9$) with distance weighting ($w = 1/d$).
        - **Feature Preprocessing**: Multi-dimensional standard normal scaling (`StandardScaler`) on 15 categorical and continuous variables.
        - **Validation Performance**: Test accuracy: **88.6%** | F1-Score: **0.90** | Cross-validated mean: **83.6%**.
        - **Reference Cohort**: UCI Heart Failure Clinical Dataset (918 international patient records).
        """)

# -----------------------------------------------------------------------------
# Footer & Legal Notice
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
st.caption("Notice: This software is designed as an educational clinical decision-support demonstration. It is not intended to replace formal diagnostic procedures, clinician judgment, or standardized medical imaging.")
