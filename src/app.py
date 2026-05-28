import streamlit as st
import requests
import pandas as pd
import numpy as np

# Config
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Depression vs ME/CFS Diagnostic Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for premium UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #4CAF50; color: white; height: 3em; font-size: 1.1rem; }
    .stButton>button:hover { background-color: #45a049; }
    h1, h2, h3 { color: #f0f2f6; }
    .prediction-card { padding: 20px; border-radius: 12px; background-color: #1f2937; margin-bottom: 20px; border: 1px solid #374151; }
    </style>
""", unsafe_allow_html=True)

st.title("Depression vs ME/CFS Diagnostic Assistant")
st.write("A clinical decision support interface powered by a trained classification model to separate overlapping symptoms.")

# Sidebar - API Health and Controls
st.sidebar.header("System Controls")
try:
    health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("● API Status: Healthy & Connected")
        model_info = health_resp.json()
        st.sidebar.info(f"Features: {model_info.get('n_features_expected', 'Unknown')}")
    else:
        st.sidebar.warning("● API Status: Warning (Backend up but model artifacts may be missing)")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ API Status: Offline (FastAPI is not running on port 8000)")

# Evidently monitoring dashboard link
st.sidebar.subheader("Analytics & Observability")
st.sidebar.markdown(f"[Open Evidently Monitoring Dashboard]({API_BASE_URL}/monitor)", unsafe_allow_html=True)

# Retraining button
if st.sidebar.button("Retrain Model from DB"):
    with st.spinner("Retraining model in backend..."):
        try:
            train_resp = requests.post(f"{API_BASE_URL}/train-model")
            if train_resp.status_code == 200:
                st.sidebar.success("Model retrained and updated successfully!")
            else:
                st.sidebar.error(f"Retraining failed: {train_resp.text}")
        except Exception as e:
            st.sidebar.error(f"Error connecting to backend: {e}")

# Main Tabs
tab1, tab2 = st.tabs(["Single Patient Prediction", "Batch Diagnostics"])

with tab1:
    st.header("Patient Symptom Profile")
    st.write("Provide symptom indicators based on clinical questionnaires and patient interviews.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics & Sleep")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        hours_of_sleep = st.slider("Hours of Sleep per Night", 3.0, 12.0, 7.0, 0.5)
        sleep_quality = st.slider("Sleep Quality Index (0-20, higher = worse)", 0, 20, 8)
        
    with col2:
        st.subheader("Primary Symptoms")
        depression_phq9 = st.slider("Depression Severity Score (PHQ-9)", 0, 27, 10)
        fatigue_fss = st.slider("Fatigue Severity Scale (FSS) Score", 9, 63, 35)
        brain_fog = st.slider("Brain Fog Level (0-10)", 0, 10, 4)
        physical_pain = st.slider("Physical Pain Score (0-10)", 0, 10, 3)

    with col3:
        st.subheader("PEM & Lifestyle")
        pem_present = st.selectbox("Post-Exertional Malaise (PEM) Present?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        pem_hours = st.number_input("PEM Duration (Hours after activity)", min_value=0.0, max_value=168.0, value=24.0)
        stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
        work_status = st.selectbox("Work Status", ["Working", "Partially working", "Not working"])
        social_activity = st.selectbox("Social Activity Level", ["High", "Medium", "Low"])
        exercise = st.selectbox("Exercise Frequency", ["Often", "Sometimes", "Rarely"])
        meditation = st.selectbox("Meditation or Mindfulness practice", ["Yes", "No"])

    # Predict button
    if st.button("Analyze Symptom Profile"):
        payload = [{
            "age": age,
            "gender": gender,
            "sleep_quality_index": float(sleep_quality),
            "brain_fog_level": float(brain_fog),
            "physical_pain_score": float(physical_pain),
            "stress_level": float(stress_level),
            "depression_phq9_score": float(depression_phq9),
            "fatigue_severity_scale_score": float(fatigue_fss),
            "pem_duration_hours": float(pem_hours),
            "hours_of_sleep_per_night": float(hours_of_sleep),
            "pem_present": int(pem_present),
            "work_status": work_status,
            "social_activity_level": social_activity,
            "exercise_frequency": exercise,
            "meditation_or_mindfulness": meditation
        }]
        
        with st.spinner("Analyzing data and generating model predictions..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    pred = result["predictions"][0]
                    probs = result["probabilities"]
                    conf = result["confidence"][0]
                    
                    st.success("Analysis Complete!")
                    
                    # Layout results
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        # Color coding based on prediction
                        border_color = "#3B82F6" # Blue for Depression
                        if pred == "ME/CFS":
                            border_color = "#EF4444" # Red
                        elif pred == "Both":
                            border_color = "#10B981" # Green
                            
                        st.markdown(f"""
                            <div class="prediction-card" style="border-left: 8px solid {border_color};">
                                <h4 style='margin:0;'>PREDICTED DIAGNOSIS</h4>
                                <h1 style='margin:10px 0; color:{border_color};'>{pred}</h1>
                                <p style='margin:0;'>Confidence level: <b>{conf*100:.1f}%</b></p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        st.subheader("Diagnosis Probability Distribution")
                        # Format as dataframe for chart
                        chart_data = pd.DataFrame({
                            'Diagnosis': ['Depression', 'ME/CFS', 'Both'],
                            'Probability': [probs['Depression'][0], probs['ME/CFS'][0], probs['Both'][0]]
                        })
                        st.bar_chart(chart_data, x='Diagnosis', y='Probability', color=border_color)
                        
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with API server: {e}")

with tab2:
    st.header("Batch Diagnostic Analytics")
    st.write("Upload a CSV file containing raw patient records matching the dataset schema to retrieve batch classifications.")
    
    uploaded_file = st.file_uploader("Upload Raw Patients CSV", type="csv")
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", df_batch.head(5))
            
            if st.button("Process Batch Predictions"):
                # Clean NaNs to None for JSON conversion
                records = df_batch.to_dict(orient="records")
                for rec in records:
                    for k, v in rec.items():
                        if pd.isna(v):
                            rec[k] = None
                            
                with st.spinner("Processing batch..."):
                    resp = requests.post(f"{API_BASE_URL}/predict", json=records)
                    if resp.status_code == 200:
                        result = resp.json()
                        df_batch["Predicted Diagnosis"] = result["predictions"]
                        df_batch["Confidence"] = result["confidence"]
                        
                        st.success("Batch diagnostics completed!")
                        st.write("Results Summary:")
                        
                        # Charts
                        diag_counts = df_batch["Predicted Diagnosis"].value_counts().reset_index()
                        diag_counts.columns = ["Diagnosis", "Count"]
                        
                        res_c1, res_c2 = st.columns(2)
                        with res_c1:
                            st.dataframe(df_batch[["Predicted Diagnosis", "Confidence"] + list(df_batch.columns[:-2])].head(10))
                        with res_c2:
                            st.bar_chart(diag_counts, x="Diagnosis", y="Count")
                            
                        # Download button
                        csv_data = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Diagnostic Results CSV",
                            data=csv_data,
                            file_name="patient_diagnostics_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Batch inference failed: {resp.text}")
        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")
