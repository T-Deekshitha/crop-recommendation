
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("crop_recommendation_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, le = load_model()

st.markdown("""
    <style>
    .main {
        background-color: #f0f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stDownloadButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Crop Recommendation System")
st.write("Upload a CSV file or manually input values to get crop suggestions.")

# === Option 2: Single Input Prediction ===
st.header("✏️ Predict for Single Input")

with st.form("single_input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        P = st.number_input("Phosphorous (P)", min_value=0.0, step=1.0, value=40.0)
    with col2:
        K = st.number_input("Potassium (K)", min_value=0.0, step=1.0, value=40.0)
    with col3:
        temperature = st.number_input("Temperature (°C)", step=1.0, value=25.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0, value=60.0)
    with col5:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=1.0, value=6.0)
    with col6:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0, value=100.0)

    submitted = st.form_submit_button("🔍 Predict Crop")

    if submitted:
        try:
            single_input = pd.DataFrame([[P, K, temperature, humidity, ph, rainfall]],
                                        columns=["P", "K", "temperature", "humidity", "ph", "rainfall"])
            scaled_input = scaler.transform(single_input)
            prediction = model.predict(scaled_input)
            predicted_crop = le.inverse_transform(prediction)[0]

            st.success(f"🌱 Recommended Crop: **{predicted_crop}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# === Option 1: File Upload ===
st.header("📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        features = ["P", "K", "temperature", "humidity", "ph", "rainfall"]

        if not all(feature in data.columns for feature in features):
            st.error(f"CSV must contain columns: {features}")
        else:
            input_data = data[features]
            scaled_data = scaler.transform(input_data)
            predictions = model.predict(scaled_data)
            predicted_labels = le.inverse_transform(predictions)

            data["Predicted Crop"] = predicted_labels
            st.success("✅ Batch prediction completed!")
            st.write(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predicted CSV", csv, "predicted_output.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
