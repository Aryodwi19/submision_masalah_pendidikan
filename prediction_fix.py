import numpy as np
import pandas as pd
import joblib
import os

# ===============================
# 1. LOAD MODEL & SCALER
# ===============================
# Note: Ensure model.pkl and scaler.pkl are in the same folder as this script
MODEL_PATH = "model_rf.pkl"
SCALER_PATH = "scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # ✅ FIX: Automatically extract the correct order of features from the scaler
    # This prevents 'feature_order not defined' errors
    try:
        feature_order = scaler.feature_names_in_.tolist()
    except AttributeError:
        # Fallback list if the scaler wasn't fitted with a DataFrame
        feature_order = [
            "Age", "Gender", "Scholarship_holder", "Tuition_fees_up_to_date",
            "Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved"
        ]
else:
    print("⚠️ Error: model.pkl or scaler.pkl not found!")

# ===============================
# 2. PREPROCESSING FUNCTION
# ===============================
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # Convert Categorical "Yes/No" to 1/0
    mapping_binary = {"Yes": 1, "No": 0}

    binary_cols = ['Scholarship_holder', 'Tuition_fees_up_to_date']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(mapping_binary)

    return df

# ===============================
# 3. PREDICTION FUNCTION
# ===============================
def predict(data_dict):
    # Process the dictionary into a DataFrame
    df = preprocess_input(data_dict)

    # ✅ Ensure all features required by the model exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with 0

    # ✅ Reorder columns to match the training data exactly
    df = df[feature_order]

    # Scaling
    df_scaled = scaler.transform(df)

    # Run Prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]

    # Map output to readable labels
    label_map = {0: "Dropout", 1: "Graduate"}
    label = label_map.get(prediction, "Unknown")

    return {
        "prediction": int(prediction),
        "label": label,
        "detail": {
            "Dropout (%)": round(float(probability[0] * 100), 2),
            "Graduate (%)": round(float(probability[1] * 100), 2)
        }
    }

# ===============================
# 4. EXECUTION TEST
# ===============================
if __name__ == "__main__":
    sample_input = {
        "Age": 20,
        "Gender": 1,
        "Scholarship_holder": "Yes",
        "Tuition_fees_up_to_date": "No",
        "Curricular_units_1st_sem_approved": 5,
        "Curricular_units_2nd_sem_approved": 4
    }

    try:
        result = predict(sample_input)
        print("\n--- Hasil Prediksi ---")
        print(f"Status: {result['label']}")
        print(f"Confidence: {result['detail']}")
    except Exception as e:
        print(f"Terjadi kesalahan saat prediksi: {e}")