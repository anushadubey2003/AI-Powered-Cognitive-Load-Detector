import joblib
import numpy as np

model = joblib.load("models/cognitive_model.pkl")
le = joblib.load("models/label_encoder.pkl")

def predict_load(hrv, eye_movement, skin_conductance):
    features = np.array([[hrv, eye_movement, skin_conductance]])
    prediction = model.predict(features)[0]
    label = le.inverse_transform([prediction])[0]
    return label
