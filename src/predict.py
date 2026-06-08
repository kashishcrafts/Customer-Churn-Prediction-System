import joblib

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model and Scaler loaded successfully")

print(model)

print("Number of Features:", model.n_features_in_)

feature_names = joblib.load("models/feature_names.pkl")

print("Number of Saved Features:", len(feature_names))

print(feature_names[:5])

print(feature_names)

sample_customer = [12, 90, 1080, 4000] + [0] * 27

print("Total Features:", len(sample_customer))

import pandas as pd

sample_df = pd.DataFrame(
    [sample_customer],
    columns=feature_names
)

sample_scaled = scaler.transform(sample_df)

prediction = model.predict(sample_scaled)

print("Prediction:", prediction[0])