import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

rf_model = joblib.load(r'Customer-churn\churn_model_rf.pkl')

new_data = {
    'CreditScore': [650],
    'Gender': ['Male'],  # Male/Female
    'Age': [42],
    'Tenure': [3],
    'Balance': [50000],
    'NumOfProducts': [2],
    'HasCrCard': [1],  # 1: Yes, 0: No
    'IsActiveMember': [1],  # 1: Active, 0: Inactive
    'EstimatedSalary': [70000],
    'Geography_France': [1],  # One-hot encoded columns
    'Geography_Germany': [0]
}

new_data_df = pd.DataFrame(new_data)

label_encoder = LabelEncoder()
new_data_df['Gender'] = label_encoder.fit_transform(new_data_df['Gender'])
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data_df)

# Make predictions
prediction = rf_model.predict(new_data_scaled)
probabilities = rf_model.predict_proba(new_data_scaled)

# Display the results
print("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
print(f"Probability of Churn: {probabilities[0][1]:.2f}")
