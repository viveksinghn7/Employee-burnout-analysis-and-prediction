# Importing the necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Loading the trained KNN regression model
model = joblib.load('models/knn_regression.pkl')

# Loading the scaler used for data preprocessing
scaler = joblib.load('models/scaler.pkl')

def main():
    st.title("Employee Burnout Prediction")
    st.write("Enter employee details")

    # Creating columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        company_type = st.selectbox("Company Type", ["Service", "Product"])
    with col3:
        wfh = st.selectbox("Work From Home", ["Yes", "No"])

    # Other input fields
    designation = st.slider("Designation", 0, 5, 1)
    resource_allocation = st.slider("Resource Allocation", 1, 10, 1)
    mental_fatigue_score = st.slider("Mental Fatigue Score", 0.0, 10.0, 0.1)

    # Converting inputs to a DataFrame
    input_data = pd.DataFrame({
        'Designation': [designation],
        'Resource Allocation': [resource_allocation],
        'Mental Fatigue Score': [mental_fatigue_score],
        'Company Type_Service': [1 if company_type == "Service" else 0],
        'WFH Setup Available_Yes': [1 if wfh == "Yes" else 0],
        'Gender_Male': [1 if gender == "Male" else 0]
    })

    # Ensuring the order of columns matches the training data
    input_data = input_data[['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'Company Type_Service', 'WFH Setup Available_Yes', 'Gender_Male']]

    # Transforming the input data using the fitted scaler
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # Predicting burnout rate
    if st.button("Predict Burnout Rate"):
        prediction_scaled = model.predict(input_data_scaled)
        st.write(f"Predicted Burnout Rate: {prediction_scaled[0]:.2f}")

        # Determining burnout type based on the predicted rate
        if prediction_scaled[0] <= 0.25:
            type = "Minimal burnout, employee likely feels well-rested and engaged with their work."
        elif prediction_scaled[0] <= 0.50:
            type = "Moderate burnout, showing some signs of stress and fatigue, but manageable with adjustments."
        elif prediction_scaled[0] <= 0.75:
            type = "High burnout, significant symptoms like emotional exhaustion, reduced productivity, and difficulty concentrating."
        else:
            type = "Severe burnout, potentially impacting personal life and requiring immediate intervention."

        st.write(f"{type}")


if __name__ == "__main__":
    main()