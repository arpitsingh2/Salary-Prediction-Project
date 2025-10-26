#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

# --- Data ---
data = {
    'YearsExperience': [1.1, 1.5, 2.0, 2.2, 3.0, 3.2, 3.9, 4.5, 5.1, 6.0],
    'Salary': [20000, 30000, 45000, 49525, 60150, 65400, 75000, 81111, 87938, 93088]
}
df = pd.DataFrame(data)

X = df[['YearsExperience']]
Y = df[['Salary']]

# Train Linear Regression model on full data
model = LinearRegression()
model.fit(X, Y)

# Calculate R² and Adjusted R² (optional, for your reference)
Y_pred = model.predict(X)
r2 = r2_score(Y, Y_pred)
n = len(Y)
k = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

# --- Streamlit UI ---
st.title("Salary Prediction Project")
st.write("Predict salary based on years of experience.")

years_exp = st.number_input(
    "Enter Years of Experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.1
)

if st.button("Predict Salary"):
    predicted_salary = model.predict([[years_exp]]).item()
    st.success(f"Predicted Salary: ₹{predicted_salary:,.1f}")

# Optional: show model accuracy
if st.checkbox("Show model accuracy metrics"):
    st.write(f"R²: {r2:.4f}")
    st.write(f"Adjusted R²: {adj_r2:.4f}")
