import streamlit as st
import pandas as pd
import pickle
import numpy as np

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load pre-trained model and preprocessor
model = load_pickle("final_model.pkl")
preprocessor = load_pickle("preprocessor.pkl")

# Load dataset
df = pd.read_csv("Bank_Telemarketing.csv", sep=';')

target_column = "y"
input_columns = [col for col in df.columns if col != target_column]

st.title("📞 Bank Marketing Campaign Predictor")
st.write("Fill in the details to predict whether the client will subscribe to a term deposit.")

# User inputs
data = {}
for col in input_columns:
    if df[col].dtype == 'object':
        data[col] = st.selectbox(f"{col}", df[col].unique())
    else:
        data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# Convert user input into DataFrame
input_df = pd.DataFrame([data])

# Preprocess input data
processed_input = preprocessor.transform(input_df)

# Make prediction
prediction = model.predict(processed_input)[0]
predicted_label = "Yes" if prediction == "yes" else "No"

# Show result
st.subheader("Prediction Result")
st.success(f"Term Deposit Subscription Prediction: {predicted_label}")
