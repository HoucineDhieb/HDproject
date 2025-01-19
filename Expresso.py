import webbrowser

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from ydata_profiling import ProfileReport
import numpy as np
import io

# Load Dataset
expresso = pd.read_csv('Expresso_churn_dataset.csv')

# Drop the user_id column if it exists
if 'user_id' in expresso.columns:
    expresso.drop(columns=['user_id'], inplace=True)

st.title("Churn Predictor with User Inputs")

# Data Exploration
st.subheader('Data Exploration')
buffer = io.StringIO()
expresso.info(buf=buffer)
info_string = buffer.getvalue()
st.text(info_string)
st.write('Data Description:', expresso.describe())
st.write('Missing Values:', expresso.isnull().sum())
st.write('Data Duplicates:', expresso.duplicated().sum())

#Generate profiling report
#profiler = ProfileReport(expresso,title="Expresso Profiling")
#profiler.to_file("Expresso.html")
#webbrowser.open("Expresso.html")
# Data Preprocessing
st.subheader('Data Preprocessing')
if st.checkbox('Drop Missing Values'):
    expresso.dropna(inplace=True)
if st.checkbox('Drop Duplicates'):
    expresso.drop_duplicates(inplace=True)

# Filter only numerical features
numerical_features = expresso.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Split and Train the Model
if st.checkbox('Split and train the Model'):
    # Use only numerical features
    X = expresso[numerical_features].drop('CHURN', axis=1)
    y = expresso['CHURN']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    rf_model = RandomForestClassifier(max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.5f}")

# Generate Predictions with User Inputs
if st.checkbox('Generate Predictions with User Inputs'):
    st.write("Enter values for the numerical features below:")

    if 'rf_model' in globals() and 'X' in globals():
        # Dynamically create input fields for numerical features
        input_data = {}
        with st.form(key='input_form'):
            for feature in X.columns:
                input_data[feature] = st.number_input(f'{feature}', value=0.0, format="%.2f")
            submit_button = st.form_submit_button(label='Confirm Inputs')

        # Check if inputs are confirmed
        if submit_button:
            # Convert inputs into a DataFrame
            input_df = pd.DataFrame([input_data])

            if input_df.isnull().values.any():
                st.error('Please fill in all input fields.')
            else:
                # Generate predictions
                prediction = rf_model.predict(input_df)[0]
                prediction_proba = rf_model.predict_proba(input_df)

                # Display results
                st.success('Inputs are valid.')
                st.write('Prediction:', 'CHURN' if prediction else 'NO CHURN')
                st.write('Prediction Probabilities:', prediction_proba)

# Evaluation Report
if st.checkbox('Evaluation Report') and 'rf_model' in globals():
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))
    st.write(f"Evaluation Accuracy: {accuracy:.5f}")

# Cross-Validation
if st.checkbox('Running Cross Validation') and 'rf_model' in globals():
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rf_model, X, y, cv=5)
    st.write(f"Cross-Validation Scores: {scores}")
    st.write(f"Mean Accuracy: {scores.mean():.5f}")
    st.write(f"Standard Deviation: {scores.std():.5f}")
