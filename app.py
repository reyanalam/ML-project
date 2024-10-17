import streamlit as st
import joblib
import pandas as pd
import sys
from src.logger import logging
from src.exception import CustomeExceptionClass
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PredictionApp:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        try:
            self.preprocessor = joblib.load(os.path.join('artifacts', 'preprocessor.pkl'))
            model = joblib.load('artifacts/model.pkl')  # Adjust the path as needed
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            raise CustomeExceptionClass(e,sys)

    def user_input_features(self):
        st.sidebar.header("Input Features")
        
        # Categorical features
        gender = st.sidebar.selectbox("Gender", options=['male', 'female'])
        race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", options=['group A', 'group B', 'group C', 'group D', 'group E'])
        parental_level_of_education = st.sidebar.selectbox("Parental Level of Education", options=[
            "bachelor's degree", 'some college', "master's degree", "associate's degree",
            'high school', 'some high school'])
        lunch = st.sidebar.selectbox("Lunch Type", options=['standard', 'free/reduced'])
        test_preparation_course = st.sidebar.selectbox("Test Preparation Course", options=['none', 'completed'])

        # Numerical features
        reading_score = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, step=1)
        writing_score = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, step=1)

        return pd.DataFrame({
            'gender': [gender],
            'race_ethnicity': [race_ethnicity],
            'parental_level_of_education': [parental_level_of_education],
            'lunch': [lunch],
            'test_preparation_course': [test_preparation_course],
            'reading_score': [reading_score],
            'writing_score': [writing_score]
        })

    def predict(self, input_data):
        try:
            input_data_transformed = self.preprocessor.transform(input_data)
            prediction = self.model.predict(input_data_transformed)
            return prediction[0]
        except Exception as e:
            raise CustomeExceptionClass(e,sys)

    def run(self):
        st.title("🚀 Prediction Model App")
        st.markdown("""
            Welcome to the Prediction Model App! 
            This application allows you to input features and receive predictions for Math Score.
            Please fill in the fields below and click on 'Predict'.
        """)

        input_data = self.user_input_features()

        # Button for prediction
        if st.button("Predict"):
            prediction = self.predict(input_data)
            if prediction is not None:
                st.success(f"### The predicted Math Score is: **{prediction}**")

        # Add footer or additional information
        st.markdown("---")
        st.markdown("""
            Developed by [Reyan Alam](https://yourwebsite.com) | 
            [GitHub](https://github.com/reyanalam) | 
            [LinkedIn](https://www.linkedin.com/in/reyanalam)
        """)

if __name__ == "__main__":
    app = PredictionApp()
    app.run()
