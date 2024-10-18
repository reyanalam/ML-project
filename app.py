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
            raise CustomeExceptionClass(e, sys)

    def user_input_features(self):
        # Creating an expander section for input fields for better UI
        with st.sidebar.expander("🛠️ Input Features", expanded=True):
            # Categorical features with streamlined selectboxes
            gender = st.selectbox("👤 Gender", options=['male', 'female'])
            race_ethnicity = st.selectbox("🌍 Race/Ethnicity", options=['group A', 'group B', 'group C', 'group D', 'group E'])
            parental_level_of_education = st.selectbox("🎓 Parental Level of Education", options=[
                "bachelor's degree", 'some college', "master's degree", "associate's degree",
                'high school', 'some high school'])
            lunch = st.selectbox("🍽️ Lunch Type", options=['standard', 'free/reduced'])
            test_preparation_course = st.selectbox("📝 Test Preparation Course", options=['none', 'completed'])

            # Numerical features with sliders for an interactive UX
            reading_score = st.slider("📖 Reading Score", min_value=0, max_value=100, value=50)
            writing_score = st.slider("✍️ Writing Score", min_value=0, max_value=100, value=50)

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
            raise CustomeExceptionClass(e, sys)

    def run(self):
        # Add a header with an emoji to make it visually engaging
        st.title("🚀 Math Score Prediction App")
        
        st.markdown(
            """
            Welcome to the **Math Score Prediction App**! 🎉
            
            This app predicts a student's Math score based on several factors. 
            Fill in the details in the sidebar and click on the 'Predict' button below.
            """
        )
        
        # Use columns to organize the layout
        col1, col2 = st.columns([2, 3])  # Wider column on the right for predictions

        # Input section on the left column
        with col1:
            st.subheader("🔧 Input Section")
            input_data = self.user_input_features()

        # Prediction section on the right column
        with col2:
            st.subheader("🎯 Prediction Result")
            # Show prediction result when the button is clicked
            if st.button("🔮 Predict"):
                prediction = self.predict(input_data)
                if prediction is not None:
                    st.success(f"### The predicted Math Score is: **{prediction}**")
                    st.balloons()  # Add fun balloons when prediction is made

        # Add a sidebar image with a valid URL or local file path
        # Replace with your actual image URL or local path
        st.sidebar.image("https://your_image_url.com/image.png", caption="Math Score Predictor", use_column_width=True)

        # Add a horizontal line and footer information
        st.markdown("---")
        st.markdown(
            """
            **Developed by [Reyan Alam](https://yourwebsite.com)** | 
            [GitHub](https://github.com/reyanalam) | 
            [LinkedIn](https://www.linkedin.com/in/reyanalam)
            """
        )

if __name__ == "__main__":
    app = PredictionApp()
    app.run()
