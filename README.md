# End to End Machine Learning Student Performance Prediction

Please wait while the GIF is loading. Thank you.

![bandicam2024-11-1421-08-11-837-ezgif com-speed](https://github.com/user-attachments/assets/29558989-7157-4834-ae74-11ce218e1b61)

## Project Overview:

This project involves building, training, and deploying a machine learning model to predict student math scores. The prediction is based on a variety of categorical and numerical features such as gender, parental education level, and other academic scores (e.g., reading and writing scores). The project demonstrates the full machine learning lifecycle, from data preprocessing and feature engineering to model training, evaluation, and deployment.

## Key Features:

- **Data Preprocessing and Feature Engineering:**  
  The dataset consists of both categorical and numerical features. Techniques such as one-hot encoding and normalization are used to preprocess the data. Additionally, missing values are handled, and new features are engineered to improve model performance.

- **Model Training and Evaluation:**  
  Several machine learning models are trained, including:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor  

  Each model is hyperparameter-tuned using techniques like GridSearchCV, and the models are evaluated using performance metrics such as R-squared (R²) to determine the best model for the task.

- **Custom Exception Handling:**  
  A custom exception handling module (`CustomeExceptionClass`) is implemented to ensure robust error handling during model training, deployment, and prediction.

- **Model Deployment:**  
  The trained model is deployed using Streamlit, allowing users to input student data (such as gender, parental education, reading scores, etc.) and receive predictions on their potential math score. The web app provides an intuitive and user-friendly interface for interacting with the model.

- **Interactive Frontend:**  
  The web app is designed with a simple and clean interface, where users can enter student attributes and instantly get predictions. The app also displays error messages in case of incorrect input formats or missing data.

## Technologies Used:

- **Libraries:** Streamlit, Scikit-learn, XGBoost, CatBoost, Pandas, NumPy
- **Deployment:** Streamlit for creating and hosting the web interface
- **Model Evaluation:** R² score for assessing model performance

## Installation and Setup:

Follow these steps to clone and set up the project on your local machine:

1. Clone the repository:

   ```bash
   git clone [https://github.com/yourusername/end-to-end-student-performance-prediction.git](https://github.com/reyanalam/ML-project.git)
