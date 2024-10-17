<h2>Project Title:</h2>
<p><strong>End to End Machine Learning Student Performance Prediction</strong></p>

<h2>Project Overview:</h2>
<p>
    This project involves building, training, and deploying a machine learning model using Streamlit, 
    a Python-based framework for creating interactive web applications. The goal of the project is to 
    predict student math scores based on a variety of categorical and numerical features, such as 
    gender, parental education level, and other academic scores (e.g., reading and writing scores). 
    The project demonstrates the full machine learning lifecycle, from data preprocessing and feature 
    engineering to model training, evaluation, and deployment.
</p>

<h2>Key Features:</h2>
<ul>
    <li>
        <strong>Data Preprocessing and Feature Engineering:</strong><br>
        The dataset consists of both categorical and numerical features. Techniques such as one-hot 
        encoding and normalization are used to preprocess the data. Additionally, missing values are 
        handled, and new features are engineered to improve model performance.
    </li>
    <li>
        <strong>Model Training and Evaluation:</strong><br>
        Several machine learning models are trained, including:<br>
        - Random Forest Regressor<br>
        - Decision Tree Regressor<br>
        - Gradient Boosting Regressor<br>
        - Linear Regression<br>
        - XGBoost Regressor<br>
        - CatBoost Regressor<br>
        - AdaBoost Regressor<br>
        Each model is hyperparameter-tuned using techniques like GridSearchCV, and the models are evaluated 
        using performance metrics such as R-squared (R²) to determine the best model for the task.
    </li>
    <li>
        <strong>Custom Exception Handling:</strong><br>
        A custom exception handling module (CustomeExceptionClass) is implemented to ensure robust error 
        handling during model training, deployment, and prediction.
    </li>
    <li>
        <strong>Model Deployment:</strong><br>
        The trained model is deployed using Streamlit, allowing users to input student data (such as 
        gender, parental education, reading scores, etc.) and receive predictions on their potential math 
        score. The web app provides an intuitive and user-friendly interface for interacting with the model.
    </li>
    <li>
        <strong>Interactive Frontend:</strong><br>
        The web app is designed with a simple and clean interface, where users can enter student attributes 
        and instantly get predictions. The app also displays error messages in case of incorrect input 
        formats or missing data.
    </li>
</ul>

<h2>Technologies Used:</h2>
<p>
    <strong>Libraries:</strong> Streamlit, Scikit-learn, XGBoost, CatBoost, Pandas, NumPy<br>
    <strong>Deployment:</strong> Streamlit for creating and hosting the web interface<br>
    <strong>Model Evaluation:</strong> R² score for assessing model performance
</p>
