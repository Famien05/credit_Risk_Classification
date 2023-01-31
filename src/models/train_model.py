# Import necessary libraries
import mlflow 
import mlflow.xgboost as mxgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import shap
import matplotlib.pyplot as plt
import os

def train_model(X,y, seed, n_jobs, gamma, learning_rate, base_score, reg_alpha, reg_lambda, booster, eval_metric):
    """
    This function trains an XGBoost model using the input features `X` and labels `y`. The model is trained using the specified hyperparameters and evaluated using accuracy, precision, recall, and F1 score. The trained model, along with the hyperparameters and evaluation metrics, are logged to MLflow for tracking and comparison purposes.
    
    Parameters:
    - X (pandas.DataFrame or numpy.ndarray): Input features for the model
    - y (pandas.Series or numpy.ndarray): Labels for the model
    - seed (int): Seed used for random number generation
    - n_jobs (int): Number of parallel jobs to run (default=1)
    - gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree
    - learning_rate (float): Step size shrinkage used in updates to prevent overfitting
    - base_score (float): The initial prediction score of all instances, global bias
    - reg_alpha (float): L1 regularization term on weights
    - reg_lambda (float): L2 regularization term on weights
    - booster (str): Specifies which booster to use: gbtree, gblinear or dart
    - eval_metric (str): Evaluation metric(s) to be used for early stopping

    Returns:
    - model (xgboost.Booster or xgboost.XGBClassifier): Trained XGBoost model
    - X (pandas.DataFrame or numpy.ndarray): Input features for the model

    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Start an MLflow run
    with mlflow.start_run():
        # Create an XGBoost model
        model = xgb.XGBClassifier(seed = seed ,
            n_jobs=n_jobs,
            base_score=base_score,
            booster= booster,
            gamma=gamma,
            learning_rate= learning_rate,
            reg_alpha= reg_alpha,
            reg_lambda= reg_lambda,
            eval_metric=eval_metric)

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
      

        # Evaluate the model
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy: {:.2f}%".format(acc*100))
        print("Precision: {:.2f}%".format(prec*100))
        print("Recall: {:.2f}%".format(rec*100))
        print("F1 Score: {:.2f}".format(f1))

        # Log the model and evaluation metrics
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("base_score", base_score)
        mlflow.log_param("booster", booster)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("reg_alpha",reg_alpha)
        mlflow.log_param("reg_lambda", reg_lambda)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)


    return model,X
