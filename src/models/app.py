# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.xgboost as mxgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
from sklearn.impute import SimpleImputer
import pandas as pd
from dataPrep import dataPrep
from train_model import train_model
from featureEng import featureEng 
from dataPrep import dataPrep
import os
from flask.cli import FlaskGroup
# from ..features.dataPrep import dataPrep
import click
import subprocess
import time

app = Flask(__name__)





def predict_model(model,X_test):
        """Predict target values for X_test using the trained model.
    
        Parameters
        ----------
        model : object
            The trained model.
        X_test : pandas dataframe
            Test data with features.
            
        Returns
        -------
        y_pred : numpy ndarray
            Predicted target values for X_test.
        """
        current_dir = os.path.abspath(os.path.dirname(__file__))
        output_folder = os.path.join(current_dir, '..', '..', 'output', 'prediction')
        y_pred = model.predict(X_test)
        df = pd.DataFrame(y_pred, columns=['Prediction'])
        # saving as a CSV file
        df.to_csv(os.path.join(output_folder, 'prediction.csv'), sep =',')
        return y_pred



def predict(train_path, test_path, seed, n_jobs, gamma, learning_rate, base_score, reg_alpha, reg_lambda, booster, eval_metric):
    """Predict target values for test data using a trained model.
    
    Parameters
    ----------
    train_path : str
        Path to training data file.
    test_path : str
        Path to test data file.
    seed : int
        Seed for random number generation.
    n_jobs : int
        Number of parallel jobs to run.
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    learning_rate : float
        Step size shrinkage used in update to prevents overfitting. 
    base_score : float
        The initial prediction score of all instances, global bias.
    reg_alpha : float
        L1 regularization term on weights.
    reg_lambda : float
        L2 regularization term on weights.
    booster : str
        Booster type to use, either gbtree, gblinear or dart.
    eval_metric : str
        Evaluation metric to use for early stopping.
    
    Returns
    -------
    y_pred : numpy ndarray
        Predicted target values for test data.
    """
    # faire des prédictions sur les données de test avec le modèle entraîné
    print(learning_rate)
    train_data, test_data = dataPrep(train_path,test_path)

    # appel des fonctions de feature engineering 
    X,y = featureEng(train_data)
    test_data,a = featureEng( test_data)

    # appel de la fonction d'entrainement
    model,X = train_model(X, y, seed, n_jobs, gamma, learning_rate, base_score, reg_alpha, reg_lambda, booster, eval_metric)
    
    # supprimer les colonnes qui sontt pas dans le data train donc qui ne sont pas necessaires 
    test_data.drop(columns=[col for col in test_data.columns if col not in X.columns], inplace=True)
    current_dir = os.path.abspath(os.path.dirname(__file__))
    output_folder = os.path.join(current_dir, '..', '..', 'output', 'prediction')
    y_pred = model.predict(test_data)
    df = pd.DataFrame(y_pred, columns=['Prediction'])
    # saving as a CSV file
    df.to_csv(os.path.join(output_folder, 'prediction.csv'), sep =',')
    # appel de la fonction de prediction  
    print(y_pred)
    return y_pred 

@app.route('/')
def accueil():
    return render_template('index.html')



@app.route('/flask_predict', methods=['POST'])
def flask_predict():
    """
    flask_predict()

    This function handles the prediction process. It accepts parameters from the URL query string:

    train_path: file path of the training data (default: src/data/application_train.csv)
    test_path: file path of the test data (default: src/data/application_tEST.csv)
    seed: seed for the random number generator (default: 123)
    n_jobs: number of parallel jobs to run (default: -1)
    gamma: minimum loss reduction required to make a further partition on a leaf node of the tree (default: 0.3)
    learning_rate: step size shrinkage used in update to prevents overfitting (default: 0.1)
    base_score: the initial prediction score of all instances (default: 0.2)
    reg_alpha: L1 regularization term on weights (default: 1)
    reg_lambda: L2 regularization term on weights (default: 0.5)
    booster: type of model to use (default: 'gbtree')
    eval_metric: evaluation metric to use (default: 'mlogloss')
    It calls the predict() function with these parameters and returns the prediction result to the prediction.html template.
   
    train_path = request.args.get('train_path') or 'src/data/application_train.csv'
    test_path = request.args.get('test_path') or 'src/data/application_tEST.csv'
    seed = request.args.get('seed') or 123
    n_jobs = request.args.get('n_jobs') or -1
    gamma = request.args.get('gamma') or 0.3
    learning_rate = request.args.get('learning_rate') or 0.1
    base_score = request.args.get('base_score') or 0.2
    reg_alpha = request.args.get('reg_alpha') or 1
    reg_lambda = request.args.get('reg_lambda') or 0.5
    booster = request.args.get('booster') or 'gbtree'
    eval_metric = request.args.get('eval_metric') or 'mlogloss'
     """
     
    train_path = request.form.get('train_path', 'src/data/application_train.csv') 
    test_path = request.form.get('test_path','src/data/application_tEST.csv')
    seed = request.form.get('seed', 123, type=int) 
    n_jobs = request.form.get('n_jobs', -1, type=int) 
    gamma = request.form.get('gamma', 0.3, type=float) 
    learning_rate = request.form.get('learning_rate', 0.1, type=float)
    base_score = request.form.get('base_score', 0.2, type=float) 
    reg_alpha = request.form.get('reg_alpha', 1, type=float) 
    reg_lambda = request.form.get('reg_lambda', 0.5, type=float) 
    booster = 'gbtree' 
    eval_metric ='mlogloss'
    print(eval_metric)
    print(booster)
    prediction = predict(train_path, test_path, seed, n_jobs, gamma, learning_rate, base_score, reg_alpha, reg_lambda, booster, eval_metric)
    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
     app.run(port=8000, debug=True)
