# Import necessary libraries
import mlflow
import mlflow.xgboost as mxgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
from sklearn.impute import SimpleImputer
import pandas as pd
from train_model import train_model
from featureEng import featureEng 
from dataPrep import dataPrep
# from ..features.dataPrep import dataPrep


def predict_model(model,X_test):
    # faire des prédictions sur les données de test avec le modèle entraîné
    y_pred = model.predict(X_test)
    df = pd.DataFrame(y_pred, columns=['Prediction'])
    df[df['Prediction'] == 1]
    print(df)
    return df

train_data, test_data = dataPrep('src\data\pplication_train.csv','src\data\pplication_test.csv')
train_data = featureEng(train_data)
test_data = featureEng( test_data)
model,X = train_model(train_data)

# supprimer les colonnes qui sontt pas dans le train 
test_data.drop(columns=[col for col in test_data.columns if col not in X.columns], inplace=True)

predict_model(model,test_data)
