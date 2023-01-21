# Import necessary libraries
import mlflow 
import mlflow.xgboost as mxgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 

seed = 123
n_jobs=-1
base_score=0.2
booster= 'gbtree'
gamma= 0.3
learning_rate= 0.1
reg_alpha= 1
reg_lambda= 0.50
eval_metric='mlogloss'

def train_model(train_data):
    # Find correlations
    corr = train_data.corr()
    # Select highly correlated features
    high_corr = corr[((corr > 0.05) | (corr < -0.05)) & (corr < 1) ]
    # For each feature, list its correlated feature
    correlated_columns = {}
    for col in high_corr.columns:
        correlated_features = high_corr.columns[(~high_corr[col].isna())].tolist()
        correlated_features = list(set(correlated_features).difference(set(correlated_columns.keys())))
        correlated_columns[col] = correlated_features
    # From correlated features with TARGET, select their top 3 correlated features
    selected_features = correlated_columns["TARGET"]
    for feature in selected_features.copy():
        correlated_correlated_features = high_corr[feature].abs().sort_values(ascending=False)

        features_to_select = correlated_correlated_features[correlated_correlated_features < 90][:3].index.tolist()
        selected_features.extend(features_to_select)
    X = train_data.loc[:, np.unique(selected_features)].drop(columns="TARGET")
    y = train_data.loc[:, "TARGET"]



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
