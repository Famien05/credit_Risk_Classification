from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np 

def featureEng(train_data):
    """
    This function performs feature engineering on a training dataset by performing the following steps:
    1. Categorizing columns based on data type:
    - categorical
    - discrete numerical
    - continuous numerical
    2. Handling missing values by imputing missing values based on the following strategies:
    - categorical: 'most_frequent'
    - discrete numerical: 'most_frequent'
    - continuous numerical: 'median'
    3. One-hot encoding for categorical variables
    4. Correlation analysis and feature selection for target variable (if present in the dataset)

    less
    Copy code
    Parameters:
    train_data : pd.DataFrame
        The training dataset to be transformed.

    Returns:
    X : pd.DataFrame
        The feature matrix after transformations.
    y : pd.Series
        The target variable (if present in the dataset).
    """
    cat_list = []
    dis_num_list = []
    num_list = []
    # classifier les types de colonnes
    for i in train_data.columns.tolist():
        if train_data[i].dtype == 'object':
            cat_list.append(i)
        elif train_data[i].nunique() < 25:
            dis_num_list.append(i)
        else:
            num_list.append(i)
    #remplissage des donnees manquantes par des strategies median et most_frequent
    #Categorical
    train_data[cat_list] = SimpleImputer(strategy='most_frequent').fit_transform(train_data[cat_list])

    #Discrete
    train_data[dis_num_list] = SimpleImputer(strategy='most_frequent').fit_transform(train_data[dis_num_list])

    # continuous 
    train_data[num_list] = SimpleImputer(strategy='median').fit_transform(train_data[num_list])

    # one-hot encoding pour les variables categorielles
    train_data =pd.get_dummies(train_data, columns= cat_list)


     
        # Find correlations


    if 'TARGET' in train_data.columns:
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
    else:
        X = train_data
        y = 0

    return X,y 