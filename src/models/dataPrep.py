import pandas as pd

def dataPrep(chemin1,chemin2):
    """
    Cleans the train and test data by handling missing values and removing duplicate rows.

    Inputs:
    - train_data: pandas DataFrame, the training data to be cleaned.
    - test_data: pandas DataFrame, the testing data to be cleaned.

    Outputs:
    - Tuple of two pandas DataFrames: (train_data, test_data) where
    - train_data: pandas DataFrame, the cleaned training data.
    - test_data: pandas DataFrame, the cleaned testing data.

    Algorithm:
    1. Check for missing values in the data and drop columns with more than 40% of missing values.
    2. Check for duplicate rows in the train data and drop them.
    3. Return the cleaned training and testing data.
    """
    # recuperer les datasets
    train_data =pd.read_csv(chemin1)
    test_data = pd.read_csv(chemin2)
    # Supprimer des colonnes avec 60 des donnees manquantes 
    missing_values = (train_data.isnull().sum() / len(train_data)) * 100
    missing_values = missing_values.drop(missing_values[missing_values == 0].index).sort_values(ascending=False)
    print(missing_values)
    train_data.dropna(thresh=train_data.shape[0] * 0.4, axis=1, inplace=True) 
    # Check for duplicate rows
    print(train_data.duplicated().sum())



    return train_data, test_data 