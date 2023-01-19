import pandas as pd

def dataPrep(chemin1,chemin2):
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