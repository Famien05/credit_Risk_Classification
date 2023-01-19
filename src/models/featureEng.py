from sklearn.impute import SimpleImputer
import pandas as pd


def featureEng(train_data):
    cat_list = []
    dis_num_list = []
    num_list = []
    # Sélectionner les données sans la colonne "target"
   

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

    return train_data