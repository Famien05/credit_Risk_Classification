projetData
==============================

Credit Risk Classification Dataset
Is Customer is Risky or Not Risky ?

Membres
------------
- Amoin Famien Le Roi
- Ngueyap Nguikam Godelieve


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── application_train.csv
    │   ├── application_test.csv
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── mlruns          <- Track parameters & metrics of your model and display the results in your local 
    │                           mlflow UI
    │                         
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── Output            <- output of project
    │   └── figures        <- Generated graphics and figures 
    │   └── prediction        <- contains the prediction csv file
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │              
    │   ├── templates         <- code html for deployment of flask server 
    │   │   ├── index.html
    │   │   └── prediction.html
    │   │
    │   ├── models         <- Scripts to prepare and train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── dataPrep_model.py
    │   │   └── featureEng.py
    │   │   ├── app.py      <- Flask server with prediction's function                    
    │   │   └── train_model.py
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.
    │   
    │   
    └── MLproject      <- Package your code in a reusable and reproducible model format with ML Flow projects




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
