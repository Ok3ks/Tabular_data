from os.path import realpath,join,dirname
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
import json

BASE_DIR = realpath(dirname(__file__))
DATA_DIR = join(BASE_DIR, 'data')
COLUMNS = ["fixed_acidity","volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"]

#cannot be cythonized
def process_csv(files):
  """Function processes a csv given its file name. Function uses a predefined Data directory. Output is a list of a list """
  content = []
  with open(join(DATA_DIR, files), 'r') as f: 
    for line in f:
      line = re.sub("(\n)","",line)
      content.append(line.split(';'))
  return content

def load_config(filepath):
    """Loads config given a filepath"""
    with open(filepath, 'r') as ins:
        params = json.load(ins)
    return params

def shuffle_and_split(df, main_var, test_frac=0.2):
    train_df, test_df = train_test_split(df, random_state= 32, shuffle= True, test_size= test_frac, stratify = main_var)
    return train_df, test_df

def random_search(model, parameters = parameters['xgboost'], target:str ='quality', encoder = le, sect = [3.0, 4.0, 5.0], train_df = train_df, test_df = test_df):
    """Tunes Hyperparameters with Random Search CV, parameters should correspond to model type"""

    condition = train_df[target].isin(sect)
    condition_test = test_df[target].isin(sect)

    train_df = train_df.loc[condition]
    test_df = test_df.loc[condition_test]

    y = encoder.fit_transform(train_df[target])
    x = train_df.drop(columns = target)
    
    y_test = encoder.fit_transform(test_df[target])
    x_test = test_df.drop(columns = target)

    gridsearch = RandomizedSearchCV(model, param_distributions= parameters)
    output = []

    gridsearch.fit(x,y)
    print(gridsearch.best_params_)
    
    best_xgb_model = model.set_params(**gridsearch.best_params_)
    best_xgb_model.fit(x, y)
    pred = best_xgb_model.predict(x_test)

    f1 = f1_score(y_test, pred, average = "weighted")
    output.append((f1,best_xgb_model,))
    return output,pred


def outlier(df, target:str):
    """Computes outliers on a dataframe using > Q3 + 1.5iqr and < Q1 - 1.5iqr . Returns mask of true outliers,"""
    assert len(df) is not None, 'DataFrame is empty'
    assert target in list(df.columns), 'Target should be in passed dataframe'

    rng= np.quantile(df[target], 0.75) -  np.quantile(df[target], 0.25)
    upper_limit = 1.5*rng + np.quantile(df[target], 0.75)
    lower_limit = np.quantile(df[target], 0.25) - 1.5*rng
    print(lower_limit, upper_limit)

    mask = df[target].isin(np.arange(lower_limit, upper_limit, 0.5))
    return ~mask


