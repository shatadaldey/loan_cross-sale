# import pickle
import pandas as pd
import numpy as np
import json
# from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("./data/loan_base.csv")

def json_func(cin):
    score_data = data[data.ID == int(cin)]
    dict_obj = score_data.to_dict('records')
    dump = json.dumps(dict_obj)
    return dump

# print(json_func(117))