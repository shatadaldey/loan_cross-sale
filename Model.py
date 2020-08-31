import pickle
import pandas as pd
import numpy as np
import json
import xgboost as xgb
# from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

with open('./pickle_files/model_NN','rb') as model_cv:
    model_cv = pickle.load(model_cv)

data = pd.read_csv("./data/loan_base.csv")

def json_func(cin):   
    score_data = data[data.ID == int(cin)]
    data1 = score_data[['Age', 'Experience','Income', 'Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']]
    score_data['Random'] = model_cv.predict_proba(data1)
    score_data = pd.DataFrame(score_data)
    score_data['Random'] = round(score_data['Random']*100)
    if  score_data['Securities Account'].any()==1:
        score_data['Securities Account']='Yes'
    else:
        score_data['Securities Account']='No'
    if  score_data['Online'].any()==1:
        score_data['Online']='Yes'
    else:
        score_data['Online']='No'
    dict_obj = score_data.to_dict('records')
    dump = json.dumps(dict_obj)
    return dump

#print(json_func(117))
