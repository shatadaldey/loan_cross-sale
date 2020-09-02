import pickle
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import os
from google.cloud import bigquery
# from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

with open('./pickle_files/model_NN','rb') as model_cv:
    model_cv = pickle.load(model_cv)

#data = pd.read_csv("./data/loan_base.csv")

def json_func(cin):
    # Fetch details from DB
    # 1. Establish credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tenacious-veld-282811-0965308e30e8.json"
    
    # 2. Establish BQ client
    client = bigquery.Client()

    # 3. Query
    sql_query = """
	    SELECT 
		    *
	    FROM 
		    `tenacious-veld-282811.Laons_data.Laons` as A
	    WHERE 
		    A.ID = {cin}
    """

    # 4. Fetch results
    score_data = (client.query(sql_query.format(cin = cin))).to_dataframe()
    #print(score_data)
    #score_data = data[data.ID == int(cin)]
    data1 = score_data[['Age', 'Experience','Income', 'Family','CCAvg','Education','Mortgage','Securities_Account','CD_Account','Online','CreditCard']]
    data1.rename(columns = {'Securities_Account':'Securities Account', 'CD_Account':'CD Account'}, inplace = True) 
    score_data['Random'] = model_cv.predict_proba(data1)
    score_data = pd.DataFrame(score_data)
    score_data['Random'] = round(score_data['Random']*100)
    score_data['Loan_takeup']=np.where(score_data['Random']>=80,'Very Likely',np.where(score_data['Random']>=50,'Likely','Less Likely'))
    if  score_data['Securities_Account'].any()==1:
        score_data['Securities_Account']='Yes'
    else:
        score_data['Securities_Account']='No'
    if  score_data['Online'].any()==1:
        score_data['Online']='Yes'
    else:
        score_data['Online']='No'
    dict_obj = score_data.to_dict('records')
    dump = json.dumps(dict_obj)
    return dump

#print(json_func(117))
