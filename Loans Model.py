{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import xgboost as xgb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "xl = pd.ExcelFile('datasets_48024_87370_Bank_Personal_Loan_Modelling.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Description', 'Data', 'OTT']"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xl.sheet_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_excel('datasets_48024_87370_Bank_Personal_Loan_Modelling.xlsx',sheet_name='Data')\n",
    "desc = pd.read_excel('datasets_48024_87370_Bank_Personal_Loan_Modelling.xlsx',sheet_name='Description')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4477 entries, 0 to 4476\n",
      "Data columns (total 14 columns):\n",
      " #   Column              Non-Null Count  Dtype  \n",
      "---  ------              --------------  -----  \n",
      " 0   ID                  4477 non-null   int64  \n",
      " 1   Age                 4477 non-null   int64  \n",
      " 2   Experience          4477 non-null   int64  \n",
      " 3   Income              4477 non-null   int64  \n",
      " 4   ZIP Code            4477 non-null   int64  \n",
      " 5   Family              4477 non-null   int64  \n",
      " 6   CCAvg               4477 non-null   float64\n",
      " 7   Education           4477 non-null   int64  \n",
      " 8   Mortgage            4477 non-null   int64  \n",
      " 9   Personal Loan       4477 non-null   int64  \n",
      " 10  Securities Account  4477 non-null   int64  \n",
      " 11  CD Account          4477 non-null   int64  \n",
      " 12  Online              4477 non-null   int64  \n",
      " 13  CreditCard          4477 non-null   int64  \n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 489.8 KB\n"
     ]
    }
   ],
   "source": [
    "data.describe()\n",
    "data.head()\n",
    "data.shape\n",
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data[['Age', 'Experience','Income', 'Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']]\n",
    "y=data[['Personal Loan']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "D_train = xgb.DMatrix(X_train, label=Y_train)\n",
    "D_test = xgb.DMatrix(X_test, label=Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "param = {\n",
    "    'eta': 0.2, \n",
    "    'max_depth': 3,  \n",
    "    'objective': 'multi:softprob',  \n",
    "    'num_class': 3} \n",
    "\n",
    "steps = 100  # The number of training iterations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = xgb.train(param, D_train, steps)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('model_NN','wb') as picklefile:\n",
    "    pickle.dump(model,picklefile)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision = 0.9729166666666667\n",
      "Recall = 0.9484615004692053\n",
      "Accuracy = 0.984375\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import precision_score, recall_score, accuracy_score\n",
    "\n",
    "preds = model.predict(D_test)\n",
    "best_preds = np.asarray([np.argmax(line) for line in preds])\n",
    "\n",
    "print(\"Precision = {}\".format(precision_score(Y_test, best_preds, average='macro')))\n",
    "print(\"Recall = {}\".format(recall_score(Y_test, best_preds, average='macro')))\n",
    "print(\"Accuracy = {}\".format(accuracy_score(Y_test, best_preds)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
