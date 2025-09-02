from utils import preprocess, etl, features
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
import yaml
import os
import pandas as pd

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

if os.path.exists(path['processed_data']):
    X,y = etl.load_data(path['processed_data'])
    print("processed data available")
    print(X)
else:
    df = etl.extract_data(path['train_data'])
    X,y = preprocess.preprocess(df, path['processed_data'])
    print(X)

X_train,X_val, y_train, y_val= etl.split_data(X,y, config['data_split']['val_size'], config['data_split']['random_state'] )
X_train, X_val, tfidf_model =features.get_features(X_train, X_val) 

if config['model']['type'] == 'logistic_regression':
    param = config['model']['logistic_regression']
    model = LogisticRegression(**param)

elif config['model']['type'] == 'svm':
    param = config['model']['svm']
    model = SVC(**param) 

elif config['model']['type'] == 'naive_bayes':
    param = config['model']['naive_bayes']
    model = ComplementNB(**param)

model = OneVsRestClassifier(model)
model.fit(X_train, y_train)

print("model trained:", config['model']['type'])

from sklearn.metrics import classification_report

y_pred = model.predict(X_val)

print(classification_report(y_val, y_pred, target_names=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], zero_division=0, digits=4))


