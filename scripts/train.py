from utils import preprocess, etl, features
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import yaml
import os
import pandas as pd

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

if os.path.exists(path['processed_data']):
    X,y = etl.load_data(path['processed_data'])
    
else:
    df = etl.extract_data(path['train_data'])
    X,y = preprocess.preprocess(df)
    print(df)

X_train,X_val, y_train, y_val= etl.split_data(X,y)
X_train, X_val, tfidf_model =features.get_features(X_train, X_val) 

if config['model']['type'] == 'logistic_regression':
    param = config['model']['logistic_regression']
    model = LogisticRegression(**param)

elif config['model']['type'] == 'svm':
    param = config['model']['svm']
    model = LinearSVC(**param) 

elif config['model']['type'] == 'naive_bayes':
    param = config['model']['naive_bayes']
    model = MultinomialNB(**param)

model = OneVsRestClassifier(model)
model.fit(X_train, y_train)

print("model trained")