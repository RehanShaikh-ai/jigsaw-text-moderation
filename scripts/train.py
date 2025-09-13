from scripts.utils import tfidf_features
from utils import preprocess, etl
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import yaml
import os
import pandas as pd
import joblib

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

X, y = etl.prepare_data(path['processed_data'])
print("data processed and loaded")

X_train, _ , y_train, _= etl.split_data(X,y, config['data_split']['test_size'], config['data_split']['random_state'])
print("data split complete")

vectorizer = config['vectorizer']
X_train=tfidf_features.get_features(X_train, **vectorizer) 
print("features vectorized")
print(X_train, "\n", X_train.shape)

if config['model']['type'] == 'logistic_regression':
    param = config['model']['logistic_regression']
    model = LogisticRegression(**param)

elif config['model']['type'] == 'lightgbm':
    param = config['model']['lightgbm']
    model = lgb.LGBMClassifier(**param)

elif config['model']['type'] == 'nbsvm':
    param = config['model']['nbsvm']
    



model = OneVsRestClassifier(model)
model.fit(X_train, y_train)
joblib.dump(model, config['model']['model_path'])

print("model trained:", config['model']['type'])
