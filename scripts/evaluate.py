from utils import metrics, etl, preprocess
import joblib
import yaml
import pandas
import os

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

X, y = etl.prepare_data(path['processed_data'])
_, X_test , _, y_test= etl.split_data(X,y, config['data_split']['test_size'], config['data_split']['random_state'])

vectorizer = joblib.load("models/vectorizer.joblib")
X_test = vectorizer.transform(X_test)




