from utils import metrics, etl, preprocess, features
import joblib
import yaml
import pandas
import os

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

X, y = etl.prepare_data(path['processed_data'])
_, X_test , _, y_test= etl.split_data(X,y, config['data_split']['test_size'], config['data_split']['random_state'])

features.get_features(X_test)
print(X_test)


