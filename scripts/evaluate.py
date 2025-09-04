from .utils import metrics, etl, preprocess
import joblib
import yaml
import pandas
import os

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']


_, X_test , _, y_test= etl.split_data(X,y, config['data_split']['test_size'], config['data_split']['random_state'] )


model = joblib.load(config['evaluation']['saved_model'])
y_pred = model.predict(X_test)
