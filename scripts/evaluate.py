from utils import metrics, etl, preprocess
import joblib
import yaml
import numpy as np
import os

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

X, y = etl.prepare_data(path['processed_data'])
print("data procesed and loaded")

_, X_test , _, y_test= etl.split_data(X,y, config['data_split']['test_size'], config['data_split']['random_state'])
print("data split complete")

vectorizer = joblib.load("models/vectorizer.joblib")
X_test = vectorizer.transform(X_test)
print("features vectorized")

model = joblib.load(config['evaluation']['saved_model'])
print("model loaded")

y_pred = model.predict(X_test)

metrics.report(y_test, y_pred)
# metrics.roc_auc(y_test, y_pred)
# metrics.pr_curve(y_test, y_pred)

