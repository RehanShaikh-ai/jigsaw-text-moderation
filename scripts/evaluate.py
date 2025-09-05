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
model = joblib.load("models/model2.joblib")

y_pred = model.predict(X_test)
y_pred_proba  = model.predict_proba(X_test)
metrics.report(y_test, y_pred)
metrics.roc_auc(y_test, y_pred)
metrics.pr_curve(y_test, y_pred)

