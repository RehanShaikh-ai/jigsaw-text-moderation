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
y_pred_proba  = model.predict_proba(X_test)

thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
y_pred_custom = (y_pred_proba >= thresholds).astype(int)

metrics.report(y_test, y_pred_custom)
metrics.roc_auc(y_test, y_pred_custom)
metrics.pr_curve(y_test, y_pred_custom)

