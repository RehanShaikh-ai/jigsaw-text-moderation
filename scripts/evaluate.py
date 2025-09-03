from .utils import metrics, etl
import joblib
import yaml
import pandas

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

test_set = config['data']

X_test, y_test = etl.load_test(test_set['test_data'], test_set['test_labels'])

model = joblib.load(config['evaluation']['saved_model'])
y_pred = model.predict(X_test)
