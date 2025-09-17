from scripts.utils import count_features,  etl
import yaml

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

X, y = etl.prepare_data(path['processed_data'])
print("data processed and loaded")

X_train, _ , y_train, _= etl.split_data(X,y, config['data_split']['test_size'], config['data_split']['random_state'])
print("data split complete")

r = count_features.fit(X_train, y_train)
X_train =count_features.transform(r, X_train, y_train)