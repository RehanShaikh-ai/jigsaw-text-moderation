from scripts.utils import count_features,  etl
import yaml

with open("configs/configs.yaml") as f: 
    config = yaml.safe_load(f)

path = config['data']

X, y = etl.load_data(path['train_data'])

count_features.fit()