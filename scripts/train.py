from utils import preprocess, etl 

df = etl.load_data("data/raw/train.csv")
df = preprocess.preprocess(df)