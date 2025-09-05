import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from utils import preprocess

with open("configs/configs.yaml") as f:
    config = yaml.safe_load(f)

path = config['data']
def extract_data(path):
    return pd.read_csv(path)
   

def load_data(path, downsample):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    
    if downsample:
        toxic = df[df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)!=0]
        non_toxic = df[df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)==0]
        non_toxic = non_toxic.sample(30000, random_state=42)
        df = pd.concat([toxic, non_toxic], axis=0) 
    comments = df['comment_text']
    labels = df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ]
    
    return comments, labels

def prepare_data(path):
    if os.path.exists(path):
        X,y = load_data(path)
        print("processed data available")
    else:
        df = extract_data(config['data']['train_data'])
        X,y = preprocess.preprocess(df, path)
    return X, y

def split_data(comments, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        comments, labels, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    