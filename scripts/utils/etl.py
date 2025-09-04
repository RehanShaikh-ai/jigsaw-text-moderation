import os
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess
import yaml


with open("configs/configs.yaml") as f:
    config = yaml.safe_load(f)

path = config['data']
def extract_data(path):
    return pd.read_csv(path)

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
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
        df = extract_data(path['train_data'])
        X,y = preprocess.preprocess(df, path)
    return X, y

def save_file(comments, labels, save_path="data/processed"):
    if not os.path.exists(path['processed_data']):
            
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, "processed.csv")

        # Combine
        df = pd.DataFrame({"comment_text": comments})
        df = pd.concat([df, labels.reset_index(drop=True)], axis=1)
        # Save
        df.to_csv(full_path, index=False)
        print(f"[INFO] File saved at: {full_path}")


def split_data(comments, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        comments, labels, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    