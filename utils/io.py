import os
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def save_file(comments, labels, save_path):
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame({"comment_text": comments})
    df = pd.concat([df, labels.reset_index(drop=True)], axis=1)
    df.to_csv(save_path, index=False)
