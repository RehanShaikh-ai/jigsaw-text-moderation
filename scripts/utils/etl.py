import os
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_data(path):
    return pd.read_csv(path)


def save_file(comments, labels, save_path="data\\processed"):
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, "processed.csv")

    # Combine
    df = pd.DataFrame({"comment_text": comments})
    df = pd.concat([df, labels.reset_index(drop=True)], axis=1)

    # Save
    df.to_csv(full_path, index=False)
    print(f"[INFO] File saved at: {full_path}")


def split_data(df, test_size=0.2, random_state=42):
    train, val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, val
