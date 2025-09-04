import pandas as pd
import re

CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "i'd": "i would",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "mustn't": "must not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "let's": "let us",
    "that's": "that is",
    "there's": "there is",
    "where's": "where is",
    "who's": "who is",
    "what's": "what is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "o'clock": "of the clock",
    "'cause": "because",
    "gimme": "give me",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}


def preprocess(df, save_path):
    from .etl import load_data
    df["comment_text"] = df["comment_text"].apply(preprocess_text)
    if save_path:
        df.to_csv(save_path, index=False)
    return load_data(save_path)


def preprocess_text(text: str) -> str:
    text = expand_contractions(text)
    text = clean_text(text)
    return text


def expand_contractions(text: str) -> str:
    for i, j in CONTRACTIONS.items():
        text = text.replace(i, j)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z\s?!]", " ", text)
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
