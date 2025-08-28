import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import wordnet

df = pd.read_csv("../data/raw/train.csv")
comments = df['comment_text']
labels = df[['toxic',	'severe_toxic',	'obscene',	'threat'	,'insult'	,'identity_hate']]


nltk.download(stopwords)
nltk.download(wordnet)

