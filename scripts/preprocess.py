import pandas as pd
df = pd.read_csv("data\\raw\\train.csv")
df.head()
labels = df[['toxic','severe_toxic',	'obscene',	'threat',	'insult',	'identity_hate']]
comments = df['comment_text']
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

comments = [i.lower() for i in comments]

contractions = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will","i'd": "i would", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "it'll": "it will", "we'll": "we will", "they'll": "they will", "can't": "cannot", "won't": "will not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "couldn't": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "mustn't": "must not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "let's": "let us", "that's": "that is", "there's": "there is", "where's": "where is",
    "who's": "who is", "what's": "what is", "when's": "when is", "why's": "why is", "how's": "how is",
    "o'clock": "of the clock", "'cause": "because", "gimme": "give me", "gonna": "going to", "wanna": "want to",
    "gotta": "got to"
}
for index,text in zip(range(len(comments)),comments): 
    for i,j in contractions.items():
        text = text.replace(i,j)
        comments[index] = text
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()

comments = [re.sub(r"[^a-z\s]"," ",i) for i in comments]
comments = [re.sub(r"[\n\t\r]"," ",i) for i in comments]
comments= [re.sub(r"\s+"," ",i).strip() for i in comments]
temp = [lemmatizer.lemmatize(i) for i in comments]


import spacy
nlp = spacy.load("en_core_web_sm")

doc = [nlp(i) for i in comments[0:5]]
lemmatized = [token.lemma_ for token in comments]
print(" ".join(lemmatized))

