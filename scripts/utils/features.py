from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def get_features(X, max_char = 20000, max_words = 10000):
    char_vectorizer = TfidfVectorizer(max_features=max_char,
                                 ngram_range=(3,5),
                                 sublinear_tf=True,
                                 analyzer='char'
                                 )
    word_vectorizer = TfidfVectorizer(max_features=max_words,
                                      ngram_range=(1,2),
                                      sublinear_tf=True,
                                      analyzer='word'
                                      )
    vectorizer = FeatureUnion([("char",char_vectorizer), ("word",word_vectorizer)])
    vectorizer.fit_transform(X)
    return X, vectorizer

