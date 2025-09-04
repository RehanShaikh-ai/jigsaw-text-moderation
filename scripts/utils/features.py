from sklearn.feature_extraction.text import TfidfVectorizer

def get_features(X, max_features = 5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(X)

    return X, vectorizer

