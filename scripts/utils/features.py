from sklearn.feature_extraction.text import TfidfVectorizer

def get_features(train, val, max_features = 5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train)
    X_val = vectorizer.transform(val)

    return X_train, X_val, vectorizer

