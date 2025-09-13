from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def transform(X, y):
    r_dict= get_count_features(X,y)
    transformed_X = {}
    for label in r_dict: 
        transformed_X[label] = X.multiply(r_dict[label])
    return transformed_X, y 

def get_count_features(X, y):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    alpha = 1
    r_dict = {}
    for i in y.columns:     
        pos = X[y[i]==1].sum(axis=0).flatten() + alpha
        neg = X[y[i]==0].sum(axis=0).flatten() + alpha
        
        total_pos = pos.sum()
        total_neg = neg.sum()

        P_w_pos = pos / total_pos
        P_w_neg = neg / total_neg

        r = np.log(P_w_pos/P_w_neg)
        r_dict[i] = r
    return r_dict
        
