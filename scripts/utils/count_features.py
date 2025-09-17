from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class NBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.r_dict = {}
    def fit(self, X, y):
        X = self.vectorizer.fit_transform(X)
        alpha = 1

        for i in y.columns:
            pos = X[y[i] == 1].sum(axis=0).flatten() + alpha
            neg = X[y[i] == 0].sum(axis=0).flatten() + alpha

            total_pos = pos.sum()
            total_neg = neg.sum()

            P_w_pos = pos / total_pos
            P_w_neg = neg / total_neg

            r = np.log(P_w_pos / P_w_neg)
            self.r_dict[i] = r
        return self


    def transform(self, X):
        X = self.vectorizer.transform(X)
        transformed_X = {}
        for label in self.r_dict:
            transformed_X[label] = X.multiply(self.r_dict[label])
        return transformed_X
