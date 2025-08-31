from utils import preprocess, etl, features
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import pandas as pd

df = etl.extract_data("data/raw/train.csv")
X, y = preprocess.preprocess(df)
print("DataFrame:")
print(pd.concat([X,y], axis=1))

X_train,X_val, y_train, y_val= etl.split_data(X,y)
X_train, X_val, tfidf_model =features.get_features(X_train, X_val) 

log_reg = LogisticRegression(max_iter=200, solver='liblinear')
model = OneVsRestClassifier(log_reg)

model.fit(X_train, y_train)