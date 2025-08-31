from utils import preprocess, etl, features

import pandas as pd

df = etl.extract_data("data/raw/train.csv")
X, y = preprocess.preprocess(df)
print("DataFrame:")
print(pd.concat([X,y], axis=1))

X_train,X_val, y_train, y_val= etl.split_data(X,y)
X_train, X_val, tfidf_model =features.get_features(X_train, X_val) 

