from utils import preprocess, etl 

import pandas as pd
df = etl.extract_data("data/raw/train.csv")
X, y = preprocess.preprocess(df)
print("DataFrame:")
print(pd.concat([X,y], axis=1))

X_train,X_val, y_train, y_val= etl.split_data(X,y)
print("Train DataFrame:")
print(X_train)
print("\nValidation DataFrame:")
print(X_val)