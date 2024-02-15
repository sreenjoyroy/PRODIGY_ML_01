import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
train=pd.read_csv("train.csv")
train.head()
test=pd.read_csv("test.csv")
test.head()
dir(train)
dir(test)
X_train=train[['LotArea','OverallQual','OverallCond','FullBath','HalfBath','TotRmsAbvGrd','BedroomAbvGr']]
X_train
Y_train=train[["SalePrice"]]
Y_train
X_test=test[['LotArea','OverallQual','OverallCond','FullBath','HalfBath','TotRmsAbvGrd','BedroomAbvGr']]
X_test
X_train.hist(bins=20,figsize=(15,15))
X_test.hist(bins=20,figsize=(15,15))
reg=LinearRegression()
reg.fit(X_train,Y_train)
reg.coef_
reg.predict(X_test)
reg.score(X_train,Y_train)

