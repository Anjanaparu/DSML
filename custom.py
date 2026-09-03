from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets,linear_model
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_excel('knn/aptitude.xlsx')
X=df[['X1']]
Y=df['Y1']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
clf=LinearRegression()
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
print(Y_pred)
print("Y=",clf.intercept_,"+",clf.coef_,"*X")
newvalues = float(input("Enter the value of X: "))
Y_pred=clf.intercept_+clf.coef_*newvalues
print("Predicted value of Y for X=",newvalues,"is",Y_pred)