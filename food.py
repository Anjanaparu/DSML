import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score,f1_score
data=pd.read_excel('food_knn_dataset.xlsx')
print(data)
x=data[['Feature1','Feature2']]
y=data['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
precision=precision_score(y_test,y_pred,average='macro',zero_division=1)
print("precision:",precision)
f1=f1_score(y_test,y_pred,average='macro',zero_division=1)
print("f1:",f1)
new_sample=pd.DataFrame([[6,4]],columns=['Feature1','Feature2'])
new_prediction=knn.predict(new_sample)
print("\nTomato is a:",new_prediction)