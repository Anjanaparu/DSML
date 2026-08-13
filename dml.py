import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score

file_path = Path(__file__).parent / "Book1.xlsx"
data = pd.read_excel(file_path)

data.columns = data.columns.str.strip()

print("Excel Data:")
print(data)

X = data[['feature1', 'feature2']]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

precision = precision_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=1
)
print("Precision:", precision)

f1 = f1_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=1
)
print("F1 Score:", f1)

new_sample = pd.DataFrame(
    [[6, 4]],
    columns=['feature1', 'feature2']
)
new_prediction = knn.predict(new_sample)

print("\nTomato is a:", new_prediction[0])