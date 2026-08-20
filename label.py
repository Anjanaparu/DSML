import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score

file_path = Path(__file__).parent / 'label_knn_dataset.xlsx'
data = pd.read_excel(file_path)
print("Dataset:")
print(data)

x = data[['Feature1', 'Feature2']]
y = data['Label']
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
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
    [[5, 5]],
    columns=['Feature1', 'Feature2']
)

new_prediction = knn.predict(new_sample)

print("\n--------------------------------")
print("New Point: (5,5)")
print("Predicted Class:", new_prediction[0])
print("--------------------------------")
