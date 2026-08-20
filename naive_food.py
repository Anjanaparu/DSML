import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load dataset
data = pd.read_excel('food_naivebayes_dataset.xlsx')

# Remove extra spaces
data.columns = data.columns.str.strip()
data['Class'] = data['Class'].str.strip()

print("Dataset Loaded Successfully:\n")
print(data)

# Features and target
X = data[['Feature1', 'Feature2']]
y = data['Class']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train model
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# Prediction
y_pred = gnb.predict(x_test)

print("\nActual values:", y_test.values)
print("Predicted values:", y_pred)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average='macro', zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average='macro', zero_division=0
)

print("\nModel Evaluation Metrics:")
print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("F1-score:", round(f1, 2))

# Predict new data
x_new = pd.DataFrame(
    [[6, 4]],
    columns=['Feature1', 'Feature2']
)

y_new = gnb.predict(x_new)

print("\nPredicted class for [6,4] (Tomato):", y_new[0])