import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv("iris.csv")

print(data.head())


X = data[['sepal_length',
          'sepal_width',
          'petal_length',
          'petal_width']]

y = data['species']


encoder = LabelEncoder()
y = encoder.fit_transform(y)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = KNeighborsClassifier(n_neighbors=5)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy * 100)


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


sample = pd.DataFrame({
    "sepal_length": [5.1],
    "sepal_width": [3.5],
    "petal_length": [1.4],
    "petal_width": [0.2]
})


sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)


print("\nPrediction:")
print(encoder.inverse_transform(prediction)[0])
