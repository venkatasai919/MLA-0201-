import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv("data.csv")

print(data.head())


X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

y = data['diagnosis']


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


model = GaussianNB()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy * 100)


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)


print("\nPrediction:")
print(encoder.inverse_transform(prediction)[0])
