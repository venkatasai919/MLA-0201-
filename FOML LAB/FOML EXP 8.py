import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("california_housing_test.csv")

print(data.head())


data = data.dropna()


X = data.drop(['median_house_value'], axis=1)

y = data['median_house_value']


scaler = StandardScaler()

X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


print("\nMean Squared Error:")
print(mse)


print("\nR2 Score:")
print(r2)


sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)


print("\nPrediction:")
print(prediction[0])
