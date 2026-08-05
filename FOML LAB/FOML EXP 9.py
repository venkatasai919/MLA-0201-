import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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


linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


linear_mse = mean_squared_error(y_test, linear_pred)

linear_r2 = r2_score(y_test, linear_pred)


poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)


poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)


poly_pred = poly_model.predict(X_test_poly)


poly_mse = mean_squared_error(y_test, poly_pred)

poly_r2 = r2_score(y_test, poly_pred)


print("\nLinear Regression Results")

print("Mean Squared Error:")
print(linear_mse)

print("R2 Score:")
print(linear_r2)


print("\nPolynomial Regression Results")

print("Mean Squared Error:")
print(poly_mse)

print("R2 Score:")
print(poly_r2)


sample = X_test[0].reshape(1, -1)

linear_result = linear_model.predict(sample)


sample_poly = poly.transform(sample)

poly_result = poly_model.predict(sample_poly)


print("\nLinear Regression Prediction:")
print(linear_result[0])


print("\nPolynomial Regression Prediction:")
print(poly_result[0])
