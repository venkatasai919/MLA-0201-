import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.stats import mode


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

X_scaled = scaler.fit_transform(X)


model = GaussianMixture(
    n_components=3,
    random_state=42
)


model.fit(X_scaled)


cluster = model.predict(X_scaled)


mapping = {}

for i in range(3):
    label = mode(y[cluster == i], keepdims=True).mode[0]
    mapping[i] = label


new_cluster = [mapping[i] for i in cluster]


print("\nCluster Values:")
print(cluster)


accuracy = accuracy_score(y, new_cluster)


print("\nAccuracy:")
print(accuracy * 100)


sample = pd.DataFrame({
    "sepal_length": [5.1],
    "sepal_width": [3.5],
    "petal_length": [1.4],
    "petal_width": [0.2]
})


sample_scaled = scaler.transform(sample)


prediction = model.predict(sample_scaled)


print("\nPrediction Cluster:")
print(prediction[0])
