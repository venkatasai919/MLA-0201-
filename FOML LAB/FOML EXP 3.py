# ==========================================
# EXPERIMENT 3
# ID3 Decision Tree Algorithm
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==========================================
# Step 1: Load Dataset
# ==========================================

data = pd.read_csv("PlayTennis.csv")

print("\n========== DATASET ==========\n")
print(data)

# ==========================================
# Step 2: Separate Features and Target
# ==========================================

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ==========================================
# Step 3: Encode Categorical Data
# ==========================================

encoders = {}
X_encoded = X.copy()

for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column])
    encoders[column] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ==========================================
# Step 4: Train ID3 Decision Tree
# ==========================================

model = DecisionTreeClassifier(
    criterion="entropy",   # ID3 uses Entropy
    random_state=0
)

model.fit(X_encoded, y_encoded)

# ==========================================
# Step 5: Display Decision Tree
# ==========================================

plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=target_encoder.classes_,
    filled=True,
    rounded=True
)

plt.title("ID3 Decision Tree")
plt.show()

# ==========================================
# Step 6: Predict New Sample
# ==========================================

sample = pd.DataFrame({
    "outlook": ["sunny"],
    "temp": ["cool"],
    "humidity": ["high"],
    "windy": [True]
})

sample_encoded = sample.copy()

for column in sample.columns:
    sample_encoded[column] = encoders[column].transform(sample[column])

prediction = model.predict(sample_encoded)

print("\n========== PREDICTION ==========")

print("New Sample:")
print(sample)

print("\nPredicted Class:")

print(target_encoder.inverse_transform(prediction)[0])
