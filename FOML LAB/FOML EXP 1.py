import pandas as pd

data = pd.read_csv(
    r"C:\Users\anji3\Downloads\archive (11).zip",
    compression="zip"
)

print(data)

attributes = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

hypothesis = None

for i in range(len(target)):
    if target[i] == 1:
        if hypothesis is None:
            hypothesis = attributes[i].copy()
        else:
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = "?"

print("\nMost Specific Hypothesis:")
print(hypothesis)
