# ==========================================
# Candidate Elimination Algorithm
# ==========================================

import pandas as pd
import numpy as np

# Read Dataset
data = pd.read_csv("ENJOYSPORT.csv")

print("\nTraining Data:\n")
print(data)

# Separate Features and Target
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

# ------------------------------------------
# Candidate Elimination Function
# ------------------------------------------

def candidate_elimination(concepts, target):

    specific_h = concepts[0].copy()

    general_h = [["?" for i in range(len(specific_h))]
                 for j in range(len(specific_h))]

    print("\nInitial Specific Hypothesis:")
    print(specific_h)

    print("\nInitial General Hypothesis:")
    print(np.array(general_h))

    for i, h in enumerate(concepts):

        # Positive Example
        if target[i] == 1:

            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:

                    specific_h[x] = "?"

                    general_h[x][x] = "?"

        # Negative Example
        else:

            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:

                    general_h[x][x] = specific_h[x]

                else:

                    general_h[x][x] = "?"

    # Remove duplicate rows
    general_h = [row for row in general_h if row != ["?"] * len(specific_h)]

    return specific_h, general_h


# Run Algorithm
S, G = candidate_elimination(concepts, target)

print("\n===================================")
print("Final Specific Hypothesis (S)")
print("===================================")
print(S)

print("\n===================================")
print("Final General Hypothesis (G)")
print("===================================")

for g in G:
    print(g)
