# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:29:02 2025

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Step 1: Generate dataset
np.random.seed(42)
values = np.random.rand(100)

labels = []
for i in range(50):  # Iterate over the first 50 values
    if values[i] <= 0.5:
        labels.append('Class1')
    else:
        labels.append('Class2')

# Append None for the remaining 50 values
labels += [None] * 50  

# Create DataFrame
data = {
    "Point": [f"x{i+1}" for i in range(100)],
    "Value": values,
    "Label": labels
}

df = pd.DataFrame(data)

# Split data into labeled and unlabeled
labeled_df = df[df["Label"].notna()].copy()
X_train = labeled_df[["Value"]].values
y_train = labeled_df["Label"].values

unlabeled_df = df[df["Label"].isna()].copy()
X_test = unlabeled_df[["Value"]].values #or to_numpy()

# Generate true labels for accuracy calculation
true_labels = ["Class1" if x <= 0.5 else "Class2" for x in values[50:]]

# Step 2: Perform KNN classification for different values of k
k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}
accuracies = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    # Store results
    results[k] = predictions
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100
    accuracies[k] = accuracy
    print(f"Accuracy for k={k}: {accuracy:.2f}%")
    
    # Assign predictions back to the original DataFrame
    df.loc[df["Label"].isna(), f"Label_k{k}"] = predictions
    
# Display the final predictions
print(df.tail(10))
