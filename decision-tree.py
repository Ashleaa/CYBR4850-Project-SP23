import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

# Load the dataset
data = pd.read_csv('projectDataset.csv')
data = data.replace([np.inf, -np.inf, np.nan], 0)

# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert any float integers to integers
X = X.astype(int)

# Initialize the linear SVM classifier with default parameters
clf = LinearSVC()

# Fit the linear SVM classifier to the data
clf.fit(X, y)

# Compute the absolute coefficients of the linear SVM
abs_coefs = np.abs(clf.coef_[0])

# Compute the sum of the absolute coefficients
total_coef = np.sum(abs_coefs)

# Get the top 5 features based on their coefficients
top_5 = np.argsort(abs_coefs)[-5:]

# Print the top 5 features and their coefficients as percentages
print("Top 5 features and their coefficients:")
for i in top_5:
    coef_percent = abs_coefs[i] / total_coef * 100
    print(f"{data.columns[i]}: {coef_percent:.2f}%")

# Compute cross-validation scores
scores = cross_val_score(clf, X, y, cv=10)

# Print the accuracy scores and mean accuracy
print("Accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
