import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# Load the dataset
data = pd.read_csv('projectDataset.csv')
data = data.replace([np.inf, -np.inf, np.nan], 0)

# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert any float integers to integers

X = X.astype(int)

# Initialize the decision tree classifier
dt = DecisionTreeClassifier()

# Initialize k-fold validation with 10 folds
kf = KFold(n_splits=10)

# Loop through each fold and fit the model
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the decision tree model on the training data
    dt.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = dt.score(X_test, y_test)
    print(f"Accuracy: {score}")
