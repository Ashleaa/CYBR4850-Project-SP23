import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
dt = dt.fit(X, y)

feature_importances = list(zip(X.columns, dt.feature_importances_))
feature_importances.sort(key=lambda x: x[1], reverse=True)


print("Priority order of fields used: :")
for feature, importance in feature_importances[:5]:
    print(f"{feature}: {100 * importance:.1f}%")


# Initialize k-fold validation with 10 folds
kf = KFold(n_splits=10)

# Loop through each fold and fit the model
scores = []
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the decision tree model on the training data
    dt.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = dt.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

mean_accuracy = sum(scores) / len(scores)
print(f"Mean accuracy: {100 * mean_accuracy: .1f}%")
