import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

# Initialize the SVM classifier
data = SVC(kernel='linear', C=1.0)

# Initialize k-fold validation with 10 folds
kf = KFold(n_splits=10)

# Loop through each fold and fit the model
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the SVM model on the training data
    data.fit(X_train, y_train)

    # Compute feature importance
    feature_importances = np.abs(data.coef_).flatten()
    # Sort feature importance in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Print feature importance
    print("Feature importance:")
    for i in sorted_indices:
        print(f"{X.columns[i]}: {feature_importances[i]:.3f}")

    y_pred = data.predict(X_test)

    # Evaluate the model on the test data
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {score}")

mean_accuracy = sum(scores) / len(scores)
print(f"Mean accuracy: {mean_accuracy}")
