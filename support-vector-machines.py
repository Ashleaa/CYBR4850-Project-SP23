# Tutorial followed:
# https://www.youtube.com/watch?v=LYRqcg2s03U

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data = pd.read_csv('packets_data.csv')

# Load your dataset with 50 features, let's call it X, and your target variable, let's call it y
X = data.iloc[:, :79] # assuming the first 50 columns are the features
y = data.iloc[:, -1] # assuming the last column is the target variable

# Split your data into training and testing sets with a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set up your neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)

# Set up the k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize a list to hold the accuracy scores for each fold
scores = []

# Loop through each fold
for train_index, test_index in kf.split(X_train):
    # Split the data into training and testing sets for this fold
    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    # Fit the neural network classifier to the training data for this fold
    clf.fit(X_fold_train, y_fold_train)

    # Make predictions on the testing data for this fold
    y_pred = clf.predict(X_fold_test)

    # Calculate the accuracy score for this fold and append it to the scores list
    score = accuracy_score(y_fold_test, y_pred)
    scores.append(score)

# Print the average accuracy score across all folds
print("Average accuracy score:", np.mean(scores))
