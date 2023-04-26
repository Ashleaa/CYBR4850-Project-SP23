import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
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

# Initialize the neural network classifier
nn = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)

nn.fit(X, y)

# Get the feature importances from the trained neural network
input_layer_weights = nn.coefs_[0]
feature_importances = np.mean(np.abs(input_layer_weights), axis=1)

# Sort the feature importances by descending order
feature_importances = list(zip(X.columns, feature_importances))
feature_importances.sort(key=lambda x: x[1], reverse=True)

# Print the priority order of the fields used in generating the neural network
print("Priority order of fields used in generating the neural network:")
for feature, importance in feature_importances[:5]:
    print(f"{feature}: {100 * importance:.1f}%")

# Initialize k-fold validation with 10 folds
kf = KFold(n_splits=10)

scores = []
# Loop through each fold and fit the model
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the neural network model on the training data
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)

    # Evaluate the model on the test data
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    #print(f"Accuracy: {score}")

mean_accuracy = sum(scores) / len(scores)
print(f"Mean accuracy: {100 * mean_accuracy: .1f}%")
