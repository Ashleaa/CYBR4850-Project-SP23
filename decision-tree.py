# Tutorial followed:
# https://www.w3schools.com/python/python_ml_decision_tree.asp 

## Imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold

# load data from CSV file
data = pd.read_csv('packets_data.csv')

# extract the features and target variable
X = data.iloc[:, :79] # assuming the first 50 columns are the features
y = data.iloc[:, -1] # assuming the last column is the target variable

# define the decision tree classifier
clf = DecisionTreeClassifier(random_state=0)

# define the number of folds for cross-validation
k = 10

# define the k-fold cross-validation object
kf = KFold(n_splits=k, shuffle=True, random_state=0)

# perform k-fold cross-validation and calculate accuracy score for each fold
scores = cross_val_score(clf, X, y, cv=kf)

# print the average accuracy score and standard deviation across all folds
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
