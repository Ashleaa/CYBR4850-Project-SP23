import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# load dataset
data = pd.read_csv('projectDataset.csv')
data = data.replace([np.inf, -np.inf, np.nan], 0)

# select features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# convert X to int
X = X.astype(int)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate SVM classifier
svm = SVC()

# train the SVM classifier
svm.fit(X_train, y_train)

# make predictions on test set
y_pred = svm.predict(X_test)

# calculate accuracy score and confusion matrix
acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:\n", acc_score)
print("Confusion Matrix:\n", conf_matrix)
