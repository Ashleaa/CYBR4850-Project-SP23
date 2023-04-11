# Tutorial followed:
# https://www.youtube.com/watch?v=LYRqcg2s03U

import numpy as np
import pandas as pd
from pydataset import data
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import model_selection


df = pd.read_csv("test.csv")

# All data must be numerical, use map to achieve this
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)

d = {'Y' : 1, 'N' : 0}
df['Go'] = df['Go'].map(d)


# Scaling variables
df = (df - df.min()) / (df.max() - df.min())

# Independent and Dependent
features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features] # Feature columns
y = df['Go'] # Target Prediction

# 70/30 split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3, random_state=1)

h1=svm.LinearSVC(C=1)
h1.fit(X_train, y_train)

h1.score(X_train, y_train)

y_pred=h1.predict(X_test)
pd.crosstab(y_test, y_pred)

print("Support Vector Report: \n")
print(classification_report(y_test, y_pred))


