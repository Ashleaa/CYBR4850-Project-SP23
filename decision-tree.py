# Following https://www.w3schools.com/python/python_ml_decision_tree.asp 

## Imports
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

## Code
df = pandas.read_csv("test.csv")

# All data must be numerical, use map to achieve this
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)

d = {'Y' : 1, 'N' : 0}
df['Go'] = df['Go'].map(d)

# Feature columns are columns we use to predict from
# Target column is the values we try to predict
features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features] # Feature columns
Y = df['Go'] # Target Prediction

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

print(dtree.predict([[40, 10, 7, 1]]))

#print(X)
#print('\n')
#print(Y)