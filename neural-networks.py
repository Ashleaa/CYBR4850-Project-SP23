import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('projectDataset.csv')
data = data.replace([np.inf, -np.inf, np.nan], 0)


# Separate the features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert to float data type
X = X.astype('float32')
y = y.astype('float32')

# Define the K-fold cross-validation object
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train and evaluate the model using K-fold cross-validation
scores = []
for train_idx, test_idx in kfold.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    scores.append(acc)

print('Accuracy: %.3f%% (%.3f)' % (np.mean(scores)*100, np.std(scores)))
