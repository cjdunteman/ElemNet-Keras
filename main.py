import pandas as pd
import numpy as np
from numpy import loadtxt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load dataset
df = pd.read_csv('generated_features.csv')

# Get num of cols
total_cols = len(df.axes[1]) - 1

dataset = np.genfromtxt('generated_features.csv', delimiter=",")

# Split into input (x) and output (y) variables
X = dataset[:, 0:total_cols]
y = dataset[:, total_cols]

print(X)
print(y)

# Build model
model = Sequential()
model.add(Dense(1024, input_dim=total_cols, activation='relu'))  # first layer
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(X, y, epochs=3, batch_size=10)

# Evaluate model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
