from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report

pickle_in = open("Dataset_4class_20person_TopBody.pckl","rb")
dataset = pickle.load(pickle_in)

np.random.shuffle(dataset)

x_train = dataset[:, :-1]
y_train = dataset[:, -1]
y_train = to_categorical(y_train, 4)

x_train = np.expand_dims(x_train, axis=2)
x_test = x_train

model = Sequential()
model.add(Conv1D(64, 3, activation="relu", input_shape=(990, 1)))
model.add(Conv1D(64, 3, activation="relu"))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=4, activation="softmax"))
model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.33)

model.save("Model_4class_20personX4_TopBody.h5")

y_pred = model.predict(x_train)
y_pred = np.argmax(y_pred, axis = 1)
print(classification_report(np.argmax(y_train, axis=1), y_pred))
