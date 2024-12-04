# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu')) # to cut the -ve values
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # output layer->to get the prob for the all 10 classes
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
# taking 20 percent of the data to check the model--> vali

from sklearn.metrics import accuracy_score
y_prob = model.predict(X_test)
y_pred=y_prob.argmax(axis=1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_image_prediction(index):
    plt.imshow(X_test[index], cmap='gray')
    plt.title(f'Actual: {class_names[y_test[index]]}, Predicted: {class_names[y_pred[index]]}')
    plt.axis('off')
    plt.show()

import numpy as np


show_image_prediction(1239)