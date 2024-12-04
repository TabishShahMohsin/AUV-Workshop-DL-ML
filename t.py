import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

(X_train, y_train),(X_test, y_test)= fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

X_train = X_train/255.0
X_test = X_test/255.0
model = Sequential([
    Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Flatten(),
    Dense(128,activation = 'relu'),
    Dense(10, activation = 'softmax')
])

# model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=2)
# # taking 20 percent of the data to check the model--> vali
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

import numpy as np
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to display the image and its predicted class
def show_image_prediction(index):
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')  # Reshape the image to 2D
    plt.title(f'Actual: {class_names[y_test[index]]}, Predicted: {class_names[y_pred[index]]}')
    plt.axis('off')
    plt.show()

# Display the prediction for a specific test image
show_image_prediction(2000)


