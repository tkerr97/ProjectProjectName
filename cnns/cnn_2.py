import sklearn.model_selection as sk
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D
import cv2

from utils import load_images, output_model, enable_cuda, test_model, load_model, get_char


images, labels = load_images()
images = images.reshape(images.shape[0], 28, 28, 1)

x_train, x_test, y_train, y_test = sk.train_test_split(images, labels, test_size=.15)

model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Conv2D(64, kernel_size=(4, 4)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding="same"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2048))
model.add(Dropout(0.1))
model.add(Dense(1024))
model.add(Dense(62, activation="sigmoid"))

enable_cuda()

model.compile(optimzer='adamax',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


_, acc = model.evaluate(x_test, y_test)
print(f"Accuracy {acc}")

output_model(model, "model1")

print(test_model(model, "hello_world.png", True))
