import numpy as np
import sklearn.model_selection as sk
from emnist import extract_training_samples as em
import tensorflow as tf

from utils import load_images, enable_cuda, output_model

images, labels = load_images()

# Split the labels and images into train and test
train_images, test_images, train_labels, test_labels = sk.train_test_split(images, labels, test_size=.25)

# Set up the layers of the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(62, activation='softmax')
])

# Check that TF is running on the GPU
enable_cuda()

# Train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=4)

# Check the statistics
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Accuracy: ", test_acc)

output_model(model, "model")

tf.saved_model.save(model, 'model')
print("Saved model to disk")
