import numpy as np
import sklearn.model_selection as sk
from emnist import extract_training_samples as em
import tensorflow as tf

# Read in and reshape the images
images, labels = em('byclass')
images = images.reshape(images.shape[0], 28, 28, 1)
images = np.array(images).astype(np.float32)

# Split the labels and images into train and test
train_images, test_images, train_labels, test_labels = sk.train_test_split(images, labels, test_size=.25)

# Set up the layers of the model
model = tf.keras.Sequential([
    tf.keras.layers.Convolution2D(62, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(62, activation='softmax')
])

# Check that TF is running on the GPU
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=20)

# Check the statistics
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc, test_loss)
