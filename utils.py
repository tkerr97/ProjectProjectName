from emnist import extract_training_samples as em
import numpy as np


def load_images():
    images, labels = em('byclass')
    images = images.reshape(images.shape[0], 28, 28, 1)
    images = np.array(images).astype(np.float32)
    return images, labels
