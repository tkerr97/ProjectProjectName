import json
import os

from emnist import extract_training_samples as em
import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import rescale, resize


chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
def get_char(index):
    global chars
    return chars[index]

def load_images():
    images, labels = em('byclass')
    images = images.reshape(images.shape[0], 28, 28, 1)
    return images, labels

def enable_cuda():
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

def output_model(model, name):
    with open(f"models/{name}.json", 'w') as f:
        f.write(model.to_json())

    model.save_weights(f"models/{name}.h5")

def load_model(name):
    if os.path.isfile(f'models/{name}.json') and os.path.isfile(f'models/{name}.h5'):
        with open(f'models/{name}.json', 'r') as f:
            js = json.load(f)
            model = tf.keras.models.model_from_json(json.dumps(js))
        model.load_weights(f'models/{name}.h5')
    else:
        model = tf.saved_model.load('models/')
    return model

def increase_contrast(img):
    # load image
    img_gray = img

    # increase contrast
    threshold = 150
    img_gray = 255 - img_gray  # invert color
    for row in range(img_gray.shape[0]):
        for col in range(img_gray.shape[1]):
            # print(img_gray[row][col])
            if img_gray[row][col] < threshold:
                img_gray[row][col] = 0
            else:
                img_gray[row][col] = 255

    # img_gray = 255 - img_gray #invert color back

    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.erode(img_gray, kernel, iterations=1)

    return processed_img

def get_processed_image(image):
    image = increase_contrast(image)
    cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, image)
    return image

def resize_letter(letter):
    letter = cv2.resize(letter, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
    return resize(letter, (1, 28, 28, 1))

def get_letters(image):
    contours, heir = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    letters = []
    images = []
    for ctr in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting letter
        letter = image[y:y + h, x:x + w]
        cv2.bitwise_not(letter)
        images.append(letter)
        letters.append(resize_letter(letter))

    return letters, images

def test_model(model, name, show=False):
    im = cv2.imread(f"pictures/{name}", cv2.IMREAD_GRAYSCALE)
    text = ""
    letters, images = get_letters(get_processed_image(im))
    for i, letter in enumerate(letters):
        res = model.predict(letter).argmax()
        char = get_char(res)
        if show:
            cv2.imshow(char, images[i])
            cv2.waitKey()
        text += char
    return text

