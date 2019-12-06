from emnist import extract_training_samples as em
import numpy as np
import tensorflow as tf
import cv2

chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z'

]


def get_char(index):
    global chars
    return chars[index]


def load_images():
    images, labels = em('byclass')

    return images, labels


def enable_cuda():
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


def output_model(model, name):
    model.save("models/" + name + ".h5", save_format="tf")


def load_model(name):
    model = tf.keras.models.load_model(name)
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
    img_gray = cv2.GaussianBlur(img_gray, (13, 13), 0)
    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.dilate(img_gray, kernel, iterations=1)

    return processed_img


def get_processed_image(image):
    image = increase_contrast(image)
    cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, image)
    return image


def resize_letter(letter, show=False):
    letter = cv2.resize(letter, dsize=(20, 20), interpolation=cv2.INTER_AREA)
    letter = np.pad(letter, [(4,), (4,)], mode='constant')
    if show:
        cv2.imshow('letter', letter)
        cv2.waitKey(0)

    letter = np.asarray(letter).astype(dtype="float32").reshape((1, 28, 28, 1))
    letter /= 255

    return letter


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
        if len(letter) < 100:
            continue
        letter = np.pad(letter, [(30,), (30,)])
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
