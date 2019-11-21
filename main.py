
import tkinter as tk
import tkinter.filedialog
import tensorflow as tf
import numpy as np
import cv2


def load_file(gui):
    filename = tk.filedialog.askopenfilename()
    gui.image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def load_model(gui):
    filename = tk.filedialog.askopenfilename()
    gui.model = tf.saved_model.load(filename)


def increase_contrast(img):
    # load image
    img_gray = img

    # increase contrast
    threshold = 150
    img_gray = 255 - img_gray  # invert color
    for row in range(img_gray.shape[0]):
        for col in range(img_gray.shape[1]):
            # print(img_gray[row][col])
            if (img_gray[row][col] < threshold):
                img_gray[row][col] = 0
            else:
                img_gray[row][col] = 255

    # img_gray = 255 - img_gray #invert color back

    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.erode(img_gray, kernel, iterations=1)

    return processed_img


def run_model(gui):
    gui.image = increase_contrast(gui.image)
    cv2.threshold(gui.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, gui.image)
    contours, hier = cv2.findContours(gui.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    d = 0
    for ctr in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = gui.image[y:y + h, x:x + w]
        d += 1
        cv2.bitwise_not(roi)
        # out = gui.model.predict(roi)


def save_text(gui, filename):
    # write to file
    f = open(filename, 'w')


class Window:
    global image
    global model

    def __init__(self):
        window = tk.Tk()
        window.title('PyScribe')
        button = tk.Button(window, text='Load Image', command=lambda: load_file(self)).grid(row=0)
        button2 = tk.Button(window, text='Load Model', command=lambda: load_model(self)).grid(row=1)
        button3 = tk.Button(window, text='Transcribe', command=lambda: run_model(self)).grid(row=2)
        text = tk.Entry(window)

        text.grid(row=3, column=0)

        button4 = tk.Button(window, text='Save', command=lambda: save_text(self, text.get()))
        button4.grid(row=3, column=1)
        window.mainloop()


window = Window()
