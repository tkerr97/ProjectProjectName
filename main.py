import os
import sys
from pathlib import Path

from PySide2.QtCore import QUrl, QObject, Slot, Property, Signal, QtFatalMsg, QtCriticalMsg, QtWarningMsg, QtInfoMsg, \
    qInstallMessageHandler
from PySide2.QtGui import QGuiApplication, QImage
from PySide2.QtQml import QQmlApplicationEngine
import tensorflow as tf
import numpy as np
import cv2


def qt_message_handler(mode, context, message):
    if mode == QtInfoMsg:
        mode = 'Info'
    elif mode == QtWarningMsg:
        mode = 'Warning'
    elif mode == QtCriticalMsg:
        mode = 'critical'
    elif mode == QtFatalMsg:
        mode = 'fatal'
    else:
        mode = 'Debug'
    print("%s: %s (%s:%d, %s)" % (mode, message, context.file, context.line, context.file))


print("OpenCV version: " + cv2.__version__)
print("Tensorflow version: " + tf.__version__)


class MainWindow(QQmlApplicationEngine):
    def __init__(self):
        super().__init__()
        self.load(os.path.join(os.getcwd(), "view.qml"))
        qInstallMessageHandler(qt_message_handler)
        self.rootContext().setContextProperty("MainWindow", self)

        if self.rootObjects():
            self.window = self.rootObjects()[0]
            self.imageField = self.window.findChild(QObject, "imagePreview")
            self.modelText = self.window.findChild(QObject, "modelPreview")
        else:
            sys.exit(-1)

    @Slot(str)
    def selectFile(self, file):
        self.fileName = file[len("file:///"):]
        self.colorImage = QUrl.fromLocalFile(self.fileName)
        image = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
        path = Path(self.fileName)
        newFileName = str(path)[:-len(path.suffix)] + "_bw" + path.suffix
        cv2.imwrite(newFileName, image)
        self.bwImage = QUrl.fromLocalFile(newFileName)
        self.showColor()

    @Slot(str)
    def selectModel(self, model):
        self.modelFileName = model[len("file:///"):]
        modelName = model.split("/")[-1]
        self.modelText.setProperty("text", modelName)
        self.model = tf.saved_model.load(self.modelFileName)
        #self.model = tf.saved_model.load("/".join(self.modelFileName.split("/")[:-1]))

    @Slot()
    def showColor(self):
        self.imageField.setProperty("source", self.colorImage)

    @Slot()
    def showBW(self):
        self.imageField.setProperty("source", self.bwImage)

    def increase_contrast(self, img):
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

    @Slot()
    def runModel(self):
        self.image = self.increase_contrast(self.image)
        cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, self.image)
        contours, hier = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        d = 0
        for ctr in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            # Getting ROI
            roi = self.image[y:y + h, x:x + w]
            d += 1
            cv2.bitwise_not(roi)
            # out = gui.model.predict(roi)


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Material"

    app = QGuiApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
