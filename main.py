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
from skimage.transform import rescale, resize

from utils import load_model, increase_contrast, get_letters, get_processed_image, get_char


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

        if os.name == "nt":
            self.prefix = "file:///"
        else:
            self.prefix = "file://"
        self.tmp_dir = "tmp"
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        self.fileName = None
        self.colorImage = None
        self.image = None
        self.bwImage = None
        self.model = None
        self.modelFolderName = None

        if self.rootObjects():
            self.window = self.rootObjects()[0]
            self.imageField = self.window.findChild(QObject, "imagePreview")
            self.modelText = self.window.findChild(QObject, "modelPreview")
        else:
            sys.exit(-1)

    @Slot(str)
    def selectFile(self, file):
        self.fileName = file[len(self.prefix):]
        self.colorImage = QUrl.fromLocalFile(self.fileName)
        self.image = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
        path = Path(self.fileName)
        newFileName = self.tmp_dir + "/"+path.name[:-len(path.suffix)] + "_bw" + path.suffix
        cv2.imwrite(newFileName, self.image)
        self.bwImage = QUrl.fromLocalFile(newFileName)
        self.showProcessImage()
        self.showContours()
        self.showColor()


    @Slot(str)
    def selectModel(self, model):
        self.modelFolderName = model[len(self.prefix):]
        modelName = model.split("/")[-1]
        self.modelText.setProperty("text", modelName)
        self.model = load_model(self.modelFolderName)

    @Slot()
    def showContour(self):
        self.imageField.setProperty("source", self.ctrImage)

    @Slot()
    def showColor(self):
        self.imageField.setProperty("source", self.colorImage)

    @Slot()
    def showBW(self):
        self.imageField.setProperty("source", self.bwImage)

    @Slot()
    def showProcess(self):
        self.imageField.setProperty("source", self.procImage)

    #def scale_to_emnist(self, image):

    @Slot()
    def showContours(self):
        image = increase_contrast(self.image)
        cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, image)
        contours, heir = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 255), 3)

        path = Path(self.fileName)
        name = self.tmp_dir + "/"+path.name[:-len(path.suffix)] + "_ctr" + path.suffix
        cv2.imwrite(name, image)
        self.ctrImage = QUrl.fromLocalFile(name)

    @Slot()
    def showProcessImage(self):
        image = increase_contrast(self.image)
        cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, image)
        path = Path(self.fileName)
        name = self.tmp_dir + "/"+path.name[:-len(path.suffix)] + "_prc" + path.suffix
        cv2.imwrite(name, image)
        self.procImage = QUrl.fromLocalFile(name)


    @Slot()
    def runModel(self):
        self.text = ""
        letters, _ = get_letters(get_processed_image(self.image))
        for letter in letters:

            self.text += get_char(self.model.predict(letter).argmax())
        print(self.text)

    @Slot(str)
    def saveFile(self, filename):
        with open(filename, 'w') as f:
            f.write(self.text)


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Material"

    app = QGuiApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

