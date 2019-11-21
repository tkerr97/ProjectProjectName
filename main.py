import os
import sys

from PySide2.QtCore import QUrl, QObject, Slot, Property, Signal, QtFatalMsg, QtCriticalMsg, QtWarningMsg, QtInfoMsg, \
    qInstallMessageHandler
from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine

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
        self.imageField.setProperty("source", QUrl.fromLocalFile(file[len("file://"):]))

    @Slot(str)
    def selectModel(self, model):
        self.modelText.setProperty("text", model.split("/")[-1])


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Material"

    app = QGuiApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
