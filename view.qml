import QtQuick 2.12
import QtQuick.Dialogs 1.0
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12

ApplicationWindow {
    visible: true
    title: "PyScribe"
    height: 625
    width: 800

    Material.theme: Material.Dark

    FileDialog {
      id: dialog
      visible: false
      title: "Choose your image(s)"
      nameFilters: [ "Image files (*.jpg, *.png, *.jpeg)" ]
      onAccepted: {
        MainWindow.selectFile(dialog.fileUrls)
      }
    }
    FileDialog {
      id: modelDialog
      visible: false
      title: "Select your model file"
      nameFilters: [ "Model files (*.h5, *.pb, *.pbtxt)" ]
      onAccepted: {
        MainWindow.selectModel(modelDialog.fileUrls)
      }
    }
    Column {
      anchors {
          horizontalCenter: parent.horizontalCenter
      }
      spacing: 5

      Pane {
        anchors {
          horizontalCenter: parent.horizontalCenter
          topMargin: 20
        }
        height: 525
        width: 700
        Material.elevation: 6

        Text {
          id: selectImageText
          anchors.centerIn: parent
          text: qsTr("Please select an image")
          color: Material.color(Material.Grey)
          font.pointSize: 24
        }
        Image {
          anchors.centerIn: parent
          height: 525
          width: 700
          id: selectedImage
          visible: false
          objectName: "imagePreview"

          onStatusChanged: {
            if (selectedImage.status == Image.Ready) {
              selectImageText.visible = true
              selectedImage.visible = true
            }
          }
        }
      }
      Pane {
        anchors {
          horizontalCenter: parent.horizontalCenter
        }
        Material.elevation: 4
        Text {
          text: "Please select a model"
          objectName: "modelPreview"
          anchors.centerIn: parent
          color: Material.color(Material.Grey)
          font.pointSize: 14
        }
      }
    }

    Row {
        anchors {
          bottom: parent.bottom
          horizontalCenter: parent.horizontalCenter
        }
        spacing: 10

        Button {
          text: qsTr("Load Image")
          onClicked: {
            dialog.visible = true
          }
        }
        Button {
          text: qsTr("Load Model")
          onClicked: {
            modelDialog.visible = true
          }
        }
        Button {
          text: "Transcribe"
          onClicked: MainWindow.runModel
        }
        Button {
          text: "Save"
        }
        Button {
          text: "Options"
          id: optionsButton
          onClicked: optionsMenu.open()

          Menu {
            id: optionsMenu
            y: optionsButton.height

            MenuItem {
              text: "Colored Image"
              onTriggered: MainWindow.showColor()
            }
            MenuItem {
              text: "BW Image"
              onTriggered: MainWindow.showBW()
            }
          }
        }
    }

}