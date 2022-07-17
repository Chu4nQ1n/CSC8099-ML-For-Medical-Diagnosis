import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui import *
from vgg_predict import predict
from gradcam import heatmap


class MyClass(QMainWindow, Ui_Dialog):

    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent)
        self.setupUi(self)
        self.selectImage.clicked.connect(self.openimage)
        self.predictImage.clicked.connect(self.makepredict)
        self.predictImage.clicked.connect(self.makeheatmap)


    def openimage(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "Select Image", "", "*.png")
        jpg = QtGui.QPixmap(self.imgName).scaled(self.inputImage.width(), self.inputImage.height())
        self.inputImage.setPixmap(jpg)


    def makepredict(self):
        prob, types = predict(self.imgName)
        self.textEdit.setText(f"The probability of you are diagnosed with {types} is {prob}")


    def makeheatmap(self):
        heatmap("./vgg19_final_project.pth", self.imgName)
        output_path = f'gc_{os.path.basename(self.imgName)}'
        cam = QtGui.QPixmap(output_path).scaled(self.outputImage.width(), self.outputImage.height())
        self.outputImage.setPixmap(cam)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myUi = MyClass()
    myUi.show()
    sys.exit(app.exec_())
