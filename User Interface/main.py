import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui import *
from predict import vgg_predict, simpleCNN_predict, LeNet_predict, AlexNet_predict
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
        prob1, types1 = simpleCNN_predict(self.imgName)
        prob2, types2 = vgg_predict(self.imgName)
        prob3, types3 = LeNet_predict(self.imgName)
        prob4, types4 = AlexNet_predict(self.imgName)
        self.textEdit1.setText(f"The probability of you are diagnosed with {types1} is {prob1} by simpleCNN")
        self.textEdit2.setText(f"The probability of you are diagnosed with {types2} is {prob2} by VGG19")
        self.textEdit3.setText(f"The probability of you are diagnosed with {types3} is {prob3} by LeNet")
        self.textEdit4.setText(f"The probability of you are diagnosed with {types4} is {prob4} by AlexNet")

    def makeheatmap(self):
        heatmap("../model_parameters/vgg19_final_project.pth", self.imgName)
        output_path = f'gc_{os.path.basename(self.imgName)}'
        cam = QtGui.QPixmap(output_path).scaled(self.outputImage.width(), self.outputImage.height())
        self.outputImage.setPixmap(cam)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myUi = MyClass()
    myUi.show()
    sys.exit(app.exec_())
