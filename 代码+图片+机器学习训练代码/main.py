import math
import sys

import cv2
import numpy as np
from PIL.Image import fromarray
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QProgressBar, QPushButton, QLabel, QStatusBar
import cvfuncs
import classify
from cvui import Ui_MainWindow


class MyDesiger(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyDesiger, self).__init__(parent)
        self.setupUi(self)
        self.actionOpen.triggered.connect(self.open_pic)
        self.open_file.clicked.connect(self.open_pic)
        self.do_detec.clicked.connect(self.do_detection)
        self.capSet = []
        self.pnSet = []
        self.colorSet = []
        self.pic_path = ""
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet('QStatusBar::item {border: none;}')
        self.setStatusBar(self.statusBar)
        self.progressBar = QProgressBar()
        self.sblabel = QLabel()
        self.sblabel2 = QLabel()
        self.sblabel.setText(" ")
        self.sblabel2.setText(" ")

        # self.statusBar.addWidget(self.label, 0)
        self.statusBar.addPermanentWidget(self.sblabel, stretch=2)
        self.statusBar.addPermanentWidget(self.sblabel2, stretch=0)
        self.statusBar.addPermanentWidget(self.progressBar, stretch=4)
        # self.statusBar().addWidget(self.progressBar)

        # This is simply to show the bar
        # self.progressBar.setGeometry(0, 0, 100, 5)
        self.progressBar.setRange(0, 100)  # 设置进度条的范围
        self.progressBar.setValue(0)

        self.text_labels = []
        self.text_labels.append(self.label1)
        self.text_labels.append(self.label2)
        self.text_labels.append(self.label3)
        self.text_labels.append(self.label4)
        self.text_labels.append(self.label5)
        self.text_labels.append(self.label6)
        self.text_labels.append(self.label7)
        self.text_labels.append(self.label8)
        self.text_labels.append(self.label9)
        self.text_labels.append(self.label10)

        self.pic_labels = []
        self.pic_labels.append(self.cap1)
        self.pic_labels.append(self.cap2)
        self.pic_labels.append(self.cap3)
        self.pic_labels.append(self.cap4)
        self.pic_labels.append(self.cap5)
        self.pic_labels.append(self.cap6)
        self.pic_labels.append(self.cap7)
        self.pic_labels.append(self.cap8)
        self.pic_labels.append(self.cap9)
        self.pic_labels.append(self.cap10)

        self.clear_labels()


    def open_pic(self):
        self.clear_labels()
        pic_path = QFileDialog.getOpenFileName(self, '选择文件', '', 'Img files(*.jpg , *.png)')
        self.pic_path = pic_path[0]
        print(pic_path)
        if pic_path[0] == "":
            return
        transform = QTransform()  # PyQt5
        transform.rotate(90)
        pix = QPixmap(pic_path[0])
        pix = pix.transformed(transform)
        pix = pix.scaled(self.pico.width(), self.pico.height())
        self.pico.setPixmap(pix)
        self.pico.setScaledContents(True)
        self.progressBar.setValue(100)
        self.sblabel.setText("Done")

    def do_detection(self):
        self.sblabel.setText("Processing")
        if self.pic_path == "":
            self.sblabel.setText("Error: no pic opened")
            return
        self.progressBar.setValue(0)
        img = cv2.imread(self.pic_path)
        self.progressBar.setValue(20)
        print(img)
        contour, capcolor = cvfuncs.division(img)
        self.progressBar.setValue(70)
        capSet, pnSet, colorSet = cvfuncs.judge(contour, capcolor)
        self.progressBar.setValue(95)
        print(capSet)
        print(pnSet)
        pix = QPixmap("rec.jpg")
        pix = pix.scaled(self.pico.width(), self.pico.height())
        self.pico.setPixmap(pix)
        self.capSet = capSet
        self.pnSet = pnSet
        self.colorSet = colorSet
        self.show_labels()
        self.progressBar.setValue(100)
        self.sblabel.setText("Done")

    def show_labels(self):
        colors = ("红色", "绿色", "蓝色", "黄色", "紫色", "青色")
        for i in range(min(len(self.pnSet), 10)):
            self.text_labels[i].setText(self.pnSet[i]+","+colors[self.colorSet[i]])
        for i in range(min(len(self.capSet),10)):
            pix = QPixmap("littlecap" + str(self.capSet[i]) + ".jpg")
            pix = pix.scaled(self.cap1.height(), self.cap1.height())
            self.pic_labels[i].resize(self.cap1.height(), self.cap1.height())
            self.pic_labels[i].setPixmap(pix)
            self.pic_labels[i].setScaledContents(True)
        if len(self.capSet) < 10:
            for i in range(len(self.capSet)-1, 10):
                self.pic_labels[i].setText("pic" + str(i+1) + " missed")
        if len(self.pnSet) < 10:
            for i in range(len(self.pnSet) - 1, 10):
                print(i)
                self.text_labels[i].setText("no pred")

    def clear_labels(self):
        self.pico.setText("Wating for a image to load")
        for i in range(10):
            self.pic_labels[i].setPixmap(QPixmap(""))
            self.text_labels[i].setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MyDesiger()
    ui.show()
    sys.exit(app.exec_())
