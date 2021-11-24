import PyQt5
#from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib import pyplot as plt
import cv2
from HW1_ui import Ui_MainWindow
import numpy as np
from cfg import *


class Mainwindow_set(Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.setupUi(self)  # ????
        self.button_set()  # define button function
        self.title = 'opencv HW1 (Q1 ~ Q4)'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480

    def button_set(self):
        self.ui.pushButton.clicked.connect(self.load_image)
        '''
        self.ui.pushButton_2.clicked.connect()
        self.ui.pushButton_3.clicked.connect
        self.ui.pushButton_4.clicked.connect
        self.ui.pushButton_5.clicked.connect
        self.ui.pushButton_6.clicked.connect
        self.ui.pushButton_7.clicked.connect
        self.ui.pushButton_9.clicked.connect
        self.ui.pushButton_10.clicked.connect
        self.ui.pushButton_11.clicked.connect
        self.ui.pushButton_12.clicked.connect
        '''

    def load_image(self):
        img1_path = q1_folder + 'Sun.jpg'
        img1 = cv2.imread(img1_path)
        cv2.imshow('Sun.jpg', img1)
        print('Sun_width : ', img1.shape[1])
        print('Sun_height :', img1.shape[0])
        cv2.waitKey(0)
        # close window
        cv2.destroyWindow('Sun.jpg')
