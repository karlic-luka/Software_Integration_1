# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/baseline_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(884, 657)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.fft_window = RealTimeFFTWindow(self.centralwidget)
        self.fft_window.setObjectName("fft_window")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.fft_window)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout.addWidget(self.fft_window)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 268, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.cb_devices = QtWidgets.QComboBox(self.groupBox)
        self.cb_devices.setObjectName("cb_devices")
        self.gridLayout.addWidget(self.cb_devices, 3, 0, 1, 1)
        self.pb_start = QtWidgets.QPushButton(self.groupBox)
        self.pb_start.setObjectName("pb_start")
        self.gridLayout.addWidget(self.pb_start, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 268, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 4, 0, 1, 1)
        self.label_devices = QtWidgets.QLabel(self.groupBox)
        self.label_devices.setObjectName("label_devices")
        self.gridLayout.addWidget(self.label_devices, 2, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 884, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "MENU"))
        self.pb_start.setText(_translate("MainWindow", "Start/Stop"))
        self.label_devices.setText(_translate("MainWindow", "Available devices"))
from real_time_fft_window import RealTimeFFTWindow


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
