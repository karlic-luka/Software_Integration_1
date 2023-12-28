# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'baseline_gui_v3.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1129, 837)
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
        self.groupBox.setMaximumSize(QtCore.QSize(300, 16777215))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet("font: 12pt \"MS Reference Sans Serif\"")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.cb_input_devices = QtWidgets.QComboBox(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.cb_input_devices.setFont(font)
        self.cb_input_devices.setObjectName("cb_input_devices")
        self.gridLayout.addWidget(self.cb_input_devices, 10, 0, 1, 1)
        self.pb_save = QtWidgets.QPushButton(self.groupBox)
        self.pb_save.setObjectName("pb_save")
        self.gridLayout.addWidget(self.pb_save, 3, 0, 1, 1)
        self.label_output_devices = QtWidgets.QLabel(self.groupBox)
        self.label_output_devices.setObjectName("label_output_devices")
        self.gridLayout.addWidget(self.label_output_devices, 11, 0, 1, 1)
        self.label_input_devices = QtWidgets.QLabel(self.groupBox)
        self.label_input_devices.setObjectName("label_input_devices")
        self.gridLayout.addWidget(self.label_input_devices, 9, 0, 1, 1)
        self.pb_start = QtWidgets.QPushButton(self.groupBox)
        self.pb_start.setObjectName("pb_start")
        self.gridLayout.addWidget(self.pb_start, 0, 0, 1, 1)
        self.pb_load_files = QtWidgets.QPushButton(self.groupBox)
        self.pb_load_files.setObjectName("pb_load_files")
        self.gridLayout.addWidget(self.pb_load_files, 1, 0, 1, 1)
        self.pb_stop_audio = QtWidgets.QPushButton(self.groupBox)
        self.pb_stop_audio.setEnabled(False)
        self.pb_stop_audio.setObjectName("pb_stop_audio")
        self.gridLayout.addWidget(self.pb_stop_audio, 2, 0, 1, 1)
        self.le_noise_in_db = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.le_noise_in_db.setFont(font)
        self.le_noise_in_db.setAlignment(QtCore.Qt.AlignCenter)
        self.le_noise_in_db.setReadOnly(True)
        self.le_noise_in_db.setObjectName("le_noise_in_db")
        self.gridLayout.addWidget(self.le_noise_in_db, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 268, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 8, 0, 1, 1)
        self.cb_noise = QtWidgets.QCheckBox(self.groupBox)
        self.cb_noise.setObjectName("cb_noise")
        self.gridLayout.addWidget(self.cb_noise, 4, 0, 1, 1)
        self.slider_noise_db = QtWidgets.QSlider(self.groupBox)
        self.slider_noise_db.setMinimum(-5)
        self.slider_noise_db.setMaximum(20)
        self.slider_noise_db.setOrientation(QtCore.Qt.Horizontal)
        self.slider_noise_db.setObjectName("slider_noise_db")
        self.gridLayout.addWidget(self.slider_noise_db, 6, 0, 1, 1)
        self.cb_output_devices = QtWidgets.QComboBox(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.cb_output_devices.setFont(font)
        self.cb_output_devices.setObjectName("cb_output_devices")
        self.gridLayout.addWidget(self.cb_output_devices, 12, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 268, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 13, 0, 1, 1)
        self.pb_restart = QtWidgets.QPushButton(self.groupBox)
        self.pb_restart.setObjectName("pb_restart")
        self.gridLayout.addWidget(self.pb_restart, 7, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1129, 21))
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
        self.pb_save.setText(_translate("MainWindow", "Save Audio"))
        self.label_output_devices.setText(_translate("MainWindow", "Output devices"))
        self.label_input_devices.setText(_translate("MainWindow", "Input devices"))
        self.pb_start.setText(_translate("MainWindow", "Start/Stop"))
        self.pb_load_files.setText(_translate("MainWindow", "Listen to Audio"))
        self.pb_stop_audio.setText(_translate("MainWindow", "Stop Audio"))
        self.cb_noise.setText(_translate("MainWindow", "Add AWGN"))
        self.pb_restart.setText(_translate("MainWindow", "Restart"))
from real_time_fft_window import RealTimeFFTWindow


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())