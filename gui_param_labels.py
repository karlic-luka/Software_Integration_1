# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/first_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(884, 568)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mplwindow = QtWidgets.QWidget(self.centralwidget)
        self.mplwindow.setObjectName("mplwindow")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.mplwindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout.addWidget(self.mplwindow)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_time_end = QtWidgets.QLabel(self.groupBox)
        self.label_time_end.setObjectName("label_time_end")
        self.gridLayout_2.addWidget(self.label_time_end, 3, 0, 1, 1)
        self.rb_sawtooth = QtWidgets.QRadioButton(self.groupBox)
        self.rb_sawtooth.setObjectName("rb_sawtooth")
        self.gridLayout_2.addWidget(self.rb_sawtooth, 11, 1, 1, 1)
        self.pb_plot = QtWidgets.QPushButton(self.groupBox)
        self.pb_plot.setObjectName("pb_plot")
        self.gridLayout_2.addWidget(self.pb_plot, 1, 0, 1, 2)
        self.le_polynome = QtWidgets.QLineEdit(self.groupBox)
        self.le_polynome.setEnabled(True)
        self.le_polynome.setObjectName("le_polynome")
        self.gridLayout_2.addWidget(self.le_polynome, 9, 1, 1, 1)
        self.le_amplitude = QtWidgets.QLineEdit(self.groupBox)
        self.le_amplitude.setObjectName("le_amplitude")
        self.gridLayout_2.addWidget(self.le_amplitude, 8, 1, 1, 1)
        self.rb_polynom = QtWidgets.QRadioButton(self.groupBox)
        self.rb_polynom.setEnabled(True)
        self.rb_polynom.setObjectName("rb_polynom")
        self.gridLayout_2.addWidget(self.rb_polynom, 11, 0, 1, 1)
        self.le_time_start = QtWidgets.QLineEdit(self.groupBox)
        self.le_time_start.setObjectName("le_time_start")
        self.gridLayout_2.addWidget(self.le_time_start, 2, 1, 1, 1)
        self.label_polynome = QtWidgets.QLabel(self.groupBox)
        self.label_polynome.setObjectName("label_polynome")
        self.gridLayout_2.addWidget(self.label_polynome, 9, 0, 1, 1)
        self.label_time_step = QtWidgets.QLabel(self.groupBox)
        self.label_time_step.setEnabled(True)
        self.label_time_step.setObjectName("label_time_step")
        self.gridLayout_2.addWidget(self.label_time_step, 4, 0, 1, 1)
        self.le_sampling_rate = QtWidgets.QLineEdit(self.groupBox)
        self.le_sampling_rate.setObjectName("le_sampling_rate")
        self.gridLayout_2.addWidget(self.le_sampling_rate, 6, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 13, 0, 1, 1)
        self.le_phase = QtWidgets.QLineEdit(self.groupBox)
        self.le_phase.setObjectName("le_phase")
        self.gridLayout_2.addWidget(self.le_phase, 7, 1, 1, 1)
        self.le_time_step = QtWidgets.QLineEdit(self.groupBox)
        self.le_time_step.setEnabled(True)
        self.le_time_step.setObjectName("le_time_step")
        self.gridLayout_2.addWidget(self.le_time_step, 4, 1, 1, 1)
        self.rb_sinus = QtWidgets.QRadioButton(self.groupBox)
        self.rb_sinus.setEnabled(True)
        self.rb_sinus.setObjectName("rb_sinus")
        self.gridLayout_2.addWidget(self.rb_sinus, 10, 0, 1, 1)
        self.label_amplitude = QtWidgets.QLabel(self.groupBox)
        self.label_amplitude.setObjectName("label_amplitude")
        self.gridLayout_2.addWidget(self.label_amplitude, 8, 0, 1, 1)
        self.label_phase = QtWidgets.QLabel(self.groupBox)
        self.label_phase.setObjectName("label_phase")
        self.gridLayout_2.addWidget(self.label_phase, 7, 0, 1, 1)
        self.rb_square = QtWidgets.QRadioButton(self.groupBox)
        self.rb_square.setObjectName("rb_square")
        self.gridLayout_2.addWidget(self.rb_square, 10, 1, 1, 1)
        self.label_time_start = QtWidgets.QLabel(self.groupBox)
        self.label_time_start.setObjectName("label_time_start")
        self.gridLayout_2.addWidget(self.label_time_start, 2, 0, 1, 1)
        self.le_freq = QtWidgets.QLineEdit(self.groupBox)
        self.le_freq.setObjectName("le_freq")
        self.gridLayout_2.addWidget(self.le_freq, 5, 1, 1, 1)
        self.le_time_end = QtWidgets.QLineEdit(self.groupBox)
        self.le_time_end.setObjectName("le_time_end")
        self.gridLayout_2.addWidget(self.le_time_end, 3, 1, 1, 1)
        self.label_sampling_rate = QtWidgets.QLabel(self.groupBox)
        self.label_sampling_rate.setObjectName("label_sampling_rate")
        self.gridLayout_2.addWidget(self.label_sampling_rate, 6, 0, 1, 1)
        self.label_freq = QtWidgets.QLabel(self.groupBox)
        self.label_freq.setObjectName("label_freq")
        self.gridLayout_2.addWidget(self.label_freq, 5, 0, 1, 1)
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
        self.groupBox.setTitle(_translate("MainWindow", "Parameters"))
        self.label_time_end.setText(_translate("MainWindow", "Time End"))
        self.rb_sawtooth.setText(_translate("MainWindow", "Sawtooth"))
        self.pb_plot.setText(_translate("MainWindow", "Plot"))
        self.rb_polynom.setText(_translate("MainWindow", "Polynomial"))
        self.label_polynome.setText(_translate("MainWindow", "Polynome"))
        self.label_time_step.setText(_translate("MainWindow", "Time Step"))
        self.rb_sinus.setText(_translate("MainWindow", "Sinusoidal"))
        self.label_amplitude.setText(_translate("MainWindow", "Amplitude"))
        self.label_phase.setText(_translate("MainWindow", "Phase"))
        self.rb_square.setText(_translate("MainWindow", "Square"))
        self.label_time_start.setText(_translate("MainWindow", "Time Start"))
        self.label_sampling_rate.setText(_translate("MainWindow", "Sampling rate"))
        self.label_freq.setText(_translate("MainWindow", "Frequency"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
