# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(952, 775)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupGUI = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.groupGUI.setFont(font)
        self.groupGUI.setTitle("")
        self.groupGUI.setAlignment(QtCore.Qt.AlignCenter)
        self.groupGUI.setFlat(False)
        self.groupGUI.setObjectName("groupGUI")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupGUI)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.uipanel1 = QtWidgets.QGroupBox(self.groupGUI)
        self.uipanel1.setAlignment(QtCore.Qt.AlignCenter)
        self.uipanel1.setObjectName("uipanel1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.uipanel1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_panel1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_panel1.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_panel1.setObjectName("horizontalLayout_panel1")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_5.setContentsMargins(0, -1, -1, -1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem)
        self.LoadFile = QtWidgets.QPushButton(self.uipanel1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LoadFile.sizePolicy().hasHeightForWidth())
        self.LoadFile.setSizePolicy(sizePolicy)
        self.LoadFile.setMaximumSize(QtCore.QSize(16777215, 50))
        self.LoadFile.setObjectName("LoadFile")
        self.verticalLayout_5.addWidget(self.LoadFile)
        spacerItem1 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem1)
        self.Process = QtWidgets.QPushButton(self.uipanel1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Process.sizePolicy().hasHeightForWidth())
        self.Process.setSizePolicy(sizePolicy)
        self.Process.setMaximumSize(QtCore.QSize(16777215, 50))
        self.Process.setDefault(False)
        self.Process.setFlat(False)
        self.Process.setObjectName("Process")
        self.verticalLayout_5.addWidget(self.Process)
        spacerItem2 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem2)
        self.uipaneldisplay = QtWidgets.QGroupBox(self.uipanel1)
        self.uipaneldisplay.setObjectName("uipaneldisplay")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.uipaneldisplay)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.rbFaces = QtWidgets.QRadioButton(self.uipaneldisplay)
        self.rbFaces.setChecked(True)
        self.rbFaces.setObjectName("rbFaces")
        self.horizontalLayout_6.addWidget(self.rbFaces)
        self.rbPoints = QtWidgets.QRadioButton(self.uipaneldisplay)
        self.rbPoints.setChecked(False)
        self.rbPoints.setObjectName("rbPoints")
        self.horizontalLayout_6.addWidget(self.rbPoints)
        self.horizontalLayout_14.addLayout(self.horizontalLayout_6)
        self.verticalLayout_5.addWidget(self.uipaneldisplay)
        spacerItem3 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem3)
        self.uipanelbackground = QtWidgets.QGroupBox(self.uipanel1)
        self.uipanelbackground.setObjectName("uipanelbackground")
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.uipanelbackground)
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.rbWhite = QtWidgets.QRadioButton(self.uipanelbackground)
        self.rbWhite.setChecked(True)
        self.rbWhite.setObjectName("rbWhite")
        self.horizontalLayout_17.addWidget(self.rbWhite)
        self.rbBlack = QtWidgets.QRadioButton(self.uipanelbackground)
        self.rbBlack.setObjectName("rbBlack")
        self.horizontalLayout_17.addWidget(self.rbBlack)
        self.horizontalLayout_27.addLayout(self.horizontalLayout_17)
        self.verticalLayout_5.addWidget(self.uipanelbackground)
        spacerItem4 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem4)
        self.exportResult = QtWidgets.QPushButton(self.uipanel1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exportResult.sizePolicy().hasHeightForWidth())
        self.exportResult.setSizePolicy(sizePolicy)
        self.exportResult.setMaximumSize(QtCore.QSize(16777215, 50))
        self.exportResult.setObjectName("exportResult")
        self.verticalLayout_5.addWidget(self.exportResult)
        spacerItem5 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem5)
        self.uipanelperception = QtWidgets.QGroupBox(self.uipanel1)
        self.uipanelperception.setObjectName("uipanelperception")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.uipanelperception)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.textinputtarget = QtWidgets.QLabel(self.uipanelperception)
        self.textinputtarget.setAlignment(QtCore.Qt.AlignCenter)
        self.textinputtarget.setObjectName("textinputtarget")
        self.verticalLayout_11.addWidget(self.textinputtarget)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.Tslider = QtWidgets.QSlider(self.uipanelperception)
        self.Tslider.setEnabled(True)
        self.Tslider.setMinimum(-1)
        self.Tslider.setMaximum(1)
        self.Tslider.setOrientation(QtCore.Qt.Horizontal)
        self.Tslider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.Tslider.setObjectName("Tslider")
        self.verticalLayout_10.addWidget(self.Tslider)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.textinput = QtWidgets.QLabel(self.uipanelperception)
        self.textinput.setAlignment(QtCore.Qt.AlignCenter)
        self.textinput.setObjectName("textinput")
        self.horizontalLayout_10.addWidget(self.textinput)
        self.textoutput = QtWidgets.QLabel(self.uipanelperception)
        self.textoutput.setAlignment(QtCore.Qt.AlignCenter)
        self.textoutput.setObjectName("textoutput")
        self.horizontalLayout_10.addWidget(self.textoutput)
        self.verticalLayout_10.addLayout(self.horizontalLayout_10)
        self.verticalLayout_11.addLayout(self.verticalLayout_10)
        self.horizontalLayout_11.addLayout(self.verticalLayout_11)
        self.verticalLayout_5.addWidget(self.uipanelperception)
        spacerItem6 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem6)
        self.uipanelshape = QtWidgets.QGroupBox(self.uipanel1)
        self.uipanelshape.setObjectName("uipanelshape")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.uipanelshape)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.textmodels = QtWidgets.QLabel(self.uipanelshape)
        self.textmodels.setAlignment(QtCore.Qt.AlignCenter)
        self.textmodels.setObjectName("textmodels")
        self.verticalLayout_13.addWidget(self.textmodels)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.Gslider = QtWidgets.QSlider(self.uipanelshape)
        self.Gslider.setMinimum(-1)
        self.Gslider.setMaximum(1)
        self.Gslider.setOrientation(QtCore.Qt.Horizontal)
        self.Gslider.setObjectName("Gslider")
        self.verticalLayout_12.addWidget(self.Gslider)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.textmodel2 = QtWidgets.QLabel(self.uipanelshape)
        self.textmodel2.setAlignment(QtCore.Qt.AlignCenter)
        self.textmodel2.setObjectName("textmodel2")
        self.horizontalLayout_12.addWidget(self.textmodel2)
        self.textmodel1 = QtWidgets.QLabel(self.uipanelshape)
        self.textmodel1.setAlignment(QtCore.Qt.AlignCenter)
        self.textmodel1.setObjectName("textmodel1")
        self.horizontalLayout_12.addWidget(self.textmodel1)
        self.verticalLayout_12.addLayout(self.horizontalLayout_12)
        self.verticalLayout_13.addLayout(self.verticalLayout_12)
        self.horizontalLayout_13.addLayout(self.verticalLayout_13)
        self.verticalLayout_5.addWidget(self.uipanelshape)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem7)
        self.horizontalLayout_panel1.addLayout(self.verticalLayout_5)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_panel1)
        self.horizontalLayout_7.addWidget(self.uipanel1)
        self.uipanel2 = QtWidgets.QGroupBox(self.groupGUI)
        self.uipanel2.setAlignment(QtCore.Qt.AlignCenter)
        self.uipanel2.setObjectName("uipanel2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.uipanel2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.uipanel2)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_9.setContentsMargins(1, 1, 1, 1)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.frame_horizontalLayout = QtWidgets.QHBoxLayout()
        self.frame_horizontalLayout.setObjectName("frame_horizontalLayout")
        self.horizontalLayout_9.addLayout(self.frame_horizontalLayout)
        self.horizontalLayout.addWidget(self.frame)
        self.horizontalLayout_7.addWidget(self.uipanel2)
        self.horizontalLayout_7.setStretch(0, 20)
        self.horizontalLayout_7.setStretch(1, 80)
        self.verticalLayout_7.addLayout(self.horizontalLayout_7)
        self.verticalLayout_7.setStretch(0, 90)
        self.verticalLayout.addWidget(self.groupGUI)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 952, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Example"))
        self.uipanel1.setTitle(_translate("MainWindow", "Settings"))
        self.LoadFile.setText(_translate("MainWindow", "Load File"))
        self.Process.setText(_translate("MainWindow", "Process"))
        self.uipaneldisplay.setTitle(_translate("MainWindow", "3D Rendering Mode"))
        self.rbFaces.setText(_translate("MainWindow", "Faces"))
        self.rbPoints.setText(_translate("MainWindow", "Points"))
        self.uipanelbackground.setTitle(_translate("MainWindow", "Background Color"))
        self.rbWhite.setText(_translate("MainWindow", "Black"))
        self.rbBlack.setText(_translate("MainWindow", "White"))
        self.exportResult.setText(_translate("MainWindow", "Save 3D Face"))
        self.uipanelperception.setTitle(_translate("MainWindow", "Texture"))
        self.textinputtarget.setText(_translate("MainWindow", "0"))
        self.textinput.setText(_translate("MainWindow", "input texture"))
        self.textoutput.setText(_translate("MainWindow", "output texture"))
        self.uipanelshape.setTitle(_translate("MainWindow", "3D geometry"))
        self.textmodels.setText(_translate("MainWindow", "0"))
        self.textmodel2.setText(_translate("MainWindow", "model1"))
        self.textmodel1.setText(_translate("MainWindow", "model2"))
        self.uipanel2.setTitle(_translate("MainWindow", "3D Model"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
