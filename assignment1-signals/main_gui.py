import os, sys, math, pdb

import warnings

warnings.filterwarnings("ignore")

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from soundcardlib import SoundCardDataSource
from baseline_gui import Ui_MainWindow
from real_time_fft_window import RealTimeFFTWindow


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, soundcardlib, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.fft_window.prepare_additional_parameters(soundcardlib)

        # defaults
        self.ui.pb_start.setEnabled(True)
        self.set_fft_window_title()

        # button actions
        self.ui.pb_start.clicked.connect(self.play_button_callback)
        self.ui.pb_start.setFocusPolicy(Qt.NoFocus)
        return

    def play_button_callback(self):
        self.ui.fft_window.paused = not self.ui.fft_window.paused
        self.set_fft_window_title()
        return

    def set_fft_window_title(self):
        title_text = "PAUSED" if self.ui.fft_window.paused else ""
        self.ui.fft_window.p1.setTitle(title_text)
        return


def main():
    app = QtWidgets.QApplication([])  # for NEW versions
    # Setup soundcardlib
    FS = 44000  # Hz
    soundcardlib = SoundCardDataSource(
        num_chunks=3, sampling_rate=FS, chunk_size=4 * 1024
    )
    main_app = MyMainWindow(soundcardlib)
    main_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
