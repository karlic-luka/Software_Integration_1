import os, sys, math, pdb

import warnings

warnings.filterwarnings("ignore")

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget
from PyQt5.uic import loadUi

from soundcardlib import SoundCardDataSource
from baseline_gui_v2 import Ui_MainWindow
from real_time_fft_window import RealTimeFFTWindow

from denoiser.pretrained import get_model
from denoiser.demucs import DemucsStreamer
from misc import get_parser


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, soundcardlib, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.args = get_parser().parse_args()
        print(f'Args: {self.args}')
        self.ui.fft_window: RealTimeFFTWindow  # type hinting
        self.ui.fft_window.initialize_additional_parameters(self.args, soundcardlib)
        self.ui.fft_window.prepare_for_plotting()
        # defaults
        self.ui.pb_start.setEnabled(False)
        self.ui.cb_input_devices.setEnabled(True)
        self.ui.cb_input_devices.setCurrentIndex(-1)
        self.ui.cb_output_devices.setEnabled(True)
        # self.ui.cb_output_devices.setCurrentIndex(-1)

        # list devices
        self.list_devices()
        # connect to device
        self.ui.cb_input_devices.activated.connect(self.connect_to_device)
        # self.ui.cb_output_devices.activated.connect(self.connect_to_device) # TODO
        self.set_fft_window_title()

        # button actions
        self.ui.pb_start.clicked.connect(self.play_button_callback)
        self.ui.pb_start.setFocusPolicy(Qt.NoFocus)  # so it's not pressed with space
        return
    
    def connect_to_device(self):
        device_name = self.ui.cb_input_devices.currentText()
        self.ui.fft_window.connect_to_input_device(device_name)
        self.ui.cb_input_devices.setEnabled(False)
        self.ui.pb_start.setEnabled(True)
        return

    def play_button_callback(self):
        self.ui.fft_window.paused = not self.ui.fft_window.paused
        self.set_fft_window_title()
        return

    def set_fft_window_title(self):
        title_text = "PAUSED" if self.ui.fft_window.paused else ""
        self.ui.fft_window.p1.setTitle(title_text)
        return

    def list_devices(self):
        self.ui.cb_input_devices.clear()
        self.ui.cb_output_devices.clear()
        input_devices, output_devices = self.ui.fft_window.get_input_output_devices()
        print("input devices: ", input_devices)
        print("output devices: ", output_devices)
        for device_name in input_devices.keys():
            self.ui.cb_input_devices.addItem(device_name)
        for device_name in output_devices.keys():
            self.ui.cb_output_devices.addItem(device_name)
        return


def main():
    app = QtWidgets.QApplication([])  # for NEW versions
    # Setup soundcardlib
    FS = 44100  # Hz
    # soundcardlib = SoundCardDataSource(
    #     num_chunks=3, sampling_rate=FS, chunk_size=4 * 1024
    # )
    soundcardlib = SoundCardDataSource(
        num_chunks=3, sampling_rate=FS, chunk_size=4*1024
    )
    main_app = MyMainWindow(soundcardlib)
    main_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
