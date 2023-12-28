"""
This module contains the main GUI application for the signal processing assignment.
It utilizes PyQt5 for the graphical user interface and interacts with other modules
such as soundcardlib, baseline_gui_v3, and real_time_fft_window.
"""
import sys, pdb, os

import warnings

warnings.filterwarnings("ignore")

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi

from soundcardlib import SoundCardDataSource
from baseline_gui_v3 import Ui_MainWindow
from real_time_fft_window import RealTimeFFTWindow

from misc import get_parser

from pydub import AudioSegment
from pydub.playback import play

import logging
import time

INPUT_DEVICE_STRING = "Select input device"
OUTPUT_DEVICE_STRING = "Select output device"

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, soundcardlib, parent=None):
        """
        Initializes the main window of the GUI.

        Args:
            soundcardlib: The soundcard library used for audio processing.
            parent: The parent widget (default is None).
        """
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.args = get_parser().parse_args()
        self.create_logger()
        self.logger.info("Initializing GUI...")
        self.ui.fft_window: RealTimeFFTWindow  # type hinting
        self.ui.fft_window.initialize_additional_parameters(self.args, soundcardlib, self.logger)
        self.ui.fft_window.prepare_for_plotting()
        self.initialize_default_gui_settings_and_callbacks()
        self.logger.info("GUI initialized.")
        return
    
    def create_logger(self):
        """
        Create a logger object and configure it with the necessary settings.
        The logger will save log messages to a file in the 'logs' directory,
        with the name based on the current date and time.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # save the log file to the logs directory with the name after the current date and time
        logs_path = os.path.join(os.getcwd(), "assignment1-signals", "logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        self.handler = logging.FileHandler(os.path.join(logs_path, f'{time.strftime("%Y%m%d-%H%M%S")}_denoiser.log'))
        self.handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
        self.logger.addHandler(self.handler)
        self.logger.info(f'Arguments: {self.args}')
        return
    
    def connect_to_device(self):
        """
        Connects to the input and output devices selected in the GUI.
        Disables the input and output device selection comboboxes, enables the start button,
        and disables the noise combobox and noise slider.
        """
        input_device_name = self.ui.cb_input_devices.currentText()
        output_device_name = self.ui.cb_output_devices.currentText()

        if input_device_name != INPUT_DEVICE_STRING and output_device_name != OUTPUT_DEVICE_STRING:
            self.ui.fft_window.connect_to_devices(input_device_name, output_device_name)
            self.ui.cb_input_devices.setEnabled(False)
            self.ui.cb_output_devices.setEnabled(False)
            self.ui.pb_start.setEnabled(True)
            self.ui.cb_noise.setEnabled(False)
            self.ui.slider_noise_db.setEnabled(False)
        return
    
    def initialize_default_gui_settings_and_callbacks(self):
        """
        Initializes the default GUI settings and callbacks.

        This function sets the initial state of various GUI elements and connects them to their respective callbacks.
        It also sets default values for input and output devices, lists available devices, and sets the default index.

        Parameters:
        - self: The instance of the class.

        Returns:
        None
        """
        # defaults
        self.logger.info("Initializing default GUI settings and callbacks...")
        self.ui.pb_start.setEnabled(False)
        self.ui.pb_load_files.setEnabled(True)
        self.ui.cb_input_devices.setEnabled(True)
        self.ui.cb_output_devices.setEnabled(True)
        # Set default values
        self.ui.cb_input_devices.addItem(INPUT_DEVICE_STRING)
        self.ui.cb_output_devices.addItem(OUTPUT_DEVICE_STRING)
        self.change_noise_state() # set default noise state

        # list devices
        self.list_devices()
        # Set the default index to 0
        self.ui.cb_input_devices.setCurrentIndex(0)
        self.ui.cb_output_devices.setCurrentIndex(0)

        # wait for user to select input and output device
        self.ui.cb_input_devices.activated.connect(self.connect_to_device)
        self.ui.cb_output_devices.activated.connect(self.connect_to_device)

        self.ui.pb_load_files.clicked.connect(self.load_and_play_files)
        self.ui.slider_noise_db.valueChanged.connect(self.change_noise_state)
        self.set_fft_window_title()
        self.ui.cb_noise.stateChanged.connect(self.change_noise_state)
        self.ui.pb_start.clicked.connect(self.play_button_callback)
        self.ui.pb_save.clicked.connect(self.save_button_callback)
        self.ui.pb_stop_audio.clicked.connect(self.stop_audio_callback)

        # set focus policy
        self.ui.pb_start.setFocusPolicy(Qt.NoFocus) 
        self.ui.slider_noise_db.setFocusPolicy(Qt.NoFocus)  
        self.ui.cb_noise.setFocusPolicy(Qt.NoFocus)  
        self.ui.pb_load_files.setFocusPolicy(Qt.NoFocus)
        self.ui.pb_save.setFocusPolicy(Qt.NoFocus)  
        self.ui.le_noise_in_db.setFocusPolicy(Qt.NoFocus)
        self.ui.pb_load_files.setStyleSheet("color: blue") 
        self.ui.pb_save.setStyleSheet("color: blue")
        self.ui.pb_start.setStyleSheet("color: blue")
        return

    def play_button_callback(self):
        """
        Toggles the paused state of the FFT window and updates the title.
        """
        self.ui.fft_window.paused = not self.ui.fft_window.paused
        self.set_fft_window_title()
        return
    
    def stop_audio_callback(self):
        """
        Stops the audio playback.
        """
        if self.audio_wav:
            self.audio_wav.stop()
            self.ui.pb_stop_audio.setEnabled(False)
        return
    
    def save_button_callback(self):
        """
        Callback function for the save button.
        """
        self.ui.fft_window.on_save_button()
        return

    def set_fft_window_title(self):
        """
        Sets the title of the FFT window based on the current state.

        Returns:
            None
        """
        title_text = "PAUSED" if self.ui.fft_window.paused else ""
        self.ui.fft_window.p1.setTitle(title_text)
        return

    def list_devices(self):
        """
        Retrieves the input and output devices and populates the corresponding combo boxes in the GUI.
        """
        input_devices, output_devices = self.ui.fft_window.get_input_output_devices()
        self.logger.info(f'Input devices: {input_devices}')
        print("input devices: ", input_devices)
        self.logger.info(f'Output devices: {output_devices}')
        print("output devices: ", output_devices)
        for device_name in input_devices.keys():
            self.ui.cb_input_devices.addItem(device_name)
        for device_name in output_devices.keys():
            self.ui.cb_output_devices.addItem(device_name)
        return
    
    def load_and_play_files(self):
        """
        Opens a file dialog to select a .wav file, loads it as an audio segment,
        and plays the audio.
        """
        self.ui.pb_stop_audio.setEnabled(True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Open .wav file", "assignment1-signals/outputs", "WAV Files (*.wav)")
        if file_path:
            self.audio_wav = AudioSegment.from_wav(file_path)
            play(self.audio_wav)
            
        # TODO connect to update method to plot the file
        

    def change_noise_state(self):
        """
        Change the state of noise and update the noise level in the GUI.

        This method is called when the user changes the noise level using the slider
        or toggles the noise checkbox. It updates the noise level in the FFT window
        and displays it in the GUI.

        Returns:
            None
        """
        self.ui.fft_window.noise_level = int(self.ui.slider_noise_db.value())
        noise_text = f'SNR: {self.ui.fft_window.noise_level} dB'
        self.ui.le_noise_in_db.setText(noise_text)

        if self.ui.cb_noise.isChecked():
            self.ui.fft_window.add_noise = True
            self.ui.le_noise_in_db.setStyleSheet("color: red")  
        else:
            self.ui.fft_window.add_noise = False
            self.ui.le_noise_in_db.setStyleSheet("color: black")  
        self.logger.info(f'Noise state: {self.ui.fft_window.add_noise}')
        self.logger.info(f'SNR: {self.ui.fft_window.noise_level} dB')
        return


def main():
    app = QtWidgets.QApplication([])  # for NEW versions
    # Setup soundcardlib
    FS = 16000  # NOTE in Hz; sample rate from microphone should be 16kHz - the same as model was trained on
    soundcardlib = SoundCardDataSource(
        num_chunks=1, sampling_rate=FS, chunk_size=2*1024
    )
    main_app = MyMainWindow(soundcardlib)
    main_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
