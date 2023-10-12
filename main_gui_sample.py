import os
import typing
import pdb

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget
from gui_param_labels import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import scipy.signal


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.canvas_exists = False
        self.fig = ""

        # TODO connect when radio buttens are clicked!
        self.ui.pb_plot.clicked.connect(self.show_graph)
        return

    def show_graph(self):
        if self.canvas_exists:
            self.rm_mpl()

        t, s = self.get_signal_data()

        self.fig = Figure()
        ax1f1 = self.fig.add_subplot(111)
        ax1f1.plot(t, s)
        self.add_mpl(self.fig)

    def get_signal_data(self):
        # TODO handle better
        if self.ui.rb_sinus.isChecked():
            self.handle_unnecessary_parameters_gui("sinus")
            t, s = self.get_sinus_signal()

        elif self.ui.rb_square.isChecked():
            self.handle_unnecessary_parameters_gui("square")
            t, s = self.get_square_signal()

        elif self.ui.rb_square.isChecked():
            self.handle_unnecessary_parameters_gui("sawtooth")
            t, s = self.get_sawtooth_signal()

        elif self.ui.rb_polynom.isChecked():
            self.handle_unnecessary_parameters_gui("polynom")
            t, s = self.get_polynom_signal()
        return t, s

    def get_sinus_signal(self):
        # TODO add defaults
        assert self.ui.rb_sinus.isChecked() == True

        start = float(self.ui.le_time_start.text())
        end = float(self.ui.le_time_end.text())
        step = float(self.ui.le_time_step.text())
        amplitude = float(self.ui.le_amplitude.text())
        phase = float(self.ui.le_phase.text())
        freq = float(self.ui.le_freq.text())

        t = np.arange(start, end, step)
        s = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return t, s

    def get_square_signal(self):
        # TODO add defaults
        assert self.ui.rb_square.isChecked() == True

        start = float(self.ui.le_time_start.text())
        end = float(self.ui.le_time_end.text())
        step = float(self.ui.le_time_step.text())
        freq = float(self.ui.le_freq.text())

        t = np.arange(start, end, step)
        s = scipy.signal.square(2 * np.pi * freq * t)  # true square wave -> duty = 0.5
        return t, s

    def get_sawtooth_signal(self):
        # TODO add defaults
        assert self.ui.rb_sawtooth.isChecked() == True

        start = float(self.ui.le_time_start.text())
        end = float(self.ui.le_time_end.text())
        step = float(self.ui.le_time_step.text())
        freq = float(self.ui.le_freq.text())

        t = np.arange(start, end, step)
        s = scipy.signal.sawtooth(2 * np.pi * freq * t)
        return t, s

    def handle_unnecessary_parameters_gui(self, radio_button_string):
        if radio_button_string == "sinus":
            self.ui.le_polynome.setEnabled(False)
            self.ui.label_polynome.setEnabled(False)
        elif radio_button_string in ["square", "sawtooth"]:
            self.ui.le_amplitude.setEnabled(False)
            self.ui.label_amplitude.setEnabled(False)
            self.ui.le_phase.setEnabled(False)
            self.ui.label_amplitude.setEnabled(False)
        elif radio_button_string == "polynom":
            self.ui.le_freq.setEnabled(False)
            self.ui.label_freq.setEnabled(False)
            self.ui.le_amplitude.setEnabled(False)
            self.ui.label_amplitude.setEnabled(False)
            self.ui.le_phase.setEnabled(False)
            self.ui.label_amplitude.setEnabled(False)
        else:
            print(f"I don't know how to handle that radio button.")
        return

    def get_polynom_signal(self):
        # TODO add defaults
        assert self.ui.rb_polynom.isChecked() == True

        start = float(self.ui.le_time_start.text())
        end = float(self.ui.le_time_end.text())
        step = float(self.ui.le_time_step.text())

        t = np.arange(start, end, step)
        coeffs = self.get_polynom_coefficients()
        p = np.poly1d(coeffs)
        s = scipy.signal.sweep_poly(t, p)
        return t, s

    def get_polynom_coefficients(self):
        coeffs_list_str = self.ui.le_polynome.text().strip().split(",")
        return list(map(float, coeffs_list_str))

    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.ui.verticalLayout.addWidget(self.canvas)
        self.canvas.draw()

        self.canvas_exists = True

    def rm_mpl(self):
        self.fig.clear()
        self.ui.verticalLayout.removeWidget(self.canvas)
        self.canvas.close()
        self.canvas_exists = False


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
