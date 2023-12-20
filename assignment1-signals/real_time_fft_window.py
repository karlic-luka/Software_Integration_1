import sys
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg  # pip install pyqtgraph
from soundcardlib import SoundCardDataSource
from misc import rfftfreq, fft_buffer, ifft_buffer

import torch
import torchaudio
from denoiser.pretrained import add_model_flags, get_model
from denoiser.demucs import DemucsStreamer
from denoiser.utils import bold
from denoiser.dsp import convert_audio
import time


class RealTimeFFTWindow(pg.GraphicsLayoutWidget):  # for NEW versions
    def __init__(self, parent=None):
        super(RealTimeFFTWindow, self).__init__(parent)
        self.initialized_other_parameters = False

    def initialize_additional_parameters(
        self, args, soundcardlib: SoundCardDataSource = None
    ):
        # helper function so I don't have to change the .py file generated with pyuic5
        # every time when I change the .ui file
        self.soundcardlib = soundcardlib
        self.paused = True
        self.downsample = True

        self.setup_denoiser_model(args)
        return
    
    def setup_denoiser_model(self, args):
        # self.args.in_ = 8
        args.in_ = 2 # TODO remove - testing purposes
        args.out = 4
        args.device = 'cpu'
        args.num_threads = 1
        # dry = 0.04
        args.num_frames = 4
        self.first = True

        print(f'Args: {args}')
        print(f'Loading model')
        self.model = get_model(args).to(args.device)
        self.model.eval()
        print(f'Model loaded')
        # print(f'Model: {self.model}')
        self.streamer = DemucsStreamer(self.model, dry=args.dry, num_frames=args.num_frames)
        return
    
    def prepare_for_plotting(self):
        # Setup first plot (time domain)
        self.p1 = self.addPlot()
        self.p1.setLabel("bottom", "Time", "s")
        self.p1.setLabel("left", "Amplitude")
        self.p1.setTitle("")
        self.p1.setLimits(xMin=0, yMin=-1, yMax=1)
        self.ts = self.p1.plot(pen="y")

        # add new row for next plot
        self.nextRow()

        # Setup second plot (frequency domain)
        self.p2 = self.addPlot()
        self.p2.setLabel("bottom", "Frequency", "Hz")
        self.p2.setLimits(xMin=0, yMin=0)
        self.spec = self.p2.plot(
            pen=(50, 100, 200), brush=(50, 100, 200), fillLevel=-100
        )

        # setup third plot (inverse fft)
        self.nextRow()
        self.p3 = self.addPlot()
        self.p3.setLabel("bottom", "Time", "s")
        self.p3.setLabel("left", "Amplitude")
        self.p3.setTitle("Inverse FFT")
        self.p3.setLimits(xMin=0, yMin=-1, yMax=1)
        self.ts2 = self.p3.plot(pen="y")

        # Data ranges
        self.reset_ranges()
        # Define a timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1000 * (self.soundcardlib.chunk_size / self.soundcardlib.fs)
        self.timer.start(int(interval_ms))

        self.initialized_other_parameters = True
        # Show plots
        self.show()

    def reset_ranges(self):
        self.timeValues = self.soundcardlib.timeValues
        self.freqValues = rfftfreq(len(self.timeValues), 1.0 / self.soundcardlib.fs)

        self.p1.setRange(xRange=(0, self.timeValues[-1]), yRange=(-1, 1))
        self.p1.setLimits(xMin=0, xMax=self.timeValues[-1], yMin=-1, yMax=1)
        self.p2.setRange(xRange=(0, self.freqValues[-1] / 2), yRange=(0, 50))
        self.p2.setLimits(xMax=self.freqValues[-1], yMax=50)
        self.spec.setData(fillLevel=0)
        self.p2.setLabel("left", "Signal power", "dB")
        self.p3.setRange(xRange=(0, self.timeValues[-1]), yRange=(-1, 1))
        self.p3.setLimits(xMin=0, xMax=self.timeValues[-1], yMin=-1, yMax=1)

    # The main function that will update the plot
    def update(self):
        # if spacebar (keypressevent), we don't continue
        if not self.initialized_other_parameters or self.paused:
            return

        # collect data
        data = self.soundcardlib.get_buffer()
        # print("data shape: ", data.shape)
        # print(f'dtype: {data.dtype}')

        weighting = np.exp(self.timeValues / self.timeValues[-1])
        Pxx, fhat = fft_buffer(weighting * data[:, 0])
        ifft = ifft_buffer(fhat, threshold=-1)
        if self.downsample:
            downsample_args = dict(
                autoDownsample=False, downsampleMethod="subsample", downsample=10
            )
        else:
            downsample_args = dict(autoDownsample=True)

        self.ts.setData(x=self.timeValues, y=data[:, 0], **downsample_args)
        self.ts2.setData(x=self.timeValues, y=ifft, **downsample_args)
        self.spec.setData(x=self.freqValues, y=(20 * np.log10(Pxx)))

        # denoise
        length = len(data[:, 0])
        self.first = False
        if np.sum(data[:, 0]) == 0:
            print(f'No audio data')
            return        
        # audio_frame = torch.from_numpy(data[:, 0]).to("cpu")

        # start = time.time()
        # with torch.no_grad():
        #     denoised = self.streamer.feed(audio_frame[None])[0]

        # print(f'Inference time: {time.time() - start:.2f}s')
        # print(f'Denoised audio shape: {denoised.shape}')


    def keyPressEvent(self, event):
        text = event.text()
        # Use spacebar to pause the graph
        if text == " ":
            self.paused = not self.paused
            self.p1.setTitle("PAUSED" if self.paused else "")
        else:
            super(RealTimeFFTWindow, self).keyPressEvent(event)

    def get_input_output_devices(self):
        self.input_devices_dict, self.output_devices_dict = self.soundcardlib.get_available_devices()
        return self.input_devices_dict, self.output_devices_dict

    def connect_to_input_device(self, device_name: str):
        dev_index = self.input_devices_dict[device_name]
        self.soundcardlib.connect_and_start_streaming(dev_index)
        self.paused = False
        self.p1.setTitle("")
        return


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Setup soundcardlib
    FS = 44100  # Hz
    soundcardlib = SoundCardDataSource(
        num_chunks=3, sampling_rate=FS, chunk_size=4*1024
    )
    main_app = RealTimeFFTWindow(soundcardlib)
    main_app.show()
    sys.exit(app.exec_())
