import numpy as np
from PyQt5 import QtCore
import pyqtgraph as pg  # pip install pyqtgraph
from soundcardlib import SoundCardDataSource
from misc import rfftfreq, fft_buffer

import torch
from denoiser.pretrained import get_model
from denoiser.demucs import DemucsStreamer
import time
import os
import soundfile as sf
import sounddevice as sd

class RealTimeFFTWindow(pg.GraphicsLayoutWidget):  # for NEW versions
    def __init__(self, parent=None):
        super(RealTimeFFTWindow, self).__init__(parent)
        self.initialized_other_parameters = False
        self.noise = np.random.normal(0, 1, size=(10000, 1)).astype(np.float32) # pre-generate noise

    def initialize_additional_parameters(
        self, args, soundcardlib: SoundCardDataSource = None, logger=None
    ):
        # helper function so I don't have to change the .py file generated with pyuic5
        # every time when I change the .ui file
        self.soundcardlib = soundcardlib
        self.logger = logger
        self.paused = True
        self.downsample = True
        self.exit = False
        self.add_noise = False
        self.noise_level = 0 # dB
        self.logger.info(f'Initializing denoiser')

        self.setup_denoiser_model(args)
        self.logger.info('Denoiser model setup done')
        self.out_data = []
        self.in_data = []
        return
    
    def setup_denoiser_model(self, args):
        self.args = args
        self.args.device = 'cpu'
        self.args.num_threads = 1 # If you have DDR3 RAM, setting -t 1 can improve performance.")
        self.args.dry = 0.04
        self.args.num_frames = 4
        self.first = True

        self.logger.info(f'Args: {args}') 
        print(f'Args: {args}')
        print(f'Loading model')
        self.logger.info(f'Loading model')
        self.model = get_model(args).to(args.device)
        self.model.eval()
        print(f'Model loaded')
        self.logger.info(f'Model loaded')   
        # print(f'Model: {self.model}')
        self.streamer = DemucsStreamer(self.model, dry=self.args.dry, num_frames=self.args.num_frames)
        sr_ms = self.model.sample_rate / 1000
        self.logger.info(f"Ready to process audio, total lag: {self.streamer.total_length / sr_ms:.1f}ms.")
        print(f"Ready to process audio, total lag: {self.streamer.total_length / sr_ms:.1f}ms.")
        return
    
    def prepare_for_plotting(self):
        # Setup first plot (time domain)
        pg.setConfigOptions(antialias=True)
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

        # setup third plot (original vs noisy data)
        self.nextRow()
        self.p3 = self.addPlot()
        self.p3.setLabel("bottom", "Time", "s")
        self.p3.setLabel("left", "Amplitude")
        self.p3.setTitle("Original signal vs Noisy signal")
        self.p3.setLimits(xMin=0, yMin=-1, yMax=1)
        self.p3.addLegend()
        self.original_signal_plot = self.p3.plot(pen='y', width=2, name='Original audio')  # Plot for the original signal
        self.noisy_signal_plot = self.p3.plot(pen='r', width=1, name='Noisy audio') # Plot for the noisy signal

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
        return

    def add_awgn_in_db(self, audio, target_noise_db):
        signal_power = np.mean(audio ** 2)
        signal_power_db = 10 * np.log10(signal_power)
        noise_power_db = signal_power_db - target_noise_db
        noise_power = 10 ** (noise_power_db / 10)
        noise = self.noise[:len(audio)] * np.sqrt(noise_power)
        self.noise = np.roll(self.noise, -len(audio)) # roll the noise so it's different every time
        noisy_audio = audio + noise
        return noisy_audio
    
    
    def update(self):
        # if spacebar (keypressevent), we don't continue
        if not self.initialized_other_parameters or self.paused or self.exit:
            return

        # collect data
        start1 = time.time()
        data = self.soundcardlib.get_buffer()

        if np.sum(data[:, 0]) == 0:
            self.logger.info(f'No audio data')
            return
        
        self.original_signal_plot.setData(self.timeValues, data[:, 0])  # Update the original signal plot
        if self.add_noise:
            try:
                data = self.add_awgn_in_db(data, self.noise_level)
                self.noisy_signal_plot.setData(self.timeValues, data[:, 0])  # Update the noisy signal plot
                self.logger.info(f'Added AWGN. SNR: {self.noise_level} dB')
            except Exception as e:
                print(f'Error while trying to add AWgn: {e}')
                self.logger.info(f'Error while trying to add AWgn: {e}')
                return
        else:
            self.logger.info(f'No AWGN added')

        self.logger.info(f'Adding noise time: {time.time() - start1:.4f}s')
        self.in_data.append(data)
        self.logger.info(f'Getting buffer + adding noise time: {time.time() - start1:.4f}s')

        # self.logger.info(f'Audio data shape: {data.shape}')
        start2 = time.time()
        weighting = np.exp(self.timeValues / self.timeValues[-1])
        Pxx, fhat = fft_buffer(weighting * data[:, 0])
        # ifft = ifft_buffer(fhat, threshold=-1)
        if self.downsample:
            downsample_args = dict(
                autoDownsample=False, downsampleMethod="subsample", downsample=10
            )
        else:
            downsample_args = dict(autoDownsample=True)
        self.logger.info(f'FFT time: {time.time() - start2:.4f}s')
        start3 = time.time()
        # self.ts2.setData(x=self.timeValues, y=ifft, **downsample_args)
        self.ts.setData(x=self.timeValues, y=data[:, 0], **downsample_args)
        self.spec.setData(x=self.freqValues, y=(20 * np.log10(Pxx)))
        self.logger.info(f'Plotting time: {time.time() - start3:.4f}s')

        start4 = time.time()
        self.first = False
     
        audio_frame = torch.from_numpy(data[:, 0]).to("cpu").float()
        # start = time.time()
        with torch.no_grad():
            denoised = self.streamer.feed(audio_frame[None])[0]

        if self.args.compressor:
            denoised = 0.99 * torch.tanh(denoised)

        underflow = self.output_stream.write(denoised)

        self.out_data.append(denoised)
        self.logger.info(f'Inference time: {time.time() - start4:.4f}s')
        self.logger.info(f'Denoised audio shape: {denoised.shape}')
        self.logger.info(f'---------------------------------------------------------')


    def keyPressEvent(self, event):
        text = event.text()
        # Use spacebar to pause the graph
        if text == " ":
            self.paused = not self.paused
            self.p1.setTitle("PAUSED" if self.paused else "")
        # control+c to quit
        elif text == "\x03":
            self.exit = True
            self.on_exiting()
        else:
            super(RealTimeFFTWindow, self).keyPressEvent(event)

    def get_input_output_devices(self):
        self.input_devices_dict, self.output_devices_dict = self.soundcardlib.get_available_devices()
        return self.input_devices_dict, self.output_devices_dict

    def connect_to_devices(self, input_name: str, output_name: str):
        input_dev_index = self.input_devices_dict[input_name]
        self.soundcardlib.connect_and_start_streaming(input_dev_index)
        output_index = self.output_devices_dict[output_name]
        self.output_stream = sd.OutputStream(
            samplerate=self.soundcardlib.fs,
            device=output_index,
            channels=1 # TODO parametrize
        )
        
        try:
            self.output_stream.start()
            self.logger.info(f'Using output device: {output_name}')
            self.logger.info(f'Output stream started')
            print(f'Using output device: {output_name}')
        except Exception as e:
            print(f'Error: {e}')
            self.logger.info('Error: {e}')
            return
        self.paused = True
        return
    
    def on_exiting(self):
        self.logger.info(f'Exiting')
        self.in_data = np.concatenate(self.in_data)
        self.in_data = np.clip(self.in_data, -1.0, 1.0)
        self.in_data = (self.in_data * 2**15).astype(np.int16)
        self.out_data = np.concatenate(self.out_data)
        self.out_data = np.clip(self.out_data, -1.0, 1.0)
        self.out_data = (self.out_data * 2**15).astype(np.int16)

        self.logger.info(f'Input audio shape: {self.in_data.shape}')
        self.logger.info(f'Output audio shape: {self.out_data.shape}')
        self.logger.info(f'Dtype: {self.in_data.dtype}')
        self.logger.info(f'Min and Max: {np.min(self.in_data)}, {np.max(self.in_data)}')

        output_path = os.path.join(os.getcwd(), "assignment1-signals", "outputs")
        self.logger.info('Trying to save audio')
        sf.write(os.path.join(output_path, "input.wav"), self.in_data, self.model.sample_rate)
        sf.write(os.path.join(output_path, "output.wav"), self.out_data, self.model.sample_rate)
        self.logger.info(f'Input audio saved to {os.path.join(os.getcwd(), output_path)}')
        self.logger.info(f'Exiting')
        self.logger.removeHandler(self.handler)
        self.logger.handlers = []
        self.logger = None
        return
    
    def on_save_button(self):
        self.paused = True
        self.logger.info(f'Clicked on save button')
        self.in_data = np.concatenate(self.in_data)
        self.in_data = np.clip(self.in_data, -1.0, 1.0)
        self.in_data = (self.in_data * 2**15).astype(np.int16)
        self.out_data = np.concatenate(self.out_data)
        self.out_data = np.clip(self.out_data, -1.0, 1.0)
        self.out_data = (self.out_data * 2**15).astype(np.int16)

        self.logger.info(f'Input audio shape: {self.in_data.shape}')
        self.logger.info(f'Output audio shape: {self.out_data.shape}')
        self.logger.info(f'Dtype: {self.in_data.dtype}')
        self.logger.info(f'Min and Max: {np.min(self.in_data)}, {np.max(self.in_data)}')

        output_path = os.path.join(os.getcwd(), "assignment1-signals", "outputs")
        self.logger.info('Trying to save audio')
        sf.write(os.path.join(output_path, "input.wav"), self.in_data, self.model.sample_rate)
        sf.write(os.path.join(output_path, "output.wav"), self.out_data, self.model.sample_rate)
        self.logger.info(f'Input audio saved to {os.path.join(os.getcwd(), output_path)}')
        self.in_data = []
        self.out_data = []
        return
        