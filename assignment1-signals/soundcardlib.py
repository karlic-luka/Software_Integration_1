from __future__ import division

import numpy as np
import pyaudio


# def data_to_array(data, channels):
#     return (np.frombuffer(data, dtype=np.int16)
#             .reshape((-1, channels))
#             .astype(float) / 2**15)
def data_to_array(data, channels):
    return np.frombuffer(data, dtype=np.float32).reshape((-1, channels)).astype(float)

class SoundCardDataSource(object):
    def __init__(self, num_chunks, channels=1, sampling_rate=44100, chunk_size=1024):
        self.fs = sampling_rate
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.num_chunks = int(num_chunks)

        # Check format is supported
        # store the data for later use
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None

    def connect_and_start_streaming(self, index):
        dev = self.pyaudio.get_device_info_by_index(index)
        print(f'Connecting to device {dev["name"]}')
        print(f'Device info: {dev.items()}')
        # self.channels = min(dev["maxInputChannels"], 2) # NOTE: edited by Luka - denoiser
        if self.channels < 1:
            raise RuntimeError("The audio device does not support any input channels")
        
        print(f'Using {self.channels} channels')
        if not self.pyaudio.is_format_supported(
            rate=self.fs,
            input_device=dev["index"],
            input_channels=self.channels,
            input_format=pyaudio.paFloat32, # NOTE: paFloat32 for denoiser - doesn't work
        ):
            print(f'rate={self.fs}, input_device={dev["index"]}, input_channels={self.channels}, input_format=pyaudio.paFloat32')
            raise RuntimeError("Unsupported audio format or rate")

        # Allocate buffers
        self._allocate_buffer()

        # Callback function is called with new audio data
        def callback(in_data, frame_count, time_info, status):
            samples = data_to_array(in_data, self.channels)
            self._write_chunk(samples)
            return (None, pyaudio.paContinue)

        # Start the stream
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            frames_per_buffer=self.chunk_size,
            rate=self.fs,
            stream_callback=callback,
            input=True,
        )

    def __del__(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()

    def _write_chunk(self, samples):
        self.buffer[self.next_chunk, :, :] = samples
        self.next_chunk = (self.next_chunk + 1) % self.buffer.shape[0]

    def _allocate_buffer(self):
        self.buffer = np.zeros((self._num_chunks, self.chunk_size, self.channels))
        self.next_chunk = 0

    @property
    def num_chunks(self):
        return self._num_chunks

    @num_chunks.setter
    def num_chunks(self, num_chunks):
        n = max(1, int(num_chunks))
        if n * self.chunk_size > 2**16:
            n = 2**16 // self.chunk_size
        self._num_chunks = n
        self._allocate_buffer()

    def get_buffer(self):
        """Return all chunks joined together"""
        a = self.buffer[: self.next_chunk]
        b = self.buffer[self.next_chunk :]
        return np.concatenate((b, a), axis=0).reshape(
            (self.buffer.shape[0] * self.buffer.shape[1], self.buffer.shape[2])
        )

    @property
    def timeValues(self):
        N = self.buffer.shape[0] * self.buffer.shape[1]
        return np.linspace(0, N / self.fs, N)

    def get_available_devices(self):
        input_devices_dict = {}
        output_devices_dict = {}
        # list output devices
        
        for i in range(self.pyaudio.get_device_count()):
            dev = self.pyaudio.get_device_info_by_index(i)
            channels_in = min(dev["maxInputChannels"], 2) # NOTE: edited by Luka - denoiser
            # print(f'Index: {i}, Name: {dev["name"]}')
            try:
                if self.pyaudio.is_format_supported(
                    rate=self.fs,
                    input_device=dev["index"],
                    input_channels=channels_in,
                    input_format=pyaudio.paFloat32,
                ):
                    input_devices_dict[dev["name"]] = i
            except ValueError:
                print(f'INPUT: Index: {i}, Name: {dev["name"]} (not supported)')
            try:
                channels_out = min(dev["maxOutputChannels"], 2) # NOTE: edited by Luka - denoiser
                if self.pyaudio.is_format_supported(
                    rate=self.fs,
                    output_device=dev["index"],
                    output_channels=channels_out,
                    output_format=pyaudio.paFloat32,
                ):
                    output_devices_dict[dev["name"]] = i
            except ValueError:
                print(f'OUTPUT: Index: {i}, Name: {dev["name"]} (not supported)')
        return input_devices_dict, output_devices_dict
    