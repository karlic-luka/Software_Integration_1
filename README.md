# Real-Time FFT Window

This is a real-time FFT window for audio signal processing. It delivers a graphical interface for real-time audio signal processing, including denoising using a pre-trained model. 
The GUI is built using PyQt5 and pyqtgraph, and it relies on external libraries such as numpy, torch, soundfile and PyAudio for audio processing.

## Features

- Real-time audio signal processing
- Denoising using a pre-trained model
- Graphical interface for plotting audio signals
- Adding artificial white Gaussian noise to the audio signal
- Saving input and output audio files
- Playing (saved) input and output audio files

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Position yourself at the root directory
3. Start the application from the **root** directory
   - `python assignment1-signals\main_gui.py`
   - additional command line arguments:
       -  `--dns48` or `--dns64` or `--master64` or `--valentini_nc`
           - different pre-trained models
       -  `--dry <float>` between 0 and 1
           - level of noise removal, where 0 is maximum noise removal, but it can cause distortion. 0.04 is default value
5. Connect to the desired input and output audio devices.
6. Click the **"Save**" button to save the input and output audio files.
7. Click the **"Listen to Audio"** to load "input.wav" and "output.wav" files.
   - input.wav contains *original* signal recorded with the microphone + the AWGN
   - output.wav contains *denoised* signal
8. In the **asignment1-signals/logs** folder you can find the .log file for each run of the application

## Dependencies
- python 3.8.18
- see requirements.txt

## NOTE
- https://github.com/facebookresearch/denoiser/tree/main/denoiser

Resources:
@inproceedings{defossez2020real,
  title={Real Time Speech Enhancement in the Waveform Domain},
  author={Defossez, Alexandre and Synnaeve, Gabriel and Adi, Yossi},
  booktitle={Interspeech},
  year={2020}
}
