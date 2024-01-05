# Assignment1: Real-Time FFT Window

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

## Resources
@inproceedings{defossez2020real,
  title={Real Time Speech Enhancement in the Waveform Domain},
  author={Defossez, Alexandre and Synnaeve, Gabriel and Adi, Yossi},
  booktitle={Interspeech},
  year={2020}
}
# Assignment2: PCA 3D Morphing
This project is a software integration assignment that focuses on 3D PCA (Principal Component Analysis) for analyzing and manipulating 3D models. The main file, **assignment2.py**, serves as the entry point and incorporates several libraries, including PyQt5, OpenGL, numpy, scipy, and imageio. 

## Functionality
The project utilizes PCA to analyze and manipulate 3D models, offering users the ability to adjust texture and geometry weights. Notably, it includes **multithreading** functionality for efficient texture and geometry processing. The code showcases the integration of various libraries and provides a user-friendly interface for parameter adjustments. Users can manipulate weights for eigenvectors of the PCA using sliders in the separate parameters window.

## Dependencies
- python 3.8.18
- see requirements.txt
  
## Usage
1. Clone the directory.
   - `git clone https://github.com/karlic-luka/Software_Integration_1.git`
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Position yourself at the root directory
5. Start the application from the **root** directory
   - `python assignment2_3D_PCA\assignment2.py`
   - otherwise, there could be some problems with paths
6. Load the model1.obj by clicking on **Load paths**
7. Click **Process** to process PCA on texture and geometry
8. Play around with the sliders (SPOILER ALERT: geometry sliders are really fun!)
     - I challenge you to make as funny face as possible
9. When you're done, you can save the new model by clicking on **Save 3D Face**

