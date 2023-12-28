import numpy as np
import argparse
from denoiser.pretrained import add_model_flags

# Based on a function from numpy 1.8, not all versions have it, so we have to redefine it in case
# rfft: return the discrete Fourier transform frequencies
def rfftfreq(n, d=1.0):
    """
    Calculate the frequencies of the discrete Fourier transform (DFT) for a real-valued input signal.

    Parameters:
        n (int): Length of the input signal.
        d (float, optional): Sample spacing (inverse of the sampling rate). Defaults to 1.0.

    Returns:
        ndarray: Array of length n//2 + 1 containing the DFT frequencies.
    """
    if not isinstance(n, int):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


def fft_buffer(x):
    """
    Apply the Fast Fourier Transform (FFT) on the data from the buffer.

    Parameters:
        x (ndarray): Input signal.

    Returns:
        tuple: A tuple containing the square root of the power spectral density (PSD) and the FFT coefficients.
    """
    window = np.hanning(x.shape[0])
    fx = np.fft.rfft(window * x)
    Pxx = abs(fx) ** 2 / (np.abs(window) ** 2).sum()
    Pxx[1:-1] *= 2
    return Pxx**0.5, fx


def ifft_buffer(fhat, threshold=0):
    """
    Apply the Inverse Fast Fourier Transform (IFFT) on the given FFT coefficients.

    Parameters:
        fhat (ndarray): FFT coefficients.
        threshold (float, optional): Threshold for zeroing out small Fourier coefficients. Defaults to 0.

    Returns:
        ndarray: Inverse FFT of the given coefficients.
    """
    n = len(fhat)
    PSD = fhat * np.conj(fhat) / n
    indices = PSD > threshold
    fhat = indices * fhat
    inverse_fft = np.fft.irfft(fhat)
    return inverse_fft


def get_parser():
    """
    Create an argument parser for the denoiser.live module.

    Returns:
        ArgumentParser: An instance of the ArgumentParser class.
    """
    parser = argparse.ArgumentParser(
        "main_gui",
        description="Real-time noise suppression using a pre-trained model."
        )
    add_model_flags(parser)
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping."
    )
    parser.add_argument(
        "--dry", type=float, default=0.04,
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04"
    )
    parser.add_argument(
        "-t", "--num_threads", type=int, default=1,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance."
    )
    parser.add_argument(
        "-f", "--num_frames", type=int, default=4,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed."
    )
    return parser

