import numpy as np
import argparse
from denoiser.pretrained import add_model_flags

# Based on a function from numpy 1.8, not all versions have it, so we have to redefine it in case
# rfft: return the discrete Fourier transform frequencies
def rfftfreq(n, d=1.0):
    if not isinstance(n, int):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


# apply the fft on the data from the buffer
def fft_buffer(x):
    window = np.hanning(x.shape[0])
    # Calculate FFT
    fx = np.fft.rfft(window * x)
    # Convert to normalised PSD
    Pxx = abs(fx) ** 2 / (np.abs(window) ** 2).sum()
    # Scale for one-sided
    Pxx[1:-1] *= 2
    return Pxx**0.5, fx


def ifft_buffer(fhat, threshold=0):
    n = len(fhat)
    PSD = fhat * np.conj(fhat) / n
    indices = PSD > threshold  # find all freqs with large enough power
    # PSDclean = PSD * indices  # zero out all small Fourier coeffs. in Y
    fhat = indices * fhat  # zero out small Fourier coeffs. in Y
    # fhat signal
    inverse_fft = np.fft.irfft(fhat)
    return inverse_fft

def get_parser():
    parser = argparse.ArgumentParser(
        "denoiser.live",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="Soundflower (2ch)",
        help="name or index of output interface.")
    add_model_flags(parser)
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="cpu")
    parser.add_argument(
        "--dry", type=float, default=0.04,
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04")
    parser.add_argument(
        "-t", "--num_threads", type=int,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")
    return parser