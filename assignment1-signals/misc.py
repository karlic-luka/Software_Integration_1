import numpy as np


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
