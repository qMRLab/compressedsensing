import numpy as np


def compute_kspace(signal: np.ndarray) -> np.ndarray:
    """Compute the DFT (k-space) of a 1D signal sampled at t = 0, 1, …, N-1.

    Uses the standard DFT convention::

        X[k] = Σ_{n=0}^{N-1}  x[n] · exp(-j · 2π · k · n / N)

    This matches MRI k-space encoding where each sample is a Fourier
    coefficient measured at a discrete spatial frequency.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued 1D signal of length N.

    Returns
    -------
    np.ndarray (complex)
        DFT coefficients X[0], …, X[N-1].  DC component is at index 0.
    """
    return np.fft.fft(signal)


def make_sparse_signal(length: int, positions, heights) -> np.ndarray:
    """Return a 1D sparse signal of the given length.

    Parameters
    ----------
    length : int
        Number of samples in the output array.
    positions : array-like of int
        Indices at which the signal is non-zero.
    heights : array-like of float
        Amplitude at each corresponding position.

    Returns
    -------
    np.ndarray
        Zero-filled array of shape (length,) with non-zero values at positions.

    Raises
    ------
    ValueError
        If positions and heights have different lengths, or any position is
        outside [0, length).
    """
    positions = list(positions)
    heights = list(heights)

    if len(positions) != len(heights):
        raise ValueError(
            f"positions and heights must have the same length "
            f"(got {len(positions)} and {len(heights)})"
        )

    for p in positions:
        if p < 0 or p >= length:
            raise ValueError(
                f"Position {p} is out of bounds for signal of length {length}"
            )

    signal = np.zeros(length)
    for p, h in zip(positions, heights):
        signal[p] = h

    return signal
