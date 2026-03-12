import numpy as np


def compute_kspace(signal: np.ndarray) -> np.ndarray:
    """Compute the DFT (k-space) of a 1D signal sampled at t = 0, 1, …, N-1.

    Uses the standard DFT convention::

        X[k] = Σ_{n=0}^{N-1}  x[n] · exp(-j · 2π · k · n / N)

    This matches MRI k-space encoding where each sample is a Fourier
    coefficient measured at a discrete spatial frequency.

    **Output convention — fftshift centering**

    ``np.fft.fftshift`` is applied so that the DC component (k = 0) sits at
    array index ``N//2``, not at index 0.  The k-axis therefore runs from
    ``-N//2`` (index 0) to ``N//2 - 1`` (index N-1) for even N, or from
    ``-(N-1)//2`` to ``+(N-1)//2`` for odd N.

    **DC / flat-spectrum duality**

    Two complementary facts follow directly from DFT theory:

    1. *Spike at t = 0 → flat k-space.*
       ``signal[0] = A``, all others zero  →  ``X[k] = A`` for every k.
       The magnitude spectrum is flat (equal to A) and the phase is zero
       everywhere.  This is because the DFT basis function for k evaluated at
       n = 0 is always ``exp(0) = 1``.

    2. *Constant signal → DC spike in k-space.*
       ``signal[n] = A`` for all n  →  ``X[k=0] = N·A``, ``X[k≠0] = 0``.
       The only non-zero bin is the DC bin at index ``N//2`` after fftshift.

    More generally, *any* single spike at time ``t = t₀`` produces a
    **flat magnitude spectrum** (|X[k]| = |A| for all k) with a **linear
    phase ramp** (∠X[k] = −2π·k·t₀/N).  The DC/flat duality for t₀ = 0
    is just the special case where the phase ramp has zero slope.

    **Even vs. odd N**

    * **Even N**: The k-axis is *asymmetric*: ``k ∈ {-N/2, …, -1, 0, 1, …,
      N/2-1}``.  The Nyquist bin at ``k = -N/2`` (index 0 after fftshift)
      has no positive-frequency counterpart at ``k = +N/2``.
    * **Odd N**: The k-axis is *symmetric*: ``k ∈ {-(N-1)/2, …, 0, …,
      +(N-1)/2}``.  Every negative frequency bin has an exact positive mirror.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued 1D signal of length N.  ``signal[0]`` corresponds to
        time t = 0.

    Returns
    -------
    np.ndarray (complex)
        DFT coefficients, fftshift-centered.  DC component (k = 0) is at
        index ``N//2``.
    """
    return np.fft.fftshift(np.fft.fft(signal))


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
