"""
Figure 3 — 2D MRI Compressed Sensing Illustration.

Inspired by fig_02_interactive_v2.py but extended to 2D MRI data.
"""

import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt

try:
    import nibabel as nib
except ImportError:
    nib = None


# ── defaults ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIENTATIONS = {
    "sagittal": DATA_DIR / "brain_sagittal_slice.nii.gz",
}
DEFAULT_ORIENTATION = "sagittal"
WAVELET = "db4"
WAVELET_LEVEL = 3
R_FACTOR = 2
R_VALUES = [2, 3]
DEFAULT_R = 2
# Map R to approximate % sampled (rounded to nearest 5)
R_TO_PCT = {2: 50, 3: 35}
CENTER_LINES = 24
RNG_SEED = 42


def load_brain_slice(path, scale=0.75):
    """Load the 2D brain slice from a NIfTI file, downsampled via FFT cropping."""
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")
    img = nib.load(str(path))
    data = img.get_fdata()[:, :, 0].astype(np.float32)
    # Downsample by cropping in k-space (anti-aliased)
    if scale is not None and scale < 1.0:
        ky, kx = data.shape
        ty = int(ky * scale) // 2 * 2  # keep even
        tx = int(kx * scale) // 2 * 2
        ks = np.fft.fftshift(np.fft.fft2(data))
        cy, cx = ky // 2, kx // 2
        cropped = ks[cy - ty // 2 : cy + ty // 2,
                     cx - tx // 2 : cx + tx // 2]
        data = np.abs(np.fft.ifft2(np.fft.ifftshift(cropped))).astype(np.float32)
        # Rescale to preserve intensity range
        data *= (ky * kx) / (ty * tx)
    return data


def compute_kspace_magnitude(brain):
    """Compute log10(|FFTshift(FFT2(brain))| + 1) for display."""
    kspace = np.fft.fft2(brain)
    kspace_shifted = np.fft.fftshift(kspace)
    return np.log10(np.abs(kspace_shifted) + 1)


SAMPLING_MODES = ["random", "uniform"]
DEFAULT_SAMPLING = "random"
US_DIMS_OPTIONS = [1, 2]
DEFAULT_US_DIMS = 2


def _gen_pdf_1d(n_lines, p, pctg, radius=0.0):
    """Generate a 1D variable-density sampling PDF.

    Re-implemented from .external/sparseMRI_v0.2/utils/genPDF.m (Lustig 2007).
    """
    r = np.abs(np.linspace(-1, 1, n_lines))
    idx = r < radius
    target = int(np.floor(pctg * n_lines))

    minval, maxval = 0.0, 1.0
    for _ in range(100):
        val = (minval + maxval) / 2.0
        pdf = (1 - r) ** p + val
        pdf = np.clip(pdf, 0, 1)
        pdf[idx] = 1.0
        n_samp = int(np.floor(np.sum(pdf)))
        if n_samp > target:
            maxval = val
        elif n_samp < target:
            minval = val
        else:
            break
    return pdf


def _gen_sampling_1d(pdf, n_iter=20, tol=2, seed=RNG_SEED):
    """Generate a 1D sampling mask with minimum peak PSF sidelobe.

    Re-implemented from .external/sparseMRI_v0.2/utils/genSampling.m (Lustig 2007).
    """
    target_k = np.sum(pdf)
    best_mask = None
    best_intr = np.inf
    rng = np.random.default_rng(seed)

    for _ in range(n_iter):
        for _retry in range(1000):
            mask = rng.random(len(pdf)) < pdf
            if abs(mask.sum() - target_k) <= tol:
                break
        psf = np.fft.ifft(mask.astype(float) / np.where(pdf > 0, pdf, 1.0))
        peak = np.max(np.abs(psf[1:]))
        if peak < best_intr:
            best_intr = peak
            best_mask = mask.copy()

    return best_mask


def _gen_pdf_2d(im_shape, p, pctg, radius=0.0):
    """Generate a 2D variable-density sampling PDF.

    Re-implemented from .external/sparseMRI_v0.2/utils/genPDF.m (Lustig 2007).
    pdf = (1-r)^p + val, clamped to [0,1], where r is L2 distance from centre
    normalised to [0,1], and val is found by bisection to hit target count.
    """
    sy, sx = im_shape
    y, x = np.meshgrid(np.linspace(-1, 1, sx), np.linspace(-1, 1, sy))
    r = np.sqrt(x ** 2 + y ** 2)
    r = r / np.max(np.abs(r))
    idx = r < radius
    target = int(np.floor(pctg * sy * sx))

    minval, maxval = 0.0, 1.0
    for _ in range(100):
        val = (minval + maxval) / 2.0
        pdf = (1 - r) ** p + val
        pdf = np.clip(pdf, 0, 1)
        pdf[idx] = 1.0
        n_samp = int(np.floor(np.sum(pdf)))
        if n_samp > target:
            maxval = val
        elif n_samp < target:
            minval = val
        else:
            break
    return pdf


def _gen_sampling_2d(pdf, n_iter=10, tol=None, seed=RNG_SEED):
    """Generate a 2D sampling mask with minimum peak PSF sidelobe.

    Re-implemented from .external/sparseMRI_v0.2/utils/genSampling.m (Lustig 2007).
    """
    if tol is None:
        tol = max(10, int(np.sum(pdf) * 0.02))
    target_k = np.sum(pdf)
    best_mask = None
    best_intr = np.inf
    rng = np.random.default_rng(seed)

    for _ in range(n_iter):
        for _retry in range(1000):
            mask = rng.random(pdf.shape) < pdf
            if abs(mask.sum() - target_k) <= tol:
                break
        psf = np.fft.ifft2(mask.astype(float) / np.where(pdf > 0, pdf, 1.0))
        peak = np.max(np.abs(psf.ravel()[1:]))
        if peak < best_intr:
            best_intr = peak
            best_mask = mask.copy()

    return best_mask


def make_sampling_mask(im_shape, mode="random", R=R_FACTOR, center=CENTER_LINES,
                       seed=RNG_SEED, us_dims=2):
    """Return (mask2d, pdf2d) — 2D boolean mask and sampling PDF.

    us_dims=2: undersample individual k-space points (2D).
    us_dims=1: undersample phase-encode lines (columns) only (1D).
    mode="random": variable-density PDF + Monte Carlo PSF optimisation.
    mode="uniform": regular grid/line subsampling.
    """
    sy, sx = im_shape
    if us_dims == 1:
        # 1D: undersample along columns (phase-encode = short axis)
        if mode == "uniform":
            mask1d = np.zeros(sx, dtype=bool)
            mask1d[::R] = True
            pdf1d = np.full(sx, 1.0 / R)
            pdf1d[mask1d] = 1.0
        else:
            pctg = 1.0 / R
            radius = center / sx
            pdf1d = _gen_pdf_1d(sx, p=5, pctg=pctg, radius=radius)
            mask1d = _gen_sampling_1d(pdf1d, n_iter=20, tol=2, seed=seed)
        # Broadcast to 2D
        mask2d = np.tile(mask1d[np.newaxis, :], (sy, 1))
        pdf2d = np.tile(pdf1d[np.newaxis, :], (sy, 1))
    else:
        # 2D: undersample individual points
        if mode == "uniform":
            mask2d = np.zeros(im_shape, dtype=bool)
            mask2d[::R, ::R] = True
            pdf2d = np.full(im_shape, 1.0 / (R * R))
            pdf2d[mask2d] = 1.0
        else:
            pctg = 1.0 / R
            radius = center / max(sy, sx)
            pdf2d = _gen_pdf_2d(im_shape, p=5, pctg=pctg, radius=radius)
            mask2d = _gen_sampling_2d(pdf2d, n_iter=10, tol=None, seed=seed)
    return mask2d, pdf2d


def make_sampling_overlay(mask2d):
    """Build a 2D array for the red sampling overlay.

    Sampled points → 1.0, unsampled → NaN (transparent).
    """
    overlay = np.full(mask2d.shape, np.nan)
    overlay[mask2d] = 1.0
    return overlay


def undersampled_kspace(brain, mask2d):
    """Return the zero-filled undersampled k-space (shifted, complex).

    mask2d is a 2D boolean array matching the shifted k-space shape.
    """
    kspace = np.fft.fft2(brain)
    kspace_shifted = np.fft.fftshift(kspace)
    undersampled = np.zeros_like(kspace_shifted)
    undersampled[mask2d] = kspace_shifted[mask2d]
    return undersampled


def zero_filled_recon(brain, mask2d):
    """Reconstruct image from undersampled k-space via zero-filling."""
    us = undersampled_kspace(brain, mask2d)
    recon = np.fft.ifft2(np.fft.ifftshift(us))
    return np.abs(recon)


def compute_wavelet_map(brain, wavelet=WAVELET, level=WAVELET_LEVEL):
    """Compute 2D DWT and assemble coefficients into a single image for display."""
    coeffs = pywt.wavedec2(brain, wavelet, level=level)
    arr, _ = pywt.coeffs_to_array(coeffs)
    return arr


def soft_threshold_wavelet(image, threshold_factor=0.03, wavelet=WAVELET,
                           level=WAVELET_LEVEL):
    """Soft-threshold wavelet coefficients of image.

    threshold = threshold_factor * max(|coefficients|)

    Returns (thresholded_map, reconstructed_image) where thresholded_map
    is the assembled coefficient array for display.
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # Flatten all coefficients to find threshold
    all_coeffs = np.concatenate(
        [coeffs[0].ravel()] +
        [d.ravel() for detail in coeffs[1:] for d in detail]
    )
    threshold = threshold_factor * np.max(np.abs(all_coeffs))
    # Soft threshold: sign(x) * max(|x| - threshold, 0)
    def _soft(c):
        return np.sign(c) * np.maximum(np.abs(c) - threshold, 0)
    thresholded = [_soft(coeffs[0])]
    for detail in coeffs[1:]:
        thresholded.append(tuple(_soft(d) for d in detail))
    # Assemble for display
    arr, _ = pywt.coeffs_to_array(thresholded)
    # Reconstruct image
    recon = pywt.waverec2(thresholded, wavelet)
    return arr, recon


# ── Nonlinear Conjugate Gradient CS reconstruction ───────────────────────
# Re-implemented from .external/sparseMRI_v0.2/fnlCg.m (Lustig 2007).
# Minimises: ||F_u x - y||^2 + xfmWeight*|Ψx|_1 + TVWeight*TV(x)
# using nonlinear CG with backtracking line search.

# Default parameters from demo_Brain_2D.m / init.m
_CG_DEFAULTS = dict(
    TVWeight=0.002,
    xfmWeight=0.005,
    Itnlim=8,
    l1Smooth=1e-15,
    pNorm=1,
    lineSearchItnlim=150,
    lineSearchAlpha=0.01,
    lineSearchBeta=0.6,
    lineSearchT0=1.0,
    gradToll=1e-30,
)


def _fft2c(x):
    """Centred FFT2 (matches MATLAB fft2c)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x))) / np.sqrt(x.size)


def _ifft2c(x):
    """Centred IFFT2 (matches MATLAB ifft2c)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x))) * np.sqrt(x.size)


def _tv_forward(x):
    """Finite-difference TV operator D(x) → (Dx, Dy) stacked on axis 2."""
    dx = np.roll(x, -1, axis=0) - x
    dx[-1, :] = 0
    dy = np.roll(x, -1, axis=1) - x
    dy[:, -1] = 0
    return np.stack([dx, dy], axis=2)


def _tv_adjoint(y):
    """Adjoint of finite-difference TV operator."""
    dx = y[:, :, 0]
    dy = y[:, :, 1]
    # adjDx
    res_x = np.roll(dx, 1, axis=0) - dx
    res_x[0, :] = -dx[0, :]
    res_x[-1, :] = dx[-2, :]
    # adjDy
    res_y = np.roll(dy, 1, axis=1) - dy
    res_y[:, 0] = -dy[:, 0]
    res_y[:, -1] = dy[:, -2]
    return res_x + res_y


def _wavelet_forward(x, wavelet=WAVELET, level=WAVELET_LEVEL):
    """Forward wavelet transform, returns flat coefficient vector."""
    coeffs = pywt.wavedec2(x.real, wavelet, level=level)
    arr, slices = pywt.coeffs_to_array(coeffs)
    if np.iscomplexobj(x):
        coeffs_i = pywt.wavedec2(x.imag, wavelet, level=level)
        arr_i, _ = pywt.coeffs_to_array(coeffs_i)
        arr = arr + 1j * arr_i
    return arr, slices


def _wavelet_inverse(arr, slices, wavelet=WAVELET):
    """Inverse wavelet transform from flat coefficient array."""
    coeffs = pywt.array_to_coeffs(arr.real, slices, output_format='wavedec2')
    out = pywt.waverec2(coeffs, wavelet)
    if np.iscomplexobj(arr):
        coeffs_i = pywt.array_to_coeffs(arr.imag, slices, output_format='wavedec2')
        out = out + 1j * pywt.waverec2(coeffs_i, wavelet)
    return out


def fnlCg(x0, mask2d, data, wavelet=WAVELET, level=WAVELET_LEVEL, **kwargs):
    """Nonlinear conjugate gradient CS reconstruction.

    Re-implemented from .external/sparseMRI_v0.2/fnlCg.m (Lustig 2007).

    Parameters
    ----------
    x0 : 2D complex array — initial image estimate (e.g. zero-filled recon)
    mask2d : 2D bool array — k-space sampling mask (True = sampled)
    data : 2D complex array — measured (undersampled) k-space data
    wavelet, level : wavelet parameters
    **kwargs : override any of _CG_DEFAULTS

    Returns
    -------
    x : reconstructed image (complex)
    """
    p = {**_CG_DEFAULTS, **kwargs}
    x = x0.copy().astype(complex)

    # Pre-compute wavelet slices structure
    _, w_slices = _wavelet_forward(x, wavelet, level)

    def fwd_ft(img):
        return _fft2c(img) * mask2d

    def adj_ft(d):
        return _ifft2c(d * mask2d)

    def grad_data(img):
        return 2.0 * adj_ft(fwd_ft(img) - data)

    def grad_xfm(img):
        """Gradient of L1 wavelet penalty (smoothed)."""
        w, _ = _wavelet_forward(img, wavelet, level)
        g = p['pNorm'] * w * (w * np.conj(w) + p['l1Smooth']) ** (p['pNorm'] / 2 - 1)
        return _wavelet_inverse(g, w_slices, wavelet)

    def grad_tv(img):
        """Gradient of TV penalty (smoothed)."""
        dx = _tv_forward(img)
        g = p['pNorm'] * dx * (dx * np.conj(dx) + p['l1Smooth']) ** (p['pNorm'] / 2 - 1)
        return _tv_adjoint(g)

    def gradient(img):
        g = grad_data(img)
        if p['xfmWeight']:
            g = g + p['xfmWeight'] * grad_xfm(img)
        if p['TVWeight']:
            g = g + p['TVWeight'] * grad_tv(img)
        return g

    def objective(img):
        """Compute total objective value."""
        obj = np.sum(np.abs(fwd_ft(img) - data) ** 2)
        if p['xfmWeight']:
            w, _ = _wavelet_forward(img, wavelet, level)
            obj += p['xfmWeight'] * np.sum(
                (w * np.conj(w) + p['l1Smooth']).real ** (p['pNorm'] / 2))
        if p['TVWeight']:
            dx = _tv_forward(img)
            obj += p['TVWeight'] * np.sum(
                (dx * np.conj(dx) + p['l1Smooth']).real ** (p['pNorm'] / 2))
        return obj.real

    # CG iterations
    g0 = gradient(x)
    dx = -g0
    t0 = p['lineSearchT0']

    for k in range(p['Itnlim']):
        # Backtracking line search
        f0 = objective(x)
        t = t0
        f1 = objective(x + t * dx)
        lsiter = 0
        while (f1 > f0 - p['lineSearchAlpha'] * t *
               np.abs(np.real(np.vdot(g0, dx))) ** 2 and
               lsiter < p['lineSearchItnlim']):
            lsiter += 1
            t *= p['lineSearchBeta']
            f1 = objective(x + t * dx)

        if lsiter > 2:
            t0 *= p['lineSearchBeta']
        if lsiter < 1:
            t0 /= p['lineSearchBeta']

        x = x + t * dx

        # Conjugate gradient direction
        g1 = gradient(x)
        bk = np.real(np.vdot(g1, g1)) / (np.real(np.vdot(g0, g0)) + 1e-30)
        g0 = g1
        dx = -g1 + bk * dx

        if np.linalg.norm(dx) < p['gradToll']:
            break

    return x


def cs_reconstruct(brain, mask2d, pdf2d, n_outer=5, wavelet=WAVELET,
                   level=WAVELET_LEVEL, **kwargs):
    """Full CS reconstruction matching demo_Brain_2D.m workflow.

    Runs n_outer calls to fnlCg (each doing Itnlim=8 CG iterations).
    Density compensation: data .* mask ./ pdf (as in demo_Brain_2D.m).
    """
    # Measured data with density compensation: data .* mask ./ pdf
    data = _fft2c(brain) * mask2d / np.where(pdf2d > 0, pdf2d, 1.0)

    # Zero-filled DC image as starting point
    im_dc = _ifft2c(data * mask2d)
    # Normalise
    scale = np.max(np.abs(im_dc))
    if scale > 0:
        data = data / scale
        im_dc = im_dc / scale

    x = im_dc.copy()
    for _ in range(n_outer):
        x = fnlCg(x, mask2d, data, wavelet=wavelet, level=level, **kwargs)

    return np.abs(x) * scale


def data_consistency_kspace(brain, mask2d, thresh_recon):
    """Combine original acquired k-space with thresholded recon's k-space.

    Acquired points (mask2d=True): use original k-space data.
    Unacquired points: use k-space from the thresholded reconstruction.
    Returns the combined shifted k-space (complex).
    """
    orig_kspace = np.fft.fftshift(np.fft.fft2(brain))
    thresh_kspace = np.fft.fftshift(np.fft.fft2(thresh_recon))
    combined = thresh_kspace.copy()
    combined[mask2d] = orig_kspace[mask2d]
    return combined


def make_unacquired_overlay(mask2d):
    """Build a 2D array highlighting UNacquired points (green overlay).

    Unsampled → 1.0, sampled → NaN (transparent).
    """
    overlay = np.full(mask2d.shape, np.nan)
    overlay[~mask2d] = 1.0
    return overlay


def _build_fig(brain, kspace_mag, wavelet_map, sampling_overlay,
               us_kspace_mag, recon, recon_wavelet_map,
               thresholded_wavelet_map, thresh_kspace_mag,
               unacquired_overlay, combined_kspace_mag,
               iter1_recon, final_recon):
    """Build a 5×3 Plotly figure.

    Row 1: (a) k-space + red overlay,       (b) brain image,       (c) wavelet transform
    Row 2: (d) zero-filled k-space,          (e) zero-filled recon, (f) recon wavelet
    Row 3: (g) thresh k-space + green gaps,  (h) empty,             (i) soft-thresholded wavelet
    Row 4: (j) combined k-space + overlays,  (k) 1 CG iteration,   (l) empty
    Row 5: empty,                            (m) 5×8 CG recon,     empty
    """
    TITLE_SIZE = 24

    fig = make_subplots(
        rows=5, cols=3,
        subplot_titles=[
            "(a) k-space", "(b) MRI brain image", "(c) wavelet transform",
            "(d) zero-filled k-space", "(e) zero-filled reconstruction", "(f) wavelet transform",
            "(g) k-space (from thresholded)", "", "(i) soft-thresholded wavelet",
            "(j) data consistency k-space", "(k) 1 CG iteration", "",
            "", "(l) 5×8 CG reconstruction", "",
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )

    for ann in fig.layout.annotations:
        ann.font.size = TITLE_SIZE

    # ── Row 1 ────────────────────────────────────────────────────────────────

    # (a) K-space magnitude — trace 0
    fig.add_trace(go.Heatmap(
        z=kspace_mag[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="kx=%{x}<br>ky=%{y}<br>log10|K|=%{z:.2f}<extra></extra>",
    ), row=1, col=1)

    # (a-overlay) Red sampling lines — trace 1
    fig.add_trace(go.Heatmap(
        z=sampling_overlay[::-1],
        colorscale=[[0, "red"], [1, "red"]],
        showscale=False,
        opacity=0.20,
        hoverinfo="skip",
    ), row=1, col=1)

    # (b) Brain image — trace 2
    fig.add_trace(go.Heatmap(
        z=brain[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>intensity=%{z:.1f}<extra></extra>",
    ), row=1, col=2)

    # (c) Wavelet transform — trace 3
    fig.add_trace(go.Heatmap(
        z=np.log(np.abs(wavelet_map[::-1]) + 1),
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>|coeff|=%{z:.2f}<extra></extra>",
    ), row=1, col=3)

    # ── Row 2 ────────────────────────────────────────────────────────────────

    # (d) Zero-filled k-space — trace 4
    fig.add_trace(go.Heatmap(
        z=us_kspace_mag[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="kx=%{x}<br>ky=%{y}<br>log10|K|=%{z:.2f}<extra></extra>",
    ), row=2, col=1)

    # (e) Zero-filled reconstruction — trace 5
    fig.add_trace(go.Heatmap(
        z=recon[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>intensity=%{z:.1f}<extra></extra>",
    ), row=2, col=2)

    # (f) Wavelet of reconstruction — trace 6
    fig.add_trace(go.Heatmap(
        z=np.log(np.abs(recon_wavelet_map[::-1]) + 1),
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>|coeff|=%{z:.2f}<extra></extra>",
    ), row=2, col=3)

    # ── Row 3 ────────────────────────────────────────────────────────────────

    # (g) K-space from thresholded wavelet recon — trace 7
    fig.add_trace(go.Heatmap(
        z=thresh_kspace_mag[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="kx=%{x}<br>ky=%{y}<br>log10|K|=%{z:.2f}<extra></extra>",
    ), row=3, col=1)

    # (g-overlay) Green highlights on unacquired lines — trace 8
    fig.add_trace(go.Heatmap(
        z=unacquired_overlay[::-1],
        colorscale=[[0, "green"], [1, "green"]],
        showscale=False,
        opacity=0.20,
        hoverinfo="skip",
    ), row=3, col=1)

    # (h) Empty — trace 9
    fig.add_trace(go.Heatmap(
        z=[[0]],
        colorscale=[[0, "white"], [1, "white"]],
        showscale=False,
        hoverinfo="skip",
    ), row=3, col=2)

    # (i) Soft-thresholded wavelet — trace 10
    fig.add_trace(go.Heatmap(
        z=np.log(np.abs(thresholded_wavelet_map[::-1]) + 1),
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>|coeff|=%{z:.2f}<extra></extra>",
    ), row=3, col=3)

    # ── Row 4 ────────────────────────────────────────────────────────────────

    # (j) Combined k-space — trace 11
    fig.add_trace(go.Heatmap(
        z=combined_kspace_mag[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="kx=%{x}<br>ky=%{y}<br>log10|K|=%{z:.2f}<extra></extra>",
    ), row=4, col=1)

    # (j-overlay) Red on acquired lines — trace 12
    fig.add_trace(go.Heatmap(
        z=sampling_overlay[::-1],
        colorscale=[[0, "red"], [1, "red"]],
        showscale=False,
        opacity=0.20,
        hoverinfo="skip",
    ), row=4, col=1)

    # (j-overlay) Green on unacquired (filled) lines — trace 13
    fig.add_trace(go.Heatmap(
        z=unacquired_overlay[::-1],
        colorscale=[[0, "green"], [1, "green"]],
        showscale=False,
        opacity=0.20,
        hoverinfo="skip",
    ), row=4, col=1)

    # (k) Iteration 1 reconstruction — trace 14
    fig.add_trace(go.Heatmap(
        z=iter1_recon[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>intensity=%{z:.1f}<extra></extra>",
    ), row=4, col=2)

    # (l) Empty — trace 15
    fig.add_trace(go.Heatmap(
        z=[[0]],
        colorscale=[[0, "white"], [1, "white"]],
        showscale=False,
        hoverinfo="skip",
    ), row=4, col=3)

    # ── Row 5 ────────────────────────────────────────────────────────────────

    # Empty left — trace 16
    fig.add_trace(go.Heatmap(
        z=[[0]],
        colorscale=[[0, "white"], [1, "white"]],
        showscale=False,
        hoverinfo="skip",
    ), row=5, col=1)

    # (m) Final reconstruction (N iterations) — trace 17
    fig.add_trace(go.Heatmap(
        z=final_recon[::-1],
        colorscale="Gray",
        showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>intensity=%{z:.1f}<extra></extra>",
    ), row=5, col=2)

    # Empty right — trace 18
    fig.add_trace(go.Heatmap(
        z=[[0]],
        colorscale=[[0, "white"], [1, "white"]],
        showscale=False,
        hoverinfo="skip",
    ), row=5, col=3)

    fig.update_layout(
        height=2100,
        autosize=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    for row in [1, 2, 3, 4, 5]:
        for col in [1, 2, 3]:
            suffix = "" if (row == 1 and col == 1) else str((row - 1) * 3 + col)
            xref = f"x{suffix}"
            fig.update_yaxes(scaleanchor=xref, scaleratio=1, row=row, col=col)
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

    # ── Double-sided arrows between row 1 subplots ────────────────────────
    # ax/ay are pixel offsets from arrowhead (positive ax = tail to the left)
    row1_y = 0.92
    arrow_gap = 40  # pixels between tail and head

    arrow_ab_left = 0.305
    arrow_ab_right = 0.35
    arrow_bc_left = 0.655
    arrow_bc_right = 0.70

    arrow_style = dict(
        showarrow=True,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="#333",
        arrowhead=2,
        xref="paper", yref="paper",
    )

    # a → b (arrow pointing right, tail to the left)
    fig.add_annotation(
        x=arrow_ab_right, y=row1_y + 0.006,
        ax=-arrow_gap, ay=0,
        text="", **arrow_style,
    )
    # b → a (arrow pointing left, tail to the right)
    fig.add_annotation(
        x=arrow_ab_left, y=row1_y - 0.006,
        ax=arrow_gap, ay=0,
        text="", **arrow_style,
    )
    # Label between a↔b
    fig.add_annotation(
        x=(arrow_ab_left + arrow_ab_right) / 2, y=row1_y + 0.03,
        text="FFT", showarrow=False,
        font=dict(size=20, color="#333"),
        xref="paper", yref="paper",
    )

    # b → c (arrow pointing right)
    fig.add_annotation(
        x=arrow_bc_right, y=row1_y + 0.006,
        ax=-arrow_gap, ay=0,
        text="", **arrow_style,
    )
    # c → b (arrow pointing left)
    fig.add_annotation(
        x=arrow_bc_left, y=row1_y - 0.006,
        ax=arrow_gap, ay=0,
        text="", **arrow_style,
    )
    # Label between b↔c
    fig.add_annotation(
        x=(arrow_bc_left + arrow_bc_right) / 2, y=row1_y + 0.03,
        text="Ψ", showarrow=False,
        font=dict(size=20, color="#333"),
        xref="paper", yref="paper",
    )

    # ── Vertical arrow: a → d (Undersampling) ────────────────────────────
    col1_x = 0.157  # center of col 1 in paper space
    row1_bottom = 0.85
    row2_top = 0.81
    fig.add_annotation(
        x=col1_x, y=row2_top,
        ax=0, ay=-arrow_gap,
        text="", **arrow_style,
    )
    # Label to the left of the arrow
    fig.add_annotation(
        x=col1_x + 0.005, y=(row1_bottom + row2_top) / 2,
        text="Undersampling", showarrow=False,
        font=dict(size=20, color="#333"),
        xref="paper", yref="paper",
        textangle=0,
    )

    return fig


# Trace indices that change with sampling mode (row 1 ground-truth is fixed)
# 0=kspace, 1=red overlay, 2=brain, 3=wavelet are FIXED (row 1 ground truth)
# Everything from row 2 onward changes with sampling mode
# Traces 0,2,3 = row 1 (orientation-only); the rest depend on sampling+R too
ORIENT_TRACES = [0, 2, 3]
SAMPLING_TRACES = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def _round_z(z, decimals=1):
    """Round a z array for compact JSON."""
    if hasattr(z, "tolist"):
        return np.round(z, decimals).tolist()
    arr = np.array(z)
    return np.round(arr, decimals).tolist()


def _extract_orient_z(fig, decimals=1):
    """Extract orientation-only traces (row 1: k-space, brain, wavelet)."""
    all_z = []
    for trace in fig.data:
        if hasattr(trace, "z") and trace.z is not None:
            all_z.append(_round_z(trace.z, decimals))
    return {str(i): all_z[i] for i in ORIENT_TRACES if i < len(all_z)}


def _extract_sampling_z(fig, decimals=1):
    """Extract sampling-dependent traces (rows 2-4)."""
    all_z = []
    for trace in fig.data:
        if hasattr(trace, "z") and trace.z is not None:
            all_z.append(_round_z(trace.z, decimals))
    return {str(i): all_z[i] for i in SAMPLING_TRACES if i < len(all_z)}


def precompute(orientations=ORIENTATIONS):
    """Pre-compute figures for all orientations × sampling modes × R values.

    Returns (refs, orient_data, sampling_data) where:
      orient_data[orient_name]  = {trace_idx: z_data} for row 1 traces
      sampling_data[orient_samp_R] = {trace_idx: z_data} for rows 2-4 traces
    """
    orient_data = {}
    sampling_data = {}
    refs = {}
    for orient_name, path in orientations.items():
        brain = load_brain_slice(path)
        kspace_mag = compute_kspace_magnitude(brain)
        wavelet_map = compute_wavelet_map(brain)
        for samp_mode in SAMPLING_MODES:
            for R in R_VALUES:
                for ud in US_DIMS_OPTIONS:
                    key = f"{orient_name}_{samp_mode}_{R}_{ud}"
                    mask2d, pdf2d = make_sampling_mask(brain.shape, mode=samp_mode, R=R, us_dims=ud)
                    overlay = make_sampling_overlay(mask2d)
                    us_kspace = undersampled_kspace(brain, mask2d)
                    us_kspace_mag = np.log10(np.abs(us_kspace) + 1)
                    recon = zero_filled_recon(brain, mask2d)
                    recon_wavelet_map = compute_wavelet_map(recon)
                    thresh_wavelet_map, thresh_recon = soft_threshold_wavelet(recon)
                    thresh_kspace_mag = compute_kspace_magnitude(thresh_recon)
                    unacquired_overlay = make_unacquired_overlay(mask2d)
                    combined_ks = data_consistency_kspace(brain, mask2d, thresh_recon)
                    combined_kspace_mag = np.log10(np.abs(combined_ks) + 1)
                    # 1 CG iteration for row 4 display
                    iter1_recon = cs_reconstruct(brain, mask2d, pdf2d, n_outer=1, Itnlim=1)
                    # Full reconstruction (5 outer × 8 CG = 40 iterations)
                    final_recon = cs_reconstruct(brain, mask2d, pdf2d)
                    fig = _build_fig(brain, kspace_mag, wavelet_map, overlay,
                                     us_kspace_mag, recon, recon_wavelet_map,
                                     thresh_wavelet_map, thresh_kspace_mag,
                                     unacquired_overlay, combined_kspace_mag,
                                     iter1_recon, final_recon)
                    is_default = (orient_name == DEFAULT_ORIENTATION
                                  and samp_mode == DEFAULT_SAMPLING
                                  and R == DEFAULT_R
                                  and ud == DEFAULT_US_DIMS)
                    if is_default:
                        ref = json.loads(fig.to_json())
                        for trace in ref.get("data", []):
                            if "z" in trace and isinstance(trace["z"], list):
                                try:
                                    arr = np.array(trace["z"], dtype=float)
                                    trace["z"] = np.round(arr, 1).tolist()
                                except (ValueError, TypeError):
                                    pass
                        refs["default"] = ref
                    # Store row 1 once per orientation (same for all samp/R/ud)
                    if orient_name not in orient_data:
                        orient_data[orient_name] = _extract_orient_z(fig)
                    sampling_data[key] = _extract_sampling_z(fig)
    return refs, orient_data, sampling_data


def make_embeddable_html(orientations=ORIENTATIONS):
    """Return an HTML fragment for embedding in a MyST page via IPython.display.HTML."""
    refs, orient_data, sampling_data = precompute(orientations)
    orientation_names = list(orientations.keys())

    return f"""<style>
  .cs2d-fig-controls {{ display: flex; gap: 40px; justify-content: center;
               align-items: center; flex-wrap: wrap; margin: 8px 0 4px; }}
  .cs2d-fig-ctrl-group {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  .cs2d-fig-controls label {{ font-size: 14px; color: #444; font-weight: bold; }}
  .cs2d-fig-controls select {{ font-size: 14px; padding: 4px 10px; border-radius: 6px;
            border: 1px solid #ccc; cursor: pointer; background: white; }}
  #cs2d-fig {{ width: 100%; margin: 0 auto; }}
</style>

<div class="cs2d-fig-controls">
  <div class="cs2d-fig-ctrl-group">
    <label>Sampling</label>
    <select id="cs2d-samplingSelect">
      {"".join(f'<option value="{m}"{" selected" if m == DEFAULT_SAMPLING else ""}>{m.capitalize()}</option>' for m in SAMPLING_MODES)}
    </select>
  </div>
  <div class="cs2d-fig-ctrl-group">
    <label>% k-space sampled</label>
    <select id="cs2d-rSelect">
      {"".join(f'<option value="{r}"{" selected" if r == DEFAULT_R else ""}>{R_TO_PCT[r]}%</option>' for r in R_VALUES)}
    </select>
  </div>
  <div class="cs2d-fig-ctrl-group">
    <label>Undersampling dims</label>
    <select id="cs2d-udSelect">
      {"".join(f'<option value="{d}"{" selected" if d == DEFAULT_US_DIMS else ""}>{d}D</option>' for d in US_DIMS_OPTIONS)}
    </select>
  </div>
</div>

<div id="cs2d-fig"></div>

<script>
(function() {{
  function initFigure() {{
    const REF          = {json.dumps(refs["default"])};
    const ORIENT_DATA  = {json.dumps(orient_data)};
    const SAMP_DATA    = {json.dumps(sampling_data)};

    Plotly.newPlot("cs2d-fig", REF.data, REF.layout, {{responsive: true}});

    var figEl = document.getElementById("cs2d-fig");

    // Scale arrows and labels to plot width
    // Arrow annotations are indices 15-20 in layout.annotations
    // 15: a→b, 16: b→a, 17: FFT label, 18: b→c, 19: c→b, 20: Ψ label
    function scaleArrows() {{
      var plotWidth = figEl.offsetWidth || 800;
      var gap = Math.max(12, plotWidth * 0.035);
      var fontSize = Math.max(10, Math.min(18, plotWidth * 0.016));
      var anns = figEl.layout.annotations;
      if (!anns || anns.length < 21) return;
      // a→b (index 15): tail left of head
      anns[15].ax = -gap;
      // b→a (index 16): tail right of head
      anns[16].ax = gap;
      // FFT label (index 17)
      anns[17].font.size = fontSize;
      // b→c (index 18): tail left of head
      anns[18].ax = -gap;
      // c→b (index 19): tail right of head
      anns[19].ax = gap;
      // Ψ label (index 20)
      anns[20].font.size = fontSize;
      Plotly.relayout("cs2d-fig", {{ annotations: anns }});
    }}

    if (window.ResizeObserver) {{
      new ResizeObserver(function() {{
        Plotly.Plots.resize(figEl);
        scaleArrows();
      }}).observe(figEl);
    }}
    window.addEventListener("resize", function() {{
      Plotly.Plots.resize(figEl);
      scaleArrows();
    }});

    // Initial scale
    setTimeout(scaleArrows, 100);

    function updateFig() {{
      var orient = "{DEFAULT_ORIENTATION}";
      var samp   = document.getElementById("cs2d-samplingSelect").value;
      var rval   = document.getElementById("cs2d-rSelect").value;
      var udval  = document.getElementById("cs2d-udSelect").value;
      // Update rows 2-5 from sampling data
      var key = orient + "_" + samp + "_" + rval + "_" + udval;
      var sd = SAMP_DATA[key];
      if (sd) {{
        var si = Object.keys(sd);
        for (var k = 0; k < si.length; k++) {{
          Plotly.restyle("cs2d-fig", {{ z: [sd[si[k]]] }}, [parseInt(si[k])]);
        }}
      }}
    }}
    document.getElementById("cs2d-samplingSelect").addEventListener("change", updateFig);
    document.getElementById("cs2d-rSelect").addEventListener("change", updateFig);
    document.getElementById("cs2d-udSelect").addEventListener("change", updateFig);
  }}

  if (typeof Plotly !== "undefined") {{
    initFigure();
  }} else {{
    var s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    s.onload = initFigure;
    document.head.appendChild(s);
  }}
}})();
</script>"""


if __name__ == "__main__":
    Path(".tmp").mkdir(exist_ok=True)
    html = make_embeddable_html()
    Path(".tmp/fig_03_interactive_2d.html").write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{html}</body></html>"
    )
    print("saved → .tmp/fig_03_interactive_2d.html")
