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
WAVELET_LEVEL = 4
R_FACTOR = 2
R_VALUES = [2, 4]
DEFAULT_R = 2
CENTER_LINES = 32
RNG_SEED = 42


def load_brain_slice(path):
    """Load the 2D brain slice from a NIfTI file."""
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")
    img = nib.load(str(path))
    data = img.get_fdata()[:, :, 0].astype(np.float32)
    return data


def compute_kspace_magnitude(brain):
    """Compute log10(|FFTshift(FFT2(brain))| + 1) for display."""
    kspace = np.fft.fft2(brain)
    kspace_shifted = np.fft.fftshift(kspace)
    return np.log10(np.abs(kspace_shifted) + 1)


SAMPLING_MODES = ["random", "uniform"]
DEFAULT_SAMPLING = "random"


def make_sampling_mask(n_rows, mode="random", R=R_FACTOR, center=CENTER_LINES,
                       seed=RNG_SEED):
    """Return a boolean mask of which phase-encode lines (rows) are sampled.

    mode="random": fully sample the central `center` lines, then randomly
                   sample from the rest to reach n_rows / R total.
    mode="uniform": sample every R-th line (no special centre treatment).
    """
    mask = np.zeros(n_rows, dtype=bool)
    if mode == "uniform":
        mask[::R] = True
    else:
        # Fully sample the centre
        c0 = n_rows // 2 - center // 2
        mask[c0:c0 + center] = True
        # Randomly sample the rest
        outer_indices = np.where(~mask)[0]
        n_total = n_rows // R
        n_extra = max(n_total - center, 0)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(outer_indices, size=min(n_extra, len(outer_indices)),
                            replace=False)
        mask[chosen] = True
    return mask


def make_sampling_overlay(n_rows, n_cols, mask):
    """Build a 2D array for the red sampling overlay.

    Sampled rows → 1.0, unsampled rows → NaN (transparent).
    """
    overlay = np.full((n_rows, n_cols), np.nan)
    overlay[mask] = 1.0
    return overlay


def undersampled_kspace(brain, mask):
    """Return the zero-filled undersampled k-space (shifted, complex)."""
    kspace = np.fft.fft2(brain)
    kspace_shifted = np.fft.fftshift(kspace)
    undersampled = np.zeros_like(kspace_shifted)
    undersampled[mask] = kspace_shifted[mask]
    return undersampled


def zero_filled_recon(brain, mask):
    """Reconstruct image from undersampled k-space via zero-filling.

    The mask is applied to rows of the fftshifted k-space (so the center
    lines correspond to low spatial frequencies).
    """
    us = undersampled_kspace(brain, mask)
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


def data_consistency_kspace(brain, mask, thresh_recon):
    """Combine original acquired k-space lines with thresholded recon's k-space.

    Acquired lines (mask=True): use original k-space data.
    Unacquired lines (mask=False): use k-space from the thresholded reconstruction.
    Returns the combined shifted k-space (complex).
    """
    orig_kspace = np.fft.fftshift(np.fft.fft2(brain))
    thresh_kspace = np.fft.fftshift(np.fft.fft2(thresh_recon))
    combined = thresh_kspace.copy()
    combined[mask] = orig_kspace[mask]
    return combined


def make_unacquired_overlay(n_rows, n_cols, mask):
    """Build a 2D array highlighting UNacquired rows (green overlay).

    Unsampled rows → 1.0, sampled rows → NaN (transparent).
    """
    overlay = np.full((n_rows, n_cols), np.nan)
    overlay[~mask] = 1.0
    return overlay


def _build_fig(brain, kspace_mag, wavelet_map, sampling_overlay,
               us_kspace_mag, recon, recon_wavelet_map,
               thresholded_wavelet_map, thresh_kspace_mag,
               unacquired_overlay, combined_kspace_mag,
               combined_recon):
    """Build a 4×3 Plotly figure.

    Row 1: (a) k-space + red overlay,       (b) brain image,       (c) wavelet transform
    Row 2: (d) zero-filled k-space,          (e) zero-filled recon, (f) recon wavelet
    Row 3: (g) thresh k-space + green gaps,  (h) empty,             (i) soft-thresholded wavelet
    Row 4: (j) combined k-space + overlays,  (k) iteration 1 recon, (l) empty
    """
    TITLE_SIZE = 24

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[
            "(a) k-space", "(b) MRI brain image", "(c) wavelet transform",
            "(d) zero-filled k-space", "(e) zero-filled reconstruction", "(f) wavelet transform",
            "(g) k-space (from thresholded)", "", "(i) soft-thresholded wavelet",
            "(j) data consistency k-space", "(k) iteration 1 reconstruction", "",
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.05,
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
        z=combined_recon[::-1],
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

    fig.update_layout(
        height=1700,
        autosize=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    for row in [1, 2, 3, 4]:
        for col in [1, 2, 3]:
            suffix = "" if (row == 1 and col == 1) else str((row - 1) * 3 + col)
            xref = f"x{suffix}"
            fig.update_yaxes(scaleanchor=xref, scaleratio=1, row=row, col=col)
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

    return fig


# Trace indices that change with sampling mode (row 1 ground-truth is fixed)
# 0=kspace, 1=red overlay, 2=brain, 3=wavelet are FIXED (row 1 ground truth)
# Everything from row 2 onward changes with sampling mode
VARIABLE_TRACES = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def _round_z(z, decimals=1):
    """Round a z array for compact JSON."""
    if hasattr(z, "tolist"):
        return np.round(z, decimals).tolist()
    arr = np.array(z)
    return np.round(arr, decimals).tolist()


def _extract_variable_z(fig, decimals=1):
    """Extract only the variable traces' z data (smaller JSON)."""
    all_z = []
    for trace in fig.data:
        if hasattr(trace, "z") and trace.z is not None:
            all_z.append(_round_z(trace.z, decimals))
    return {str(i): all_z[i] for i in VARIABLE_TRACES if i < len(all_z)}


def precompute(orientations=ORIENTATIONS):
    """Pre-compute figures for all orientations × sampling modes × R values."""
    var_data = {}
    refs = {}
    for orient_name, path in orientations.items():
        brain = load_brain_slice(path)
        kspace_mag = compute_kspace_magnitude(brain)
        wavelet_map = compute_wavelet_map(brain)
        for samp_mode in SAMPLING_MODES:
            for R in R_VALUES:
                key = f"{orient_name}_{samp_mode}_{R}"
                mask = make_sampling_mask(brain.shape[0], mode=samp_mode, R=R)
                overlay = make_sampling_overlay(brain.shape[0], brain.shape[1], mask)
                us_kspace = undersampled_kspace(brain, mask)
                us_kspace_mag = np.log10(np.abs(us_kspace) + 1)
                recon = zero_filled_recon(brain, mask)
                recon_wavelet_map = compute_wavelet_map(recon)
                thresh_wavelet_map, thresh_recon = soft_threshold_wavelet(recon)
                thresh_kspace_mag = compute_kspace_magnitude(thresh_recon)
                unacquired_overlay = make_unacquired_overlay(
                    brain.shape[0], brain.shape[1], mask)
                combined_ks = data_consistency_kspace(brain, mask, thresh_recon)
                combined_kspace_mag = np.log10(np.abs(combined_ks) + 1)
                combined_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(combined_ks)))
                fig = _build_fig(brain, kspace_mag, wavelet_map, overlay,
                                 us_kspace_mag, recon, recon_wavelet_map,
                                 thresh_wavelet_map, thresh_kspace_mag,
                                 unacquired_overlay, combined_kspace_mag,
                                 combined_recon)
                is_default = (orient_name == DEFAULT_ORIENTATION
                              and samp_mode == DEFAULT_SAMPLING
                              and R == DEFAULT_R)
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
                var_data[key] = _extract_variable_z(fig)
    return refs, var_data


def make_embeddable_html(orientations=ORIENTATIONS):
    """Return an HTML fragment for embedding in a MyST page via IPython.display.HTML."""
    refs, var_data = precompute(orientations)
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
    <label>Orientation</label>
    <select id="cs2d-orientSelect">
      {"".join(f'<option value="{n}"{" selected" if n == DEFAULT_ORIENTATION else ""}>{n.capitalize()}</option>' for n in orientation_names)}
    </select>
  </div>
  <div class="cs2d-fig-ctrl-group">
    <label>Sampling</label>
    <select id="cs2d-samplingSelect">
      {"".join(f'<option value="{m}"{" selected" if m == DEFAULT_SAMPLING else ""}>{m.capitalize()}</option>' for m in SAMPLING_MODES)}
    </select>
  </div>
  <div class="cs2d-fig-ctrl-group">
    <label>R (acceleration)</label>
    <select id="cs2d-rSelect">
      {"".join(f'<option value="{r}"{" selected" if r == DEFAULT_R else ""}>{r}x</option>' for r in R_VALUES)}
    </select>
  </div>
</div>

<div id="cs2d-fig"></div>

<script>
(function() {{
  function initFigure() {{
    const REF      = {json.dumps(refs["default"])};
    const VAR_DATA = {json.dumps(var_data)};

    Plotly.newPlot("cs2d-fig", REF.data, REF.layout, {{responsive: true}});

    var figEl = document.getElementById("cs2d-fig");
    if (window.ResizeObserver) {{
      new ResizeObserver(function() {{ Plotly.Plots.resize(figEl); }}).observe(figEl);
    }}
    window.addEventListener("resize", function() {{ Plotly.Plots.resize(figEl); }});

    function updateFig() {{
      var orient = document.getElementById("cs2d-orientSelect").value;
      var samp   = document.getElementById("cs2d-samplingSelect").value;
      var rval   = document.getElementById("cs2d-rSelect").value;
      var key    = orient + "_" + samp + "_" + rval;
      var d = VAR_DATA[key];
      if (d) {{
        var indices = Object.keys(d);
        for (var k = 0; k < indices.length; k++) {{
          Plotly.restyle("cs2d-fig", {{ z: [d[indices[k]]] }}, [parseInt(indices[k])]);
        }}
      }}
    }}
    document.getElementById("cs2d-orientSelect").addEventListener("change", updateFig);
    document.getElementById("cs2d-samplingSelect").addEventListener("change", updateFig);
    document.getElementById("cs2d-rSelect").addEventListener("change", updateFig);
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
