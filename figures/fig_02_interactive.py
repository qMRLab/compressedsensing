"""
Figure 2 — Interactive CS reconstruction demo.

Strategy
--------
Python builds the complete Plotly figure for the default parameters and
exports it as JSON (so axis assignments, domains, and layout are 100% correct).
It also pre-computes every (R, σ) combination and exports just the x/y arrays
in trace order.  The browser uses Plotly.js + two sliders; on each change it
calls Plotly.restyle() to swap the x/y data of existing traces — no axis
remapping needed, no server required.
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from figures.fig_02_sparse_signal_reconstruction import make_sparse_signal
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from figures.fig_02_sparse_signal_reconstruction import make_sparse_signal


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_POSITIONS = [10, 50, 100]
DEFAULT_HEIGHTS   = [1.0, 0.85, 0.40]
R_VALUES     = [2, 3, 4, 6, 8]
SIGMA_VALUES = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
SIGMA2       = 3.0
N            = 128
SEED         = 42
R_DEFAULT    = 4
SIGMA_DEFAULT= 5.0

C_UNI  = "#f57c00"
C_RAND = "#2e7d32"
C_TRUE = "#9e9e9e"
C_THR  = "#e53935"


# ── CS pipeline ───────────────────────────────────────────────────────────────

def _run(signal, kspace_full, mask, R, sigma1, sigma2=SIGMA2):
    N = len(signal)
    k_samp = np.where(mask, kspace_full, 0)
    x_samp = np.fft.ifft(k_samp).real

    thr1 = float(x_samp.mean() + sigma1 * x_samp.std())
    peaks1, _ = find_peaks(
        np.abs(np.where(np.abs(x_samp) >= thr1, x_samp, 0.0)), height=thr1 * 0.9
    )
    x_rec1 = np.zeros(N)
    x_rec1[peaks1] = x_samp[peaks1] * R
    k_res1 = k_samp - np.where(mask, np.fft.fft(x_rec1), 0)
    x_res1 = np.fft.ifft(k_res1).real

    thr2 = float(x_res1.mean() + sigma2 * x_res1.std())
    peaks2, _ = find_peaks(
        np.abs(np.where(np.abs(x_res1) >= thr2, x_res1, 0.0)), height=thr2 * 0.9
    )
    x_rec2 = np.zeros(N)
    x_rec2[peaks2] = x_res1[peaks2] * R
    x_combined = x_rec1.copy()
    for p in peaks2:
        x_combined[p] += x_rec2[p]

    all_peaks = sorted(set(peaks1.tolist()) | set(peaks2.tolist()))
    pct_err = [
        float((x_combined[p] - signal[p]) / signal[p] * 100) if signal[p] != 0 else 0.0
        for p in all_peaks
    ]
    return dict(
        x_samp=x_samp, thr1=thr1,
        k_det_mag=np.abs(np.fft.fftshift(np.fft.fft(x_rec1))),
        x_res1=x_res1, thr2=thr2,
        x_combined=x_combined,
        all_peaks=all_peaks, pct_err=pct_err,
    )


# ── stem helper ───────────────────────────────────────────────────────────────

def _stem_xy(y):
    """Return (x_list, y_list) for a stem plot (lines from 0 to y[i])."""
    xs, ys = [], []
    for i, v in enumerate(y):
        xs += [i, i, None]
        ys += [0, float(v), None]
    return xs, ys


# ── build Plotly figure for one (R, sigma1) combo ────────────────────────────
# mode: "uni" | "rand" | "both"

def _build_fig(signal, kspace_full, mask_uni, mask_rand, idx_uni, idx_rand,
               R, sigma1, positions, mode="rand"):
    N = len(signal)
    t = list(range(N))
    k_axis = [i - N // 2 for i in range(N)]
    k_mag_full = np.abs(np.fft.fftshift(kspace_full))

    ru = _run(signal, kspace_full, mask_uni,  R, sigma1)
    rr = _run(signal, kspace_full, mask_rand, R, sigma1)

    if mode == "uni":
        col_title   = "<b>Uniform</b> undersampling"
        panel_pairs = [(ru, C_UNI, idx_uni, "Uniform")]
    elif mode == "rand":
        col_title   = "<b>Random</b> undersampling"
        panel_pairs = [(rr, C_RAND, idx_rand, "Random")]
    else:  # "both"
        col_title   = "<b>Uniform</b> (orange) vs <b>Random</b> (green)"
        panel_pairs = [(ru, C_UNI, idx_uni, "Uniform"),
                       (rr, C_RAND, idx_rand, "Random")]

    fig = make_subplots(
        rows=5, cols=1,
        column_titles=[col_title],
        row_titles=[
            "True signal  |  k-space",
            "Reconstruction + threshold (iter 1)",
            "K-space of detected",
            "Residual + threshold (iter 2)",
            "Final reconstruction",
        ],
        vertical_spacing=0.065,
    )

    def add_stems(y, row, color, hover_name):
        sx, sy = _stem_xy(y)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
            line=dict(color=color, width=0.8),
            hoverinfo="skip", showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=list(y), mode="markers",
            marker=dict(color=color, size=3), showlegend=False,
            hovertemplate=f"t=%{{x}}<br>%{{y:.3f}}<extra>{hover_name}</extra>"),
            row=row, col=1)

    def add_thr(thr, row):
        for sign in [1, -1]:
            fig.add_trace(go.Scatter(
                x=[0, N], y=[sign * thr, sign * thr], mode="lines",
                line=dict(color=C_THR, width=1.5, dash="dot"),
                hoverinfo="skip", showlegend=False), row=row, col=1)

    # ── Row 1: shared traces (true signal + k-space background) ──────────────
    sx, sy = _stem_xy(signal)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
        line=dict(color=C_TRUE, width=0.8),
        hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=signal.tolist(), mode="markers",
        marker=dict(color=C_TRUE, size=4), showlegend=False,
        hovertemplate="t=%{x}<br>amp=%{y:.2f}<extra>true signal</extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=k_axis, y=k_mag_full.tolist(), mode="lines",
        line=dict(color="#cccccc", width=0.5),
        hoverinfo="skip", showlegend=False), row=1, col=1)

    # ── Row 1: per-panel acquired k-space points ──────────────────────────────
    for res, c, idx, label in panel_pairs:
        acq_k   = [int(i) - N // 2 for i in idx]
        acq_mag = [float(k_mag_full[i]) for i in idx]
        fig.add_trace(go.Scatter(x=acq_k, y=acq_mag, mode="markers",
            marker=dict(color=c, size=4),
            name=label, legendgroup=label,
            showlegend=(mode == "both"),
            hovertemplate=f"k=%{{x}}<br>|X|=%{{y:.2f}}<extra>{label} acquired</extra>"),
            row=1, col=1)

    # ── Rows 2–5: per-panel traces ────────────────────────────────────────────
    for res, c, idx, label in panel_pairs:
        # Row 2: reconstruction + threshold
        add_stems(res["x_samp"], 2, c, f"recon iter1 {label}")
        add_thr(res["thr1"], 2)

        # Row 3: k-space of detected
        fig.add_trace(go.Scatter(
            x=k_axis, y=res["k_det_mag"].tolist(), mode="lines",
            line=dict(color=c, width=0.8), showlegend=False,
            hovertemplate=f"k=%{{x}}<br>|X|=%{{y:.3f}}<extra>k detected {label}</extra>"),
            row=3, col=1)

        # Row 4: residual + threshold
        add_stems(res["x_res1"], 4, c, f"residual {label}")
        add_thr(res["thr2"], 4)

        # Row 5: final reconstruction
        add_stems(res["x_combined"], 5, c, f"final recon {label}")

    # ── Row 5: shared ground-truth overlay ───────────────────────────────────
    true_x = [i for i, v in enumerate(signal) if v != 0]
    true_y = [float(v) for v in signal if v != 0]
    fig.add_trace(go.Scatter(x=true_x, y=true_y, mode="markers",
        marker=dict(color=C_TRUE, size=10, symbol="circle-open",
                    line=dict(width=2, color=C_TRUE)),
        showlegend=False,
        hovertemplate="t=%{x}<br>true=%{y:.2f}<extra>ground truth</extra>"),
        row=5, col=1)

    # ── layout polish ─────────────────────────────────────────────────────────
    n_kept = len(idx_uni) if mode in ("uni", "both") else len(idx_rand)
    fig.update_layout(
        title=dict(
            text=(f"<b>Compressed Sensing MRI — Figure 2</b>   "
                  f"R={R}  ({n_kept}/{N} k-samples)   σ={sigma1}"),
            x=0.5,
        ),
        height=1150, width=620,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02,
                    yanchor="bottom") if mode == "both" else dict(visible=False),
    )
    for row in range(1, 6):
        fig.update_xaxes(showgrid=True, gridcolor="#eee", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#eee", row=row, col=1)
        fig.update_yaxes(title_text="Amplitude", row=row, col=1)

    fig.update_yaxes(title_text="|K-space|", row=1, col=1)
    fig.update_yaxes(title_text="|K-space|", row=3, col=1)

    for p in positions:
        for row in [2, 4, 5]:
            fig.add_vline(x=p, line_width=0.8, line_dash="dash",
                          line_color="lightgray", row=row, col=1)
    return fig


# ── extract trace x/y arrays in order ────────────────────────────────────────

def _extract_xy(fig):
    """Return list of {x, y} dicts in trace order, ready for Plotly.restyle."""
    result = []
    for trace in fig.data:
        x = trace.x
        y = trace.y
        result.append({
            "x": list(x) if x is not None else [],
            "y": list(y) if y is not None else [],
        })
    return result


# ── pre-compute all combos ────────────────────────────────────────────────────

def precompute(
    positions=DEFAULT_POSITIONS,
    heights=DEFAULT_HEIGHTS,
    r_values=R_VALUES,
    sigma_values=SIGMA_VALUES,
    n=N, seed=SEED,
):
    rng = np.random.default_rng(seed)
    signal = make_sparse_signal(n, positions, heights)
    kspace_full = np.fft.fft(signal)

    masks = {}
    for R in r_values:
        n_kept   = n // R
        idx_uni  = list(range(0, n, R))
        idx_rand = sorted(rng.choice(n, n_kept, replace=False).tolist())
        mask_uni  = np.zeros(n, dtype=bool); mask_uni[idx_uni]  = True
        mask_rand = np.zeros(n, dtype=bool); mask_rand[idx_rand]= True
        masks[R]  = (mask_uni, mask_rand, idx_uni, idx_rand)

    # build one reference figure per mode to get correct layouts + axis refs
    R0, s0 = R_DEFAULT, SIGMA_DEFAULT
    mu0, mr0, iu0, ir0 = masks[R0]
    refs = {}
    for mode in ("uni", "rand", "both"):
        rf = _build_fig(signal, kspace_full, mu0, mr0, iu0, ir0,
                        R0, s0, positions, mode=mode)
        refs[mode] = json.loads(rf.to_json())

    # pre-compute x/y arrays for every (mode, R, sigma) combo
    # sigma key: f"{s:.1f}" so JS toFixed(1) matches
    combos = {mode: {} for mode in ("uni", "rand", "both")}
    for R in r_values:
        mu, mr, iu, ir = masks[R]
        for mode in ("uni", "rand", "both"):
            combos[mode][R] = {}
            for sigma1 in sigma_values:
                f = _build_fig(signal, kspace_full, mu, mr, iu, ir,
                               R, sigma1, positions, mode=mode)
                key = f"{sigma1:.1f}"
                xy  = _extract_xy(f)
                title = (f"<b>Compressed Sensing MRI — Figure 2</b>   "
                         f"R={R}  ({n//R}/{n} k-samples)   σ={sigma1}")
                xy.append({"title": title})
                combos[mode][R][key] = xy

    return refs, combos


# ── render HTML from pre-computed data ────────────────────────────────────────

def _render_html(refs, combos, r_values, sigma_values):
    """Render the complete interactive HTML string from pre-computed data."""
    combos_json = {
        mode: {str(R): inner for R, inner in by_r.items()}
        for mode, by_r in combos.items()
    }
    r_default_idx = r_values.index(R_DEFAULT) if R_DEFAULT in r_values else 0
    s_default_idx = sigma_values.index(SIGMA_DEFAULT) if SIGMA_DEFAULT in sigma_values else 0
    r_show = r_values[r_default_idx]

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Compressed Sensing — Figure 2</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
  h2   {{ text-align: center; color: #333; margin-bottom: 4px; }}
  .sub {{ text-align: center; color: #777; font-size: 13px; margin-bottom: 10px; }}
  .controls {{ display: flex; gap: 40px; justify-content: center;
               align-items: center; flex-wrap: wrap; margin: 8px 0 4px; }}
  .ctrl-group {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  label {{ font-size: 14px; color: #444; font-weight: bold; }}
  input[type=range] {{ width: 240px; cursor: pointer; accent-color: #555; }}
  select {{ font-size: 14px; padding: 4px 10px; border-radius: 6px;
            border: 1px solid #ccc; cursor: pointer; background: white; }}
  .val  {{ font-size: 14px; color: #111; min-width: 70px; text-align: center;
           background: #eee; border-radius: 4px; padding: 2px 8px; }}
  #fig  {{ width: 100%; }}
</style>
</head>
<body>
<h2>Compressed Sensing MRI — Figure 2</h2>
<p class="sub">Iterative thresholding recovery from undersampled k-space</p>

<div class="controls">
  <div class="ctrl-group">
    <label>Sampling</label>
    <select id="modeSelect">
      <option value="rand" selected>Random only</option>
      <option value="uni">Uniform only</option>
      <option value="both">Both overlaid</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label>Acceleration R</label>
    <input type="range" id="rSlider"
           min="0" max="{len(r_values)-1}" step="1" value="{r_default_idx}">
    <span class="val" id="rVal">R = {r_show}</span>
  </div>
  <div class="ctrl-group">
    <label>Threshold σ</label>
    <input type="range" id="sSlider"
           min="0" max="{len(sigma_values)-1}" step="1" value="{s_default_idx}">
    <span class="val" id="sVal">σ = {sigma_values[s_default_idx]}</span>
  </div>
</div>

<div id="fig"></div>

<script>
const REFS     = {json.dumps(refs)};
const COMBOS   = {json.dumps(combos_json)};
const R_VALUES = {json.dumps(r_values)};
const S_VALUES = {json.dumps(sigma_values)};

let currentR     = R_VALUES[{r_default_idx}];
let currentSigma = S_VALUES[{s_default_idx}];
let currentMode  = "rand";

// initial render
Plotly.newPlot("fig", REFS[currentMode].data, REFS[currentMode].layout,
               {{responsive: true}});

function update() {{
  const sigmaKey = currentSigma.toFixed(1);
  const entries  = COMBOS[currentMode][String(currentR)][sigmaKey];
  if (!entries) {{ console.error("No data for", currentMode, currentR, sigmaKey); return; }}

  const titleEntry = entries[entries.length - 1];
  const xyEntries  = entries.slice(0, -1);
  const ref        = REFS[currentMode];

  const newTraces = ref.data.map((base, i) => ({{
    ...base,
    x: xyEntries[i].x,
    y: xyEntries[i].y,
  }}));

  Plotly.react("fig", newTraces, ref.layout);
  Plotly.relayout("fig", {{ "title.text": titleEntry.title }});
}}

document.getElementById("modeSelect").addEventListener("change", function() {{
  currentMode = this.value;
  const ref = REFS[currentMode];
  Plotly.newPlot("fig", ref.data, ref.layout, {{responsive: true}});
  update();
}});

document.getElementById("rSlider").addEventListener("input", function() {{
  currentR = R_VALUES[+this.value];
  document.getElementById("rVal").textContent = "R = " + currentR;
  update();
}});

document.getElementById("sSlider").addEventListener("input", function() {{
  currentSigma = S_VALUES[+this.value];
  document.getElementById("sVal").textContent = "σ = " + currentSigma;
  update();
}});
</script>
</body>
</html>"""


# ── public API ─────────────────────────────────────────────────────────────────

def make_html_string(
    positions=DEFAULT_POSITIONS,
    heights=DEFAULT_HEIGHTS,
    r_values=R_VALUES,
    sigma_values=SIGMA_VALUES,
):
    """Return the complete interactive HTML as a string (no file I/O)."""
    refs, combos = precompute(positions, heights, r_values, sigma_values)
    return _render_html(refs, combos, r_values, sigma_values)


def write_interactive_html(
    path=".tmp/fig_02_interactive.html",
    positions=DEFAULT_POSITIONS,
    heights=DEFAULT_HEIGHTS,
    r_values=R_VALUES,
    sigma_values=SIGMA_VALUES,
):
    print("Pre-computing all combinations…")
    refs, combos = precompute(positions, heights, r_values, sigma_values)
    html = _render_html(refs, combos, r_values, sigma_values)
    Path(path).write_text(html)
    print(f"saved → {path}  ({Path(path).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    write_interactive_html()
