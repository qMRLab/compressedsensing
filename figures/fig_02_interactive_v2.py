"""
Figure 2 — Interactive CS reconstruction demo (v2).

Layout (5 rows × 2 columns) matching fig_02_cs_demo.py:
  (a) True signal             (b) True k-space + acquired points
  (c) Recon iter 1 + thr      (d) K-space of detected components
  (e) [annotation]            (f) Residual k-space
  (g) Recon iter 2 + thr      (h) [annotation]
  (i) Full reconstruction     (j) [empty]
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
R_VALUES     = [2, 3, 4]
SIGMA_VALUES = [2.0, 3.0, 4.0]
SIGMA2       = 3.0
N            = 128
SEED         = 42
R_DEFAULT    = 2
SIGMA_DEFAULT= 3.0

C_THR  = "navy"


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
    x_rec1[peaks1] = np.abs(x_samp[peaks1]) * R
    k_rec1 = np.where(mask, np.fft.fft(x_rec1), 0)
    k_res1 = k_samp - k_rec1
    x_res1 = np.fft.ifft(k_res1).real

    thr2 = float(x_res1.mean() + sigma2 * x_res1.std())
    peaks2, _ = find_peaks(
        np.abs(np.where(np.abs(x_res1) >= thr2, x_res1, 0.0)), height=thr2 * 0.9
    )
    x_rec2 = np.zeros(N)
    x_rec2[peaks2] = np.abs(x_res1[peaks2]) * R
    x_combined = x_rec1.copy()
    for p in peaks2:
        x_combined[p] += x_rec2[p]

    return dict(
        x_samp=x_samp, thr1=thr1,
        peaks1=sorted(peaks1.tolist()),
        k_det=np.fft.fft(x_rec1),
        k_res1=k_res1,
        x_res1=x_res1, thr2=thr2,
        peaks2=sorted(peaks2.tolist()),
        x_combined=x_combined,
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

def _build_fig(signal, kspace_full, mask_uni, mask_rand, idx_uni, idx_rand,
               R, sigma1, positions, mode="rand"):
    N = len(signal)
    t = list(range(N))
    k_axis = [i - N // 2 for i in range(N)]
    k_mag_full = np.abs(np.fft.fftshift(kspace_full)).tolist()

    if mode == "uni":
        res = _run(signal, kspace_full, mask_uni, R, sigma1)
        idx = idx_uni
        label = "Uniform"
    else:  # "rand"
        res = _run(signal, kspace_full, mask_rand, R, sigma1)
        idx = idx_rand
        label = "Random"

    n_kept = N // R

    # ── Normalize (c) and (g) by ×R so all image-space panels share the
    #    same amplitude scale as (a) true signal and (i) final recon.
    #    Plot magnitude (abs) in image space — no negatives. ──────────────
    x_samp_norm = np.abs(res["x_samp"] * R)
    thr1_norm   = abs(res["thr1"] * R)
    x_res1_norm = np.abs(res["x_res1"] * R)
    thr2_norm   = abs(res["thr2"] * R)
    x_combined_mag = np.abs(res["x_combined"])

    # Common y-range for all left-column (image-space) panels
    y_max = float(max(signal.max(),
                      x_samp_norm.max(),
                      x_res1_norm.max(),
                      x_combined_mag.max()))
    y_pad = y_max * 0.15
    img_yrange = [-y_pad * 0.2, y_max + y_pad]

    # ── font sizes (publication quality) ─────────────────────────────────
    TITLE_SIZE  = 24
    AXIS_SIZE   = 20
    TICK_SIZE   = 16
    ANNOT_SIZE  = 26
    LEGEND_SIZE = 18

    # ── 5×2 subplot grid ─────────────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            "(a) True signal", "(b) True k-space",
            f"(c) Recon iter 1 ({sigma1}σ thr)", f"(d) K-space of detected (peaks: {res['peaks1']})",
            "", "(f) Residual k-space",
            f"(g) Recon iter 2 ({SIGMA2}σ thr)", "",
            "(i) Full reconstruction", "",
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.14,
    )

    # Double subplot title font sizes
    for ann in fig.layout.annotations:
        ann.font.size = TITLE_SIZE

    # ── Row 1, Col 1: True signal (stem) ─────────────────────────────────
    sx, sy = _stem_xy(signal)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
        line=dict(color="steelblue", width=4), hoverinfo="skip",
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=signal.tolist(), mode="markers",
        marker=dict(color="steelblue", size=12), showlegend=False,
        hovertemplate="t=%{x}<br>amp=%{y:.2f}<extra>true</extra>"),
        row=1, col=1)

    # ── Acquired-sample dot y-level: 10% above max k-space magnitude ────
    k_ymax = max(k_mag_full)
    acq_dot_y = k_ymax * 1.10
    acq_k = [int(i) - N // 2 for i in idx]

    # ── Row 1, Col 2: True k-space + sampled trace (green) + dots ───────
    fig.add_trace(go.Scatter(x=k_axis, y=k_mag_full, mode="lines",
        line=dict(color="steelblue", width=3.2), showlegend=False,
        hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=acq_k, y=[acq_dot_y] * len(acq_k),
        mode="markers", marker=dict(color="red", size=8),
        showlegend=False,
        hovertemplate="k=%{x}<br>acquired<extra></extra>"),
        row=1, col=2)

    # ── Row 2, Col 1: Recon iter 1 (magnitude, normalized ×R) + threshold
    sx, sy = _stem_xy(x_samp_norm)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
        line=dict(color="green", width=3.2), hoverinfo="skip",
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_samp_norm.tolist(), mode="markers",
        marker=dict(color="green", size=10), showlegend=False,
        hovertemplate="t=%{x}<br>%{y:.3f}<extra>recon iter1</extra>"),
        row=2, col=1)
    # threshold line (positive only — magnitude plot)
    fig.add_trace(go.Scatter(
        x=[0, N-1], y=[thr1_norm, thr1_norm],
        mode="lines", line=dict(color=C_THR, width=6, dash="dot"),
        hoverinfo="skip", showlegend=False), row=2, col=1)

    # ── Row 2, Col 2: K-space of detected components ─────────────────────
    k_det_mag = np.abs(np.fft.fftshift(res["k_det"])).tolist()
    fig.add_trace(go.Scatter(x=k_axis, y=k_det_mag, mode="lines",
        line=dict(color="purple", width=3.2), showlegend=False,
        hovertemplate="k=%{x}<br>|X|=%{y:.3f}<extra>k detected</extra>"),
        row=2, col=2)
    fig.add_trace(go.Scatter(x=acq_k, y=[acq_dot_y] * len(acq_k),
        mode="markers", marker=dict(color="red", size=8),
        showlegend=False, hoverinfo="skip"),
        row=2, col=2)

    # ── Row 3, Col 1: Annotation (empty plot with text) ──────────────────
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text",
        text=["← subtract (d) from (b) →"],
        textfont=dict(size=ANNOT_SIZE, color="gray"),
        showlegend=False, hoverinfo="skip"), row=3, col=1)

    # ── Row 3, Col 2: Residual k-space ───────────────────────────────────
    k_res_mag = np.abs(np.fft.fftshift(res["k_res1"])).tolist()
    fig.add_trace(go.Scatter(x=k_axis, y=k_res_mag, mode="lines",
        line=dict(color="red", width=3.2), showlegend=False,
        hovertemplate="k=%{x}<br>|X|=%{y:.3f}<extra>residual k</extra>"),
        row=3, col=2)
    fig.add_trace(go.Scatter(x=acq_k, y=[acq_dot_y] * len(acq_k),
        mode="markers", marker=dict(color="red", size=8),
        showlegend=False, hoverinfo="skip"),
        row=3, col=2)

    # ── Row 4, Col 1: Recon iter 2 (magnitude, normalized ×R) + threshold
    sx, sy = _stem_xy(x_res1_norm)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
        line=dict(color="red", width=3.2), hoverinfo="skip",
        showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_res1_norm.tolist(), mode="markers",
        marker=dict(color="red", size=10), showlegend=False,
        hovertemplate="t=%{x}<br>%{y:.3f}<extra>residual</extra>"),
        row=4, col=1)
    # threshold line (positive only — magnitude plot)
    fig.add_trace(go.Scatter(
        x=[0, N-1], y=[thr2_norm, thr2_norm],
        mode="lines", line=dict(color=C_THR, width=6, dash="dot"),
        hoverinfo="skip", showlegend=False), row=4, col=1)

    # ── Row 4, Col 2: Annotation ─────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text",
        text=[f"Iter 2 recovered peaks:<br>{res['peaks2']}"],
        textfont=dict(size=ANNOT_SIZE, color="gray"),
        showlegend=False, hoverinfo="skip"), row=4, col=2)

    # ── Row 5, Col 1: Full reconstruction (magnitude) ──────────────────
    sx, sy = _stem_xy(x_combined_mag)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
        line=dict(color="purple", width=3.2), hoverinfo="skip",
        showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_combined_mag.tolist(), mode="markers",
        marker=dict(color="purple", size=10), showlegend=False,
        hovertemplate="t=%{x}<br>%{y:.3f}<extra>final recon</extra>"),
        row=5, col=1)

    # ── Row 5, Col 2: Empty ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
        showlegend=False, hoverinfo="skip"), row=5, col=2)

    # ── Layout polish ────────────────────────────────────────────────────
    fig.update_layout(
        height=1600,
        autosize=True,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(x=0.75, y=0.95, font=dict(size=LEGEND_SIZE)),
        margin=dict(l=80, r=30, t=60, b=50),
    )

    # ── Axis styling: borders, grid, font sizes ──────────────────────────
    border_kw = dict(showline=True, linewidth=2, linecolor="black", mirror=True)
    for row in range(1, 6):
        for col in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor="#ddd",
                             tickfont=dict(size=TICK_SIZE), **border_kw,
                             row=row, col=col)
            fig.update_yaxes(showgrid=True, gridcolor="#ddd",
                             tickfont=dict(size=TICK_SIZE), **border_kw,
                             row=row, col=col)

    # Left column: signal domain — identical y-range on all panels
    for row in [1, 2, 4, 5]:
        fig.update_xaxes(title_text="Sample index",
                         title_font=dict(size=AXIS_SIZE), row=row, col=1)
        fig.update_yaxes(title_text="Amplitude",
                         title_font=dict(size=AXIS_SIZE),
                         range=img_yrange, row=row, col=1)
        fig.update_xaxes(range=[-5, N + 5], row=row, col=1)

    # Right column: k-space — identical x-range and y-range on all panels
    # acq_dot_y is at k_ymax*1.10; leave room above dots
    k_xrange = [-N // 2 - 2, N // 2 + 2]
    k_yrange = [-k_ymax * 0.05, k_ymax * 1.25]
    for row in [1, 2, 3]:
        fig.update_xaxes(title_text="k",
                         title_font=dict(size=AXIS_SIZE),
                         range=k_xrange, row=row, col=2)
        fig.update_yaxes(title_text="|X(k)|",
                         title_font=dict(size=AXIS_SIZE),
                         range=k_yrange, row=row, col=2)

    # Annotation panels — hide axes and fix range for centered text
    for (r, c) in [(3, 1), (4, 2), (5, 2)]:
        fig.update_xaxes(visible=False, range=[0, 1], row=r, col=c)
        fig.update_yaxes(visible=False, range=[0, 1], row=r, col=c)

    return fig


# ── extract trace x/y arrays in order ────────────────────────────────────────

def _extract_xy(fig):
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

    R0, s0 = R_DEFAULT, SIGMA_DEFAULT
    mu0, mr0, iu0, ir0 = masks[R0]
    refs = {}
    for mode in ("uni", "rand"):
        rf = _build_fig(signal, kspace_full, mu0, mr0, iu0, ir0,
                        R0, s0, positions, mode=mode)
        refs[mode] = json.loads(rf.to_json())

    combos = {mode: {} for mode in ("uni", "rand")}
    for R in r_values:
        mu, mr, iu, ir = masks[R]
        for mode in ("uni", "rand"):
            combos[mode][R] = {}
            for sigma1 in sigma_values:
                f = _build_fig(signal, kspace_full, mu, mr, iu, ir,
                               R, sigma1, positions, mode=mode)
                key = f"{sigma1:.1f}"
                xy  = _extract_xy(f)
                xy.append({"title": ""})
                combos[mode][R][key] = xy

    return refs, combos


# ── render embeddable HTML ────────────────────────────────────────────────────

def _render_embeddable_html(refs, combos, r_values, sigma_values):
    combos_json = {
        mode: {str(R): inner for R, inner in by_r.items()}
        for mode, by_r in combos.items()
    }
    r_default_idx = r_values.index(R_DEFAULT) if R_DEFAULT in r_values else 0
    s_default_idx = sigma_values.index(SIGMA_DEFAULT) if SIGMA_DEFAULT in sigma_values else 0
    r_show = r_values[r_default_idx]

    return f"""<style>
  .cs-fig-controls {{ display: flex; gap: 40px; justify-content: center;
               align-items: center; flex-wrap: wrap; margin: 8px 0 4px; }}
  .cs-fig-ctrl-group {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  .cs-fig-controls label {{ font-size: 14px; color: #444; font-weight: bold; }}
  .cs-fig-controls select {{ font-size: 14px; padding: 4px 10px; border-radius: 6px;
            border: 1px solid #ccc; cursor: pointer; background: white; }}
  #cs-fig  {{ width: 100%; margin: 0 auto; }}
</style>

<div class="cs-fig-controls">
  <div class="cs-fig-ctrl-group">
    <label>Sampling</label>
    <select id="cs-modeSelect">
      <option value="rand" selected>Random</option>
      <option value="uni">Uniform</option>
    </select>
  </div>
  <div class="cs-fig-ctrl-group">
    <label>Acceleration R</label>
    <select id="cs-rSelect">
      {"".join(f'<option value="{R}"{" selected" if R == R_DEFAULT else ""}>{R}×</option>' for R in r_values)}
    </select>
  </div>
  <div class="cs-fig-ctrl-group">
    <label>Threshold &sigma;</label>
    <select id="cs-sSelect">
      {"".join(f'<option value="{s}"{" selected" if s == SIGMA_DEFAULT else ""}>{s:.0f}&sigma;</option>' for s in sigma_values)}
    </select>
  </div>
</div>

<div id="cs-fig"></div>

<script>
(function() {{
  function initFigure() {{
    const REFS     = {json.dumps(refs)};
    const COMBOS   = {json.dumps(combos_json)};
    const R_VALUES = {json.dumps(r_values)};
    const S_VALUES = {json.dumps(sigma_values)};

    let currentR     = R_VALUES[{r_default_idx}];
    let currentSigma = S_VALUES[{s_default_idx}];
    let currentMode  = "rand";

    Plotly.newPlot("cs-fig", REFS[currentMode].data, REFS[currentMode].layout,
                   {{responsive: true}});

    // Auto-resize on window/container resize
    var figEl = document.getElementById("cs-fig");
    if (window.ResizeObserver) {{
      new ResizeObserver(function() {{ Plotly.Plots.resize(figEl); }}).observe(figEl);
    }}
    window.addEventListener("resize", function() {{ Plotly.Plots.resize(figEl); }});

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

      Plotly.react("cs-fig", newTraces, ref.layout);
      Plotly.relayout("cs-fig", {{ "title.text": titleEntry.title }});
    }}

    document.getElementById("cs-modeSelect").addEventListener("change", function() {{
      currentMode = this.value;
      const ref = REFS[currentMode];
      Plotly.newPlot("cs-fig", ref.data, ref.layout, {{responsive: true}});
      update();
    }});

    document.getElementById("cs-rSelect").addEventListener("change", function() {{
      currentR = +this.value;
      update();
    }});

    document.getElementById("cs-sSelect").addEventListener("change", function() {{
      currentSigma = +this.value;
      update();
    }});
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


# ── public API ─────────────────────────────────────────────────────────────────

def make_embeddable_html(
    positions=DEFAULT_POSITIONS,
    heights=DEFAULT_HEIGHTS,
    r_values=R_VALUES,
    sigma_values=SIGMA_VALUES,
):
    """Return an HTML fragment for embedding in a MyST page via IPython.display.HTML."""
    refs, combos = precompute(positions, heights, r_values, sigma_values)
    return _render_embeddable_html(refs, combos, r_values, sigma_values)


if __name__ == "__main__":
    refs, combos = precompute()
    html = _render_embeddable_html(refs, combos, R_VALUES, SIGMA_VALUES)
    Path(".tmp/fig_02_interactive_v2.html").write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{html}</body></html>"
    )
    print("saved → .tmp/fig_02_interactive_v2.html")
