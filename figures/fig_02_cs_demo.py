"""
Figure 2 — Compressed sensing reconstruction demo.

Reproduces the iterative thresholding pipeline from:
  Lustig M, Donoho D, Pauly JM. Sparse MRI. MRM 58(6):1182-1195, 2007.

Layout (5 rows × 2 columns):
  (a) True signal          (b) True k-space + acquired points (red)
  (c) Recon iter 1 + thr   (d) K-space of detected components
  (e) [annotation]         (f) Residual k-space
  (g) Recon iter 2 + thr   (h) [annotation]
  (i) Full reconstruction  (j) % error per peak
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import find_peaks

try:
    from figures.fig_02_sparse_signal_reconstruction import make_sparse_signal
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from figures.fig_02_sparse_signal_reconstruction import make_sparse_signal


def make_figure(
    N: int = 512,
    R: int = 4,
    positions: list = None,
    heights: list = None,
    sigma_iter1: float = 5.0,
    sigma_iter2: float = 3.0,
    seed: int = 42,
) -> plt.Figure:
    """
    Parameters
    ----------
    N            : signal length
    R            : undersampling factor (n_kept = N // R)
    positions    : spike positions in the sparse signal
    heights      : spike amplitudes
    sigma_iter1  : threshold multiplier for iteration 1 (strong components)
    sigma_iter2  : threshold multiplier for iteration 2 (weak components)
    seed         : random seed for reproducible k-space sampling pattern
    """
    if positions is None:
        positions = [50, 200, 310, 380, 460]
    if heights is None:
        heights = [1.0, 0.85, 0.42, 0.38, 0.40]

    rng = np.random.default_rng(seed)
    n_kept = N // R

    signal = make_sparse_signal(N, positions, heights)
    kspace_full = np.fft.fft(signal)

    idx_rand = np.sort(rng.choice(N, n_kept, replace=False))
    mask_rand = np.zeros(N, dtype=bool)
    mask_rand[idx_rand] = True

    k_rand = np.where(mask_rand, kspace_full, 0)
    x_rand = np.fft.ifft(k_rand).real

    # --- Iteration 1: detect strong components ---
    thr1 = x_rand.mean() + sigma_iter1 * x_rand.std()
    peaks1, _ = find_peaks(np.abs(np.where(np.abs(x_rand) >= thr1, x_rand, 0.0)),
                           height=thr1 * 0.9)
    x_rec1 = np.zeros(N)
    x_rec1[peaks1] = x_rand[peaks1] * R          # rescale by R
    k_rec1 = np.where(mask_rand, np.fft.fft(x_rec1), 0)
    k_res1 = k_rand - k_rec1
    x_res1 = np.fft.ifft(k_res1).real

    # --- Iteration 2: detect weak components from residual ---
    thr2 = x_res1.mean() + sigma_iter2 * x_res1.std()
    peaks2, _ = find_peaks(np.abs(np.where(np.abs(x_res1) >= thr2, x_res1, 0.0)),
                           height=thr2 * 0.9)
    x_rec2 = np.zeros(N)
    x_rec2[peaks2] = x_res1[peaks2] * R

    x_combined = x_rec1.copy()
    for p in peaks2:
        x_combined[p] += x_rec2[p]

    all_peaks = sorted(peaks1.tolist() + peaks2.tolist())
    pct_err = {p: (x_combined[p] - signal[p]) / signal[p] * 100 for p in all_peaks}

    # --- Plot ---
    t = np.arange(N)
    k_axis = np.arange(N) - N // 2

    fig, axes = plt.subplots(5, 2, figsize=(13, 17))
    fig.suptitle(
        f'Compressed sensing reconstruction  (R={R}, N={N}, '
        f'{n_kept} k-space samples)',
        fontsize=13,
    )

    def sig_plot(ax, y, color, title, thr=None):
        ax.stem(t, y, markerfmt=color + 'o', linefmt=color + '-', basefmt='k-')
        for p in positions:
            ax.axvline(p, color='gray', lw=0.7, ls='--', alpha=0.3)
        if thr is not None:
            ax.axhline( thr, color='r', lw=1.2, ls=':', label=f'thr={thr:.3f}')
            ax.axhline(-thr, color='r', lw=1.2, ls=':')
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-5, N + 5)
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Sample index')

    def ksp_plot(ax, k, color, title, sampled_idx=None):
        mag = np.abs(np.fft.fftshift(k))
        ax.plot(k_axis, mag, color=color, lw=0.8)
        if sampled_idx is not None:
            s_shift = sampled_idx - N // 2
            ax.scatter(s_shift, mag[sampled_idx], color='red', s=18, zorder=5,
                       label=f'{len(sampled_idx)} acquired')
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('k')
        ax.set_ylabel('|K-space|')

    # Row 0
    sig_plot(axes[0, 0], signal, 'C0', '(a) True signal (image domain)')
    ksp_plot(axes[0, 1], kspace_full, 'C0',
             '(b) True k-space — red = randomly acquired points',
             sampled_idx=idx_rand)

    # Row 1
    sig_plot(axes[1, 0], x_rand, 'C2',
             f'(c) Recon from sampled k-space  (iter 1, {sigma_iter1}σ thr)', thr=thr1)
    ksp_plot(axes[1, 1], np.fft.fft(x_rec1), 'C3',
             f'(d) K-space of detected components  (peaks: {sorted(peaks1.tolist())})')

    # Row 2
    axes[2, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, '← subtract (d) from (b) →',
                    ha='center', va='center', fontsize=11, color='gray',
                    transform=axes[2, 0].transAxes)
    ksp_plot(axes[2, 1], k_res1, 'C7',
             '(f) Residual k-space  =  (b) sampled  −  (d)')

    # Row 3
    sig_plot(axes[3, 0], x_res1, 'C7',
             f'(g) Recon from residual k-space  (iter 2, {sigma_iter2}σ thr)', thr=thr2)
    axes[3, 1].axis('off')
    axes[3, 1].text(0.5, 0.5,
                    f'Iter 2 recovered peaks:\n{sorted(peaks2.tolist())}',
                    ha='center', va='center', fontsize=11, color='gray',
                    transform=axes[3, 1].transAxes)

    # Row 4
    sig_plot(axes[4, 0], x_combined, 'C4',
             '(i) Full reconstruction  (iter 1 + iter 2)')

    ax = axes[4, 1]
    bar_colors = ['C3' if p in peaks1 else 'C4' for p in all_peaks]
    bars = ax.bar([str(p) for p in all_peaks],
                  [pct_err[p] for p in all_peaks],
                  color=bar_colors)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('(j) % error at each recovered peak', fontsize=9)
    ax.set_xlabel('Peak position')
    ax.set_ylabel('% error')
    for bar, p in zip(bars, all_peaks):
        v = pct_err[p]
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.3 if v >= 0 else -1.0),
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.legend(handles=[
        Patch(color='C3', label=f'iter 1 ({sigma_iter1}σ)'),
        Patch(color='C4', label=f'iter 2 ({sigma_iter2}σ)'),
    ], fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    fig = make_figure()
    fig.savefig('.tmp/fig_02_cs_demo.png', dpi=120)
    print('saved → .tmp/fig_02_cs_demo.png')
