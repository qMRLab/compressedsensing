---
title: Introduction to Wavelet Decomposition
subtitle: A Discovery-Based Exam
---

**Philosophy.** This exam is modelled after a Fourier decomposition exam by Prof. Normand Beaudoin (Université de Moncton, 2008). That exam forced you to *discover* the properties of spectral decomposition by trying incomplete tools and watching them fail. This exam follows the same arc for wavelets.

You will start with a kernel that cannot possibly work, fix its deficiencies one by one, and arrive at a working wavelet transform not because you memorised the axioms, but because you derived each one from necessity. By the end, you will understand why wavelets have the properties they do, how the forward and inverse transforms work, and how compression falls out naturally.

**Tools.** You are encouraged to use Maple, MATLAB, Python, or any computational tool. Show all important steps and explain your reasoning at every stage. A correct answer without explanation will receive at most half marks.

**Signal design.** The signal you will build contains a loud high frequency and a quiet low frequency inside a smooth window. The 1:10 amplitude ratio means the slow oscillation is nearly invisible in the raw signal. As you progress through the questions, you will watch the wavelet transform *reveal* the hidden frequency, separate the two components into different scales, remove the dominant one to expose the quiet one, and compress by discarding near-zero coefficients.

---

## Question 1. Build your signal

A) Define a signal $h(x)$ of the form:

$$h(x) = w(x,\,\sigma) \cdot \bigl[\sin(\omega_1 x) + 10\cos(\omega_2 x)\bigr]$$

where $w(x, \sigma)$ is a smooth window function (e.g. a Gaussian) with a width parameter $\sigma$, $\omega_1$ is a low frequency with a small amplitude, and $\omega_2$ is a high frequency with a large amplitude. The 1:10 amplitude ratio is deliberate: the slow oscillation should be nearly invisible in the raw signal but clearly visible in the wavelet coefficients at the appropriate scale. The use of sin for one component and cos for the other ensures that $h(x)$ is neither symmetric nor anti-symmetric — a requirement inherited from the Fourier exam. Choose $\omega_1$ and $\omega_2$ sufficiently far apart. Ensure that $h(x)$ tends rapidly to zero for large $|x|$.

B) Plot $h(x)$. The fast oscillation ($\omega_2$) should dominate visually. Can you see the slow oscillation ($\omega_1$) at all, or is it buried? This is deliberate — you will later use the wavelet transform to reveal it.

C) Choose $\sigma$ wide enough that several full periods of $\omega_1$ fit inside the window. Now try a $\sigma$ that is too narrow — so narrow that fewer than one or two periods of $\omega_1$ fit inside. Plot both cases. In the narrow case, can you distinguish the slow oscillation from a simple slope or offset? Keep the wider $\sigma$ for all subsequent questions.

> *Your signal plays the role of $h(x)$ in the Fourier exam. The kernels you will try in the following questions play the role of cos, sin, and complex exponentials. The window width matters practically: if $\sigma$ is too small, the low frequency doesn't have room to express itself, and no transform can recover what isn't there.*

```{raw} html
<div class="controls-panel">
  <label>Window
    <select id="q1a-window-type">
      <option value="gaussian">Gaussian</option>
      <option value="fermi">Fermi</option>
    </select>
  </label>
  <span id="q1a-gaussian-params" style="display:contents">
    <label>σ <input type="number" id="q1a-sigma" step="0.1" min="0.1"></label>
  </span>
  <span id="q1a-fermi-params" style="display:none">
    <label>width <input type="number" id="q1a-fermi-width" step="0.5" min="0.5"></label>
    <label>edge <input type="number" id="q1a-fermi-edge" step="0.05" min="0.01"></label>
  </span>
  <label>center <input type="number" id="q1a-center" step="0.5"></label>
  <label>ω₁ <input type="number" id="q1a-omega1" step="0.5" min="0.1"></label>
  <label>ω₂ <input type="number" id="q1a-omega2" step="1" min="0.1"></label>
  <label>amp₁ <input type="number" id="q1a-amp1" step="0.5"></label>
  <label>amp₂ <input type="number" id="q1a-amp2" step="1"></label>
  <label>t_max <input type="number" id="q1a-tmax" step="1" min="1"></label>
  <label>N <input type="number" id="q1a-N" step="128" min="128"></label>
</div>
<div id="q1a-plot" style="width:100%;height:400px;"></div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="/_static/js/signal.js"></script>
<script src="/_static/js/q1a.js"></script>
```

This is a test — if this renders correctly with working interactivity, the full exam can be converted.
