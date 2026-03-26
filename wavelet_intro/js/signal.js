/**
 * WaveletLib — reusable signal utilities for the wavelet exam.
 * Attached to window.WaveletLib for use across questions.
 */
(function () {
  "use strict";

  function linspace(start, stop, n) {
    var arr = new Array(n);
    var step = (stop - start) / (n - 1);
    for (var i = 0; i < n; i++) arr[i] = start + i * step;
    return arr;
  }

  /**
   * Gaussian window centred at `center` with standard deviation `sigma`.
   *   w(t) = exp(-0.5 * ((t - center) / sigma)^2)
   */
  function gaussianWindow(t, center, sigma) {
    var u = (t - center) / sigma;
    return Math.exp(-0.5 * u * u);
  }

  /**
   * Fermi–Dirac window: flat in the middle, smooth roll-off at the edges.
   *   w(t) = 1 / { [1 + exp((|t - center| - width/2) / edge)] }
   *
   * `width`  — full width of the flat-top region
   * `edge`   — controls steepness of the roll-off (smaller = sharper)
   */
  function fermiWindow(t, center, width, edge) {
    var d = Math.abs(t - center) - width / 2;
    return 1.0 / (1.0 + Math.exp(d / edge));
  }

  /**
   * Build the exam signal:
   *   h(t) = w(t) * [ amp1 * sin(omega1 * t) + amp2 * cos(omega2 * t) ]
   *
   * @param {number[]} tArray  — sample points
   * @param {function} windowFn — one of gaussianWindow / fermiWindow
   * @param {object}   wp       — window parameters (passed after t to windowFn)
   * @param {number}   omega1
   * @param {number}   omega2
   * @param {number}   amp1
   * @param {number}   amp2
   * @returns {{h: number[], w: number[], osc: number[]}}
   *   h   = full signal
   *   w   = window envelope
   *   osc = oscillatory part (before windowing)
   */
  function buildSignal(tArray, windowFn, wp, omega1, omega2, amp1, amp2) {
    var n = tArray.length;
    var h = new Array(n);
    var w = new Array(n);
    var osc = new Array(n);
    for (var i = 0; i < n; i++) {
      var t = tArray[i];
      var wVal;
      if (windowFn === gaussianWindow) {
        wVal = gaussianWindow(t, wp.center, wp.sigma);
      } else {
        wVal = fermiWindow(t, wp.center, wp.width, wp.edge);
      }
      var oscVal = amp1 * Math.sin(omega1 * t) + amp2 * Math.cos(omega2 * t);
      w[i] = wVal;
      osc[i] = oscVal;
      h[i] = wVal * oscVal;
    }
    return { h: h, w: w, osc: osc };
  }

  /**
   * Build a Gaussian kernel of `length` samples, centred in the array.
   * The kernel is normalised so its values sum to 1.
   *
   * @param {number} length — number of samples (e.g. 32)
   * @param {number} sigma  — standard deviation in sample units
   * @returns {number[]}
   */
  function buildGaussianKernel(length, sigma) {
    var kernel = new Array(length);
    var center = (length - 1) / 2;
    var sum = 0;
    for (var i = 0; i < length; i++) {
      var u = (i - center) / sigma;
      kernel[i] = Math.exp(-0.5 * u * u);
      sum += kernel[i];
    }
    for (var i = 0; i < length; i++) kernel[i] /= sum;
    return kernel;
  }

  /**
   * Discrete convolution (full output).
   * Returns array of length signal.length + kernel.length - 1.
   *
   * @param {number[]} signal
   * @param {number[]} kernel
   * @returns {number[]}
   */
  function convolve(signal, kernel) {
    var sLen = signal.length;
    var kLen = kernel.length;
    var outLen = sLen + kLen - 1;
    var out = new Array(outLen);
    for (var i = 0; i < outLen; i++) {
      var sum = 0;
      for (var j = 0; j < kLen; j++) {
        var si = i - j;
        if (si >= 0 && si < sLen) {
          sum += signal[si] * kernel[j];
        }
      }
      out[i] = sum;
    }
    return out;
  }

  /**
   * Same-size convolution: returns the central portion of the full
   * convolution, matching the input signal length.
   */
  function convolveSame(signal, kernel) {
    var full = convolve(signal, kernel);
    var offset = Math.floor(kernel.length / 2);
    return full.slice(offset, offset + signal.length);
  }

  /**
   * Bipolar Gaussian kernel: a positive Gaussian bump followed by a
   * negative Gaussian bump. Zero-mean by symmetry.
   *
   * @param {number} length     — number of samples
   * @param {number} sigma      — width of each Gaussian lobe (in samples)
   * @param {number} separation — distance between lobe centres (in samples)
   */
  function buildBipolarGaussianKernel(length, sigma, separation) {
    var kernel = new Array(length);
    var mid = (length - 1) / 2;
    var q1 = mid - separation / 2;   // centre of positive lobe
    var q3 = mid + separation / 2;   // centre of negative lobe
    for (var i = 0; i < length; i++) {
      var u1 = (i - q1) / sigma;
      var u2 = (i - q3) / sigma;
      kernel[i] = Math.exp(-0.5 * u1 * u1) - Math.exp(-0.5 * u2 * u2);
    }
    // force exact zero mean (should already be ~0 by symmetry)
    var mean = 0;
    for (var i = 0; i < length; i++) mean += kernel[i];
    mean /= length;
    for (var i = 0; i < length; i++) kernel[i] -= mean;
    return kernel;
  }

  /**
   * Difference-of-Gaussians (DoG) kernel: a narrow positive Gaussian
   * minus a wider negative Gaussian. Zero-mean by construction (after
   * explicit mean removal). Smooth but zero-mean.
   */
  function buildDoGKernel(length, sigma1, sigma2) {
    var kernel = new Array(length);
    var center = (length - 1) / 2;
    for (var i = 0; i < length; i++) {
      var u = i - center;
      var g1 = Math.exp(-0.5 * (u / sigma1) * (u / sigma1));
      var g2 = Math.exp(-0.5 * (u / sigma2) * (u / sigma2));
      kernel[i] = g1 - g2;
    }
    // force exact zero mean
    var mean = 0;
    for (var i = 0; i < length; i++) mean += kernel[i];
    mean /= length;
    for (var i = 0; i < length; i++) kernel[i] -= mean;
    return kernel;
  }

  /**
   * Haar wavelet kernel of a given width (in samples).
   * First half = +1, second half = -1.
   *
   * @param {number} width — total number of samples
   * @returns {number[]}
   */
  function buildHaarKernel(width) {
    var kernel = new Array(width);
    var half = Math.floor(width / 2);
    for (var i = 0; i < width; i++) {
      kernel[i] = i < half ? 1 : -1;
    }
    return kernel;
  }

  /**
   * Haar filter bank: low-pass and high-pass analysis filters.
   */
  var HAAR_LO = [1 / Math.sqrt(2), 1 / Math.sqrt(2)];
  var HAAR_HI = [1 / Math.sqrt(2), -1 / Math.sqrt(2)];

  /**
   * Build a Haar-like low-pass filter of given length.
   * All coefficients = 1/sqrt(length), so it averages and normalises.
   */
  function buildHaarLo(length) {
    var val = 1 / Math.sqrt(length);
    var f = new Array(length);
    for (var i = 0; i < length; i++) f[i] = val;
    return f;
  }

  /**
   * Build a Haar-like high-pass filter of given length.
   * First half = +1/sqrt(length), second half = -1/sqrt(length).
   */
  function buildHaarHi(length) {
    var val = 1 / Math.sqrt(length);
    var half = Math.floor(length / 2);
    var f = new Array(length);
    for (var i = 0; i < length; i++) {
      f[i] = i < half ? val : -val;
    }
    return f;
  }

  /**
   * One level of the standard Haar DWT.
   * a[n] = (x[2n] + x[2n+1]) / sqrt(2)
   * d[n] = (x[2n] - x[2n+1]) / sqrt(2)
   * Returns { a, d }, each of length floor(signal.length / 2).
   */
  function haarDecompose(signal) {
    var N = signal.length;
    var half = Math.floor(N / 2);
    var a = new Array(half);
    var d = new Array(half);
    var s = 1 / Math.sqrt(2);
    for (var n = 0; n < half; n++) {
      a[n] = (signal[2 * n] + signal[2 * n + 1]) * s;
      d[n] = (signal[2 * n] - signal[2 * n + 1]) * s;
    }
    return { a: a, d: d };
  }

  /**
   * One level of Haar inverse DWT.
   * x[2n]     = (a[n] + d[n]) / sqrt(2)
   * x[2n + 1] = (a[n] - d[n]) / sqrt(2)
   * Returns array of length 2 * ak.length.
   */
  function haarReconstruct(ak, dk) {
    var half = ak.length;
    var out = new Array(half * 2);
    var s = 1 / Math.sqrt(2);
    for (var n = 0; n < half; n++) {
      out[2 * n]     = (ak[n] + dk[n]) * s;
      out[2 * n + 1] = (ak[n] - dk[n]) * s;
    }
    return out;
  }

  /**
   * Convolve with a filter and downsample by 2 (for Q5 visualization).
   * Uses full convolution, takes every 2nd sample from convolveSame output.
   */
  function filterAndDownsample(signal, filter) {
    var conv = convolveSame(signal, filter);
    var out = [];
    for (var i = 0; i < conv.length; i += 2) {
      out.push(conv[i]);
    }
    return out;
  }

  // Public API
  window.WaveletLib = {
    linspace: linspace,
    gaussianWindow: gaussianWindow,
    fermiWindow: fermiWindow,
    buildSignal: buildSignal,
    buildGaussianKernel: buildGaussianKernel,
    buildBipolarGaussianKernel: buildBipolarGaussianKernel,
    buildDoGKernel: buildDoGKernel,
    buildHaarKernel: buildHaarKernel,
    convolve: convolve,
    convolveSame: convolveSame,
    HAAR_LO: HAAR_LO,
    HAAR_HI: HAAR_HI,
    buildHaarLo: buildHaarLo,
    buildHaarHi: buildHaarHi,
    haarDecompose: haarDecompose,
    haarReconstruct: haarReconstruct,
    filterAndDownsample: filterAndDownsample,
  };
})();
