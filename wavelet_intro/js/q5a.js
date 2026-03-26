/**
 * Q5a — Two-panel filter bank visualizer.
 *
 * Panel 1 (top pair):  signal + high-pass Haar filter sliding → d₁ (detail)
 * Panel 2 (bottom pair): signal + low-pass Haar filter sliding → a₁ (approx)
 *
 * Each panel has: top row = signal + kernel overlay, bottom row = filtered + downsampled output.
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;

  function getSignal() {
    var wt = document.getElementById("q1a-window-type").value;
    var sigma = parseFloat(document.getElementById("q1a-sigma").value);
    var fWidth = parseFloat(document.getElementById("q1a-fermi-width").value);
    var fEdge = parseFloat(document.getElementById("q1a-fermi-edge").value);
    var center = parseFloat(document.getElementById("q1a-center").value);
    var omega1 = parseFloat(document.getElementById("q1a-omega1").value);
    var omega2 = parseFloat(document.getElementById("q1a-omega2").value);
    var amp1 = parseFloat(document.getElementById("q1a-amp1").value);
    var amp2 = parseFloat(document.getElementById("q1a-amp2").value);
    var tMax = parseFloat(document.getElementById("q1a-tmax").value);
    var N = parseInt(document.getElementById("q1a-N").value, 10);

    var t = lib.linspace(0, tMax, N);
    var windowFn = wt === "gaussian" ? lib.gaussianWindow : lib.fermiWindow;
    var wp = wt === "gaussian"
      ? { center: center, sigma: sigma }
      : { center: center, width: fWidth, edge: fEdge };

    var sig = lib.buildSignal(t, windowFn, wp, omega1, omega2, amp1, amp2);
    return { t: t, h: sig.h, N: N, tMax: tMax };
  }

  /* ---- high-pass plot ---- */
  function updateHi() {
    var sig = getSignal();
    var t = sig.t;
    var h = sig.h;
    var N = sig.N;
    var fLen = parseInt(document.getElementById("q5a-filter-length").value, 10) || 2;
    var filter = lib.buildHaarHi(fLen);
    var kLen = filter.length;
    var kHalf = Math.floor(kLen / 2);

    var convFull = lib.convolveSame(h, filter);
    // downsampled
    var d1 = [];
    var d1t = [];
    for (var i = 0; i < N; i += 2) {
      d1.push(convFull[i]);
      d1t.push(t[i]);
    }

    // slider
    var sliderEl = document.getElementById("q5a-hi-slider");
    var pos = parseInt(sliderEl.value, 10);
    if (pos < kHalf) pos = kHalf;
    if (pos >= N - kHalf) pos = N - kHalf - 1;

    // kernel overlay
    var maxH = 0;
    for (var i = 0; i < N; i++) { var a = Math.abs(h[i]); if (a > maxH) maxH = a; }
    var maxK = 0;
    for (var i = 0; i < kLen; i++) { var a = Math.abs(filter[i]); if (a > maxK) maxK = a; }
    var kernelScale = maxK > 0 ? maxH * 0.5 / maxK : 1;

    var kernelT = [];
    var kernelY = [];
    for (var i = 0; i < kLen; i++) {
      var si = pos - kHalf + i;
      if (si >= 0 && si < N) {
        kernelT.push(t[si]);
        kernelY.push(filter[i] * kernelScale);
      }
    }

    // partial convolution (before downsampling) up to pos
    var convT = t.slice(0, pos + 1);
    var convY = convFull.slice(0, pos + 1);

    // downsampled partial
    var d1partial = [];
    var d1tpartial = [];
    for (var i = 0; i <= pos; i += 2) {
      d1partial.push(convFull[i]);
      d1tpartial.push(t[i]);
    }

    var traces = [
      { x: t, y: h, name: "h(t)", line: { color: "#870000", width: 1.5 }, xaxis: "x", yaxis: "y" },
      { x: kernelT, y: kernelY, name: "g (high-pass)", mode: "lines", fill: "tozeroy",
        fillcolor: "rgba(255,0,0,0.15)", line: { color: "red", width: 2, shape: "hv" }, xaxis: "x", yaxis: "y" },
      { x: convT, y: convY, name: "g * h(t)", mode: "lines",
        line: { color: "rgba(135,0,0,0.3)", width: 1.5 }, showlegend: false, xaxis: "x2", yaxis: "y2" },
      { x: d1tpartial, y: d1partial, name: "d₁ (↓2)", mode: "markers",
        marker: { color: "#870000", size: 4 }, xaxis: "x2", yaxis: "y2" },
      { x: d1tpartial.length ? [d1tpartial[d1tpartial.length-1]] : [], y: d1partial.length ? [d1partial[d1partial.length-1]] : [],
        mode: "markers", marker: { color: "red", size: 8 }, showlegend: false, xaxis: "x2", yaxis: "y2" },
    ];

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 40, b: 50, l: 60, r: 20 },
      title: { text: "High-pass filter g → d₁ (detail coefficients)", font: { family: "STIX Two Text", size: 15 } },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1, xanchor: "right", y: 1 },
      grid: { rows: 2, columns: 1, subplots: [["xy"], ["x2y2"]], roworder: "top to bottom" },
      xaxis:  { title: "", zeroline: true, range: [0, sig.tMax] },
      yaxis:  { title: "h(t)", zeroline: true },
      xaxis2: { title: "t", zeroline: true, range: [0, sig.tMax] },
      yaxis2: { title: "d₁", zeroline: true },
    };

    Plotly.react("q5a-hi-plot", traces, layout, { responsive: true });
  }

  /* ---- low-pass plot ---- */
  function updateLo() {
    var sig = getSignal();
    var t = sig.t;
    var h = sig.h;
    var N = sig.N;
    var fLen = parseInt(document.getElementById("q5a-filter-length").value, 10) || 2;
    var filter = lib.buildHaarLo(fLen);
    var kLen = filter.length;
    var kHalf = Math.floor(kLen / 2);

    var convFull = lib.convolveSame(h, filter);
    // downsampled
    var a1 = [];
    var a1t = [];
    for (var i = 0; i < N; i += 2) {
      a1.push(convFull[i]);
      a1t.push(t[i]);
    }

    // slider
    var sliderEl = document.getElementById("q5a-lo-slider");
    var pos = parseInt(sliderEl.value, 10);
    if (pos < kHalf) pos = kHalf;
    if (pos >= N - kHalf) pos = N - kHalf - 1;

    // kernel overlay
    var maxH = 0;
    for (var i = 0; i < N; i++) { var a = Math.abs(h[i]); if (a > maxH) maxH = a; }
    var maxK = 0;
    for (var i = 0; i < kLen; i++) { var a = Math.abs(filter[i]); if (a > maxK) maxK = a; }
    var kernelScale = maxK > 0 ? maxH * 0.5 / maxK : 1;

    var kernelT = [];
    var kernelY = [];
    for (var i = 0; i < kLen; i++) {
      var si = pos - kHalf + i;
      if (si >= 0 && si < N) {
        kernelT.push(t[si]);
        kernelY.push(filter[i] * kernelScale);
      }
    }

    // partial convolution (before downsampling) up to pos
    var convLoT = t.slice(0, pos + 1);
    var convLoY = convFull.slice(0, pos + 1);

    // downsampled partial
    var a1partial = [];
    var a1tpartial = [];
    for (var i = 0; i <= pos; i += 2) {
      a1partial.push(convFull[i]);
      a1tpartial.push(t[i]);
    }

    var traces = [
      { x: t, y: h, name: "h(t)", line: { color: "#870000", width: 1.5 }, xaxis: "x", yaxis: "y" },
      { x: kernelT, y: kernelY, name: "h (low-pass)", mode: "lines", fill: "tozeroy",
        fillcolor: "rgba(0,100,255,0.15)", line: { color: "#2c5f8a", width: 2, shape: "hv" }, xaxis: "x", yaxis: "y" },
      { x: convLoT, y: convLoY, name: "h * h(t)", mode: "lines",
        line: { color: "rgba(44,95,138,0.3)", width: 1.5 }, showlegend: false, xaxis: "x2", yaxis: "y2" },
      { x: a1tpartial, y: a1partial, name: "a₁ (↓2)", mode: "markers",
        marker: { color: "#2c5f8a", size: 4 }, xaxis: "x2", yaxis: "y2" },
      { x: a1tpartial.length ? [a1tpartial[a1tpartial.length-1]] : [], y: a1partial.length ? [a1partial[a1partial.length-1]] : [],
        mode: "markers", marker: { color: "red", size: 8 }, showlegend: false, xaxis: "x2", yaxis: "y2" },
    ];

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 40, b: 50, l: 60, r: 20 },
      title: { text: "Low-pass filter h → a₁ (approximation coefficients)", font: { family: "STIX Two Text", size: 15 } },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1, xanchor: "right", y: 1 },
      grid: { rows: 2, columns: 1, subplots: [["xy"], ["x2y2"]], roworder: "top to bottom" },
      xaxis:  { title: "", zeroline: true, range: [0, sig.tMax] },
      yaxis:  { title: "h(t)", zeroline: true },
      xaxis2: { title: "t", zeroline: true, range: [0, sig.tMax] },
      yaxis2: { title: "a₁", zeroline: true },
    };

    Plotly.react("q5a-lo-plot", traces, layout, { responsive: true });
  }

  function init() {
    var sig = getSignal();
    var N = sig.N;

    // high-pass slider
    var hiSlider = document.getElementById("q5a-hi-slider");
    hiSlider.min = 1;
    hiSlider.max = N - 2;
    hiSlider.value = 1;
    hiSlider.addEventListener("input", updateHi);

    // low-pass slider
    var loSlider = document.getElementById("q5a-lo-slider");
    loSlider.min = 1;
    loSlider.max = N - 2;
    loSlider.value = 1;
    loSlider.addEventListener("input", updateLo);

    // listen to Q1 and Q5a control changes
    var q1Ids = [
      "q1a-window-type", "q1a-sigma", "q1a-fermi-width", "q1a-fermi-edge",
      "q1a-center", "q1a-omega1", "q1a-omega2", "q1a-amp1", "q1a-amp2",
      "q1a-tmax", "q1a-N", "q5a-filter-length",
    ];
    q1Ids.forEach(function (id) {
      var el = document.getElementById(id);
      el.addEventListener("input", function () {
        var s = getSignal();
        hiSlider.max = s.N - 2;
        loSlider.max = s.N - 2;
        if (parseInt(hiSlider.value) > parseInt(hiSlider.max)) hiSlider.value = hiSlider.max;
        if (parseInt(loSlider.value) > parseInt(loSlider.max)) loSlider.value = loSlider.max;
        updateHi();
        updateLo();
      });
      el.addEventListener("change", function () {
        var s = getSignal();
        hiSlider.max = s.N - 2;
        loSlider.max = s.N - 2;
        if (parseInt(hiSlider.value) > parseInt(hiSlider.max)) hiSlider.value = hiSlider.max;
        if (parseInt(loSlider.value) > parseInt(loSlider.max)) loSlider.value = loSlider.max;
        updateHi();
        updateLo();
      });
    });

    updateHi();
    updateLo();
  }

  // expose for Q6
  window.Q5a = { getSignal: getSignal };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
