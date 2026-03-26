/**
 * Q4 convolution visualizer — Haar wavelet at controllable scale.
 * Two rows: signal + sliding kernel, convolution output.
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q4conv-plot";

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

  function getKernel() {
    var w = parseInt(document.getElementById("q4a-haar-width").value, 10);
    return lib.buildHaarKernel(w);
  }

  function update() {
    var sig = getSignal();
    var kernel = getKernel();
    var t = sig.t;
    var h = sig.h;
    var N = sig.N;
    var kLen = kernel.length;
    var kHalf = Math.floor(kLen / 2);

    var conv = lib.convolveSame(h, kernel);

    var sliderEl = document.getElementById("q4conv-slider");
    var pos = parseInt(sliderEl.value, 10);
    if (pos < kHalf) pos = kHalf;
    if (pos >= N - kHalf) pos = N - kHalf - 1;

    // kernel overlay scaled to signal amplitude
    var maxH = 0;
    for (var i = 0; i < N; i++) {
      var a = Math.abs(h[i]);
      if (a > maxH) maxH = a;
    }
    var kernelScale = maxH * 0.5;

    var kernelT = [];
    var kernelY = [];
    for (var i = 0; i < kLen; i++) {
      var sampleIdx = pos - kHalf + i;
      if (sampleIdx >= 0 && sampleIdx < N) {
        kernelT.push(t[sampleIdx]);
        kernelY.push(kernel[i] * kernelScale);
      }
    }

    var convT = t.slice(0, pos + 1);
    var convY = conv.slice(0, pos + 1);

    var traces = [
      {
        x: t, y: h,
        name: "h(t)",
        line: { color: "#870000", width: 1.5 },
        xaxis: "x", yaxis: "y",
      },
      {
        x: kernelT, y: kernelY,
        name: "Haar kernel",
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(255,0,0,0.15)",
        line: { color: "red", width: 2, shape: "hv" },
        xaxis: "x", yaxis: "y",
      },
      {
        x: convT, y: convY,
        name: "convolution",
        line: { color: "#870000", width: 1.5 },
        xaxis: "x2", yaxis: "y2",
      },
      {
        x: [convT[convT.length - 1]],
        y: [convY[convY.length - 1]],
        mode: "markers",
        marker: { color: "red", size: 8 },
        showlegend: false,
        xaxis: "x2", yaxis: "y2",
      },
    ];

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 30, b: 50, l: 60, r: 20 },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1, xanchor: "right", y: 1 },
      grid: { rows: 2, columns: 1, subplots: [["xy"], ["x2y2"]], roworder: "top to bottom" },
      xaxis:  { title: "", zeroline: true, range: [0, sig.tMax] },
      yaxis:  { title: "h(t) + kernel", zeroline: true },
      xaxis2: { title: "t", zeroline: true, range: [0, sig.tMax] },
      yaxis2: { title: "convolution", zeroline: true },
    };

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  function init() {
    var sig = getSignal();
    var kernel = getKernel();
    var kHalf = Math.floor(kernel.length / 2);

    var sliderEl = document.getElementById("q4conv-slider");
    sliderEl.min = kHalf;
    sliderEl.max = sig.N - kHalf - 1;
    sliderEl.value = kHalf;
    sliderEl.addEventListener("input", update);

    var q1Ids = [
      "q1a-window-type", "q1a-sigma", "q1a-fermi-width", "q1a-fermi-edge",
      "q1a-center", "q1a-omega1", "q1a-omega2", "q1a-amp1", "q1a-amp2",
      "q1a-tmax", "q1a-N",
    ];
    var q4aIds = ["q4a-haar-width"];
    q1Ids.concat(q4aIds).forEach(function (id) {
      var el = document.getElementById(id);
      el.addEventListener("input", function () {
        var s = getSignal();
        var k = getKernel();
        var kh = Math.floor(k.length / 2);
        sliderEl.min = kh;
        sliderEl.max = s.N - kh - 1;
        if (parseInt(sliderEl.value) > parseInt(sliderEl.max)) sliderEl.value = sliderEl.max;
        update();
      });
      el.addEventListener("change", function () {
        var s = getSignal();
        var k = getKernel();
        var kh = Math.floor(k.length / 2);
        sliderEl.min = kh;
        sliderEl.max = s.N - kh - 1;
        if (parseInt(sliderEl.value) > parseInt(sliderEl.max)) sliderEl.value = sliderEl.max;
        update();
      });
    });

    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
