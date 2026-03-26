/**
 * Q9 — Wavelet transform at a glance.
 *
 * Four subplots:
 *   1. Original signal h(t)
 *   2. All wavelet coefficients laid out at full length, colour-coded by level
 *   3. Same layout but only a_L kept (rest zeroed)
 *   4. Reconstruction from only a_L
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q9-plot";

  // colour palette for levels
  var COLORS = [
    "#e41a1c", // d1 — red
    "#ff7f00", // d2 — orange
    "#4daf4a", // d3 — green
    "#377eb8", // d4 — blue
    "#984ea3", // d5 — purple
    "#a65628", // d6 — brown
    "#f781bf", // d7 — pink
    "#999999", // d8 — grey
  ];
  var COLOR_A = "#2c5f8a"; // a_L — dark blue

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

  function getLevels() {
    var el = document.getElementById("q6a-levels");
    return el ? parseInt(el.value, 10) || 3 : 3;
  }

  /**
   * Lay out wavelet coefficients in MRI-style order:
   * [a_L | d_L | d_{L-1} | ... | d_1]
   * Each segment is stretched to fill its proportional width of the original N samples.
   */
  function buildCoeffLayout(a, d, levels, N, tMax) {
    // Segments: a_L, d_L, d_{L-1}, ..., d_1
    // Lengths:  N/2^L, N/2^L, N/2^{L-1}, ..., N/2
    var segments = [];

    // a_L
    segments.push({ data: a[levels], label: "a" + levels, color: COLOR_A });
    // d_L down to d_1
    for (var k = levels - 1; k >= 0; k--) {
      segments.push({ data: d[k], label: "d" + (k + 1), color: COLORS[k % COLORS.length] });
    }

    // Now assign x positions: each coefficient gets an integer index
    // laid out sequentially [a_L | d_L | d_{L-1} | ... | d_1] totalling N points
    var result = [];
    var xOffset = 0;
    for (var s = 0; s < segments.length; s++) {
      var seg = segments[s];
      var segLen = seg.data.length;
      var xArr = [];
      var yArr = [];
      for (var i = 0; i < segLen; i++) {
        xArr.push(xOffset + i);
        yArr.push(seg.data[i]);
      }
      result.push({ x: xArr, y: yArr, label: seg.label, color: seg.color });
      xOffset += segLen;
    }
    return result;
  }

  function update() {
    var sig = getSignal();
    var levels = getLevels();
    var N = sig.N;

    // decompose
    var a = [sig.h];
    var d = [];
    for (var k = 0; k < levels; k++) {
      var result = lib.haarDecompose(a[k]);
      a.push(result.a);
      d.push(result.d);
    }

    // build coefficient layout
    var layout1 = buildCoeffLayout(a, d, levels, N, sig.tMax);

    // build zeroed layout (only a_L)
    var zeroD = [];
    for (var k = 0; k < levels; k++) {
      var z = new Array(d[k].length);
      for (var i = 0; i < z.length; i++) z[i] = 0;
      zeroD.push(z);
    }
    var layout2 = buildCoeffLayout(a, zeroD, levels, N, sig.tMax);

    // reconstruct from only a_L
    var current = a[levels].slice();
    for (var k = levels - 1; k >= 0; k--) {
      var zeroDk = new Array(d[k].length);
      for (var i = 0; i < zeroDk.length; i++) zeroDk[i] = 0;
      current = lib.haarReconstruct(current, zeroDk);
    }
    var recon = current;

    var traces = [];

    // Row 1: original signal (x = sample index)
    var sampleIdx = [];
    for (var i = 0; i < N; i++) sampleIdx.push(i);

    traces.push({
      x: sampleIdx, y: sig.h,
      name: "h(t)",
      line: { color: "#870000", width: 1.5 },
      xaxis: "x", yaxis: "y",
    });

    // Row 2: all coefficients colour-coded
    for (var s = 0; s < layout1.length; s++) {
      traces.push({
        x: layout1[s].x, y: layout1[s].y,
        name: layout1[s].label,
        mode: "lines",
        line: { color: layout1[s].color, width: 1 },
        xaxis: "x2", yaxis: "y2",
      });
    }

    // Row 3: zeroed coefficients (only a_L visible)
    for (var s = 0; s < layout2.length; s++) {
      traces.push({
        x: layout2[s].x, y: layout2[s].y,
        name: layout2[s].label + (s === 0 ? "" : " (zeroed)"),
        mode: "lines",
        line: { color: s === 0 ? layout2[s].color : "#ddd", width: s === 0 ? 1 : 0.5 },
        showlegend: false,
        xaxis: "x3", yaxis: "y3",
      });
    }

    // Row 4: reconstruction from a_L only
    var reconIdx = [];
    for (var i = 0; i < recon.length; i++) reconIdx.push(i);

    traces.push({
      x: reconIdx, y: recon,
      name: "recon (a" + levels + " only)",
      line: { color: COLOR_A, width: 1.5 },
      xaxis: "x4", yaxis: "y4",
      showlegend: false,
    });
    // faint original for comparison
    traces.push({
      x: sampleIdx, y: sig.h,
      name: "original",
      line: { color: "rgba(135,0,0,0.25)", width: 1 },
      xaxis: "x4", yaxis: "y4",
      showlegend: false,
    });

    var plotLayout = {
      font: { family: "STIX Two Text" },
      margin: { t: 30, b: 50, l: 60, r: 120 },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1.02, xanchor: "left", y: 1, font: { size: 10 } },
      grid: { rows: 4, columns: 1, subplots: [["xy"], ["x2y2"], ["x3y3"], ["x4y4"]], roworder: "top to bottom" },
      xaxis:  { title: "", zeroline: true, range: [0, N] },
      yaxis:  { title: "h(t)", zeroline: true },
      xaxis2: { title: "", zeroline: true, range: [0, N] },
      yaxis2: { title: "coefficients", zeroline: true },
      xaxis3: { title: "", zeroline: true, range: [0, N] },
      yaxis3: { title: "kept", zeroline: true },
      xaxis4: { title: "sample index", zeroline: true, range: [0, N] },
      yaxis4: { title: "recon", zeroline: true },
    };

    Plotly.react(plotDiv, traces, plotLayout, { responsive: true });
  }

  function init() {
    var ids = [
      "q1a-window-type", "q1a-sigma", "q1a-fermi-width", "q1a-fermi-edge",
      "q1a-center", "q1a-omega1", "q1a-omega2", "q1a-amp1", "q1a-amp2",
      "q1a-tmax", "q1a-N", "q6a-levels",
    ];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) {
        el.addEventListener("input", update);
        el.addEventListener("change", update);
      }
    });

    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
