/**
 * Q6 — The cascade: multi-scale decomposition.
 *
 * Q6a: Shows the cascade step by step — user picks number of levels (1–4).
 *      Each level: low-pass output from previous level → filter → downsample.
 *      Displayed as stacked rows: a_prev → (filter+↓2) → a_k and d_k.
 *
 * Q6b: Stacked plot of d₁, d₂, d₃, ..., d_L, a_L aligned by position.
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

  function getFilterLength() {
    var el = document.getElementById("q5a-filter-length");
    return el ? parseInt(el.value, 10) || 2 : 2;
  }

  /**
   * Run the cascade for `levels` levels.
   * Returns { d: [d1, d2, ...], a: [a0, a1, a2, ...], tArrays: [...] }
   * where a[0] = original signal, a[k] = approx at level k.
   */
  function cascade(signal, tArray, levels, fLen) {
    var d = [];
    var a = [signal];
    var tArrays = [tArray];

    for (var k = 0; k < levels; k++) {
      var prev = a[k];
      var prevT = tArrays[k];

      var result = lib.haarDecompose(prev);
      var ak = result.a;
      var dk = result.d;

      var tk = [];
      for (var i = 0; i < prev.length; i += 2) {
        tk.push(prevT[i]);
      }

      d.push(dk);
      a.push(ak);
      tArrays.push(tk);
    }

    return { d: d, a: a, tArrays: tArrays };
  }

  /* ---- Q6a: cascade step-by-step ---- */
  function updateQ6a() {
    var sig = getSignal();
    var levels = parseInt(document.getElementById("q6a-levels").value, 10) || 3;
    var fLen = getFilterLength();
    var res = cascade(sig.h, sig.t, levels, fLen);

    var traces = [];
    var nRows = levels + 1; // original + one row per level

    // Row 1: original signal
    traces.push({
      x: sig.t, y: sig.h,
      name: "h(t)",
      line: { color: "#870000", width: 1.5 },
      xaxis: "x", yaxis: "y",
    });

    // Subsequent rows: each level shows a_k (blue) and d_k (red) side by side
    var colors = ["#2c5f8a", "#870000", "#2e7d32", "#6f42c1"];
    for (var k = 0; k < levels; k++) {
      var axSuffix = (k + 2).toString();
      var xax = "x" + axSuffix;
      var yax = "y" + axSuffix;

      // d_k
      traces.push({
        x: res.tArrays[k + 1], y: res.d[k],
        name: "d" + (k + 1),
        mode: "lines+markers",
        line: { color: "#870000", width: 0.5 },
        marker: { color: "#870000", size: 3 },
        xaxis: xax, yaxis: yax,
      });

      // a_k
      traces.push({
        x: res.tArrays[k + 1], y: res.a[k + 1],
        name: "a" + (k + 1),
        mode: "lines+markers",
        line: { color: "#2c5f8a", width: 0.5 },
        marker: { color: "#2c5f8a", size: 3 },
        xaxis: xax, yaxis: yax,
      });
    }

    var subplots = [["xy"]];
    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 30, b: 50, l: 60, r: 20 },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1, xanchor: "right", y: 1, font: { size: 11 } },
      xaxis: { title: "", zeroline: true, range: [0, sig.tMax] },
      yaxis: { title: "h(t)", zeroline: true },
    };

    for (var k = 0; k < levels; k++) {
      var axSuffix = (k + 2).toString();
      subplots.push(["x" + axSuffix + "y" + axSuffix]);
      layout["xaxis" + axSuffix] = { title: k === levels - 1 ? "t" : "", zeroline: true, range: [0, sig.tMax] };
      layout["yaxis" + axSuffix] = { title: "level " + (k + 1), zeroline: true };
    }

    layout.grid = { rows: nRows, columns: 1, subplots: subplots, roworder: "top to bottom" };

    var height = 150 + nRows * 120;
    Plotly.react("q6a-plot", traces, layout, { responsive: true });
    document.getElementById("q6a-plot").style.height = height + "px";
  }

  /* ---- Q6b: stacked d₁, d₂, ..., d_L, a_L ---- */
  function updateQ6b() {
    var sig = getSignal();
    var levels = parseInt(document.getElementById("q6a-levels").value, 10) || 3;
    var fLen = getFilterLength();
    var res = cascade(sig.h, sig.t, levels, fLen);

    var nRows = levels + 1; // d₁..d_L + a_L
    var traces = [];
    var subplots = [];

    for (var k = 0; k < levels; k++) {
      var axSuffix = k === 0 ? "" : (k + 1).toString();
      var xax = "x" + (k === 0 ? "" : axSuffix);
      var yax = "y" + (k === 0 ? "" : axSuffix);

      traces.push({
        x: res.tArrays[k + 1], y: res.d[k],
        name: "d" + (k + 1) + " (" + res.d[k].length + " pts)",
        mode: "lines+markers",
        line: { color: "#870000", width: 0.5 },
        marker: { color: "#870000", size: 3 },
        xaxis: xax, yaxis: yax,
      });

      subplots.push([xax.replace("x", "x") + yax.replace("y", "y")]);
    }

    // final row: a_L
    var lastAx = levels + 1;
    var xaxLast = "x" + lastAx;
    var yaxLast = "y" + lastAx;
    traces.push({
      x: res.tArrays[levels], y: res.a[levels],
      name: "a" + levels + " (" + res.a[levels].length + " pts)",
      mode: "lines+markers",
      line: { color: "#2c5f8a", width: 0.5 },
      marker: { color: "#2c5f8a", size: 3 },
      xaxis: xaxLast, yaxis: yaxLast,
    });

    // build layout
    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 30, b: 50, l: 60, r: 20 },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1, xanchor: "right", y: 1, font: { size: 11 } },
    };

    var subplotGrid = [];
    for (var k = 0; k < levels; k++) {
      var axSuffix = k === 0 ? "" : (k + 1).toString();
      var xKey = "xaxis" + (k === 0 ? "" : axSuffix);
      var yKey = "yaxis" + (k === 0 ? "" : axSuffix);
      layout[xKey] = { title: "", zeroline: true, range: [0, sig.tMax] };
      layout[yKey] = { title: "d" + (k + 1), zeroline: true };
      var xRef = "x" + (k === 0 ? "" : axSuffix);
      var yRef = "y" + (k === 0 ? "" : axSuffix);
      subplotGrid.push([xRef + yRef]);
    }
    layout["xaxis" + lastAx] = { title: "t", zeroline: true, range: [0, sig.tMax] };
    layout["yaxis" + lastAx] = { title: "a" + levels, zeroline: true };
    subplotGrid.push([xaxLast + yaxLast]);

    layout.grid = { rows: nRows, columns: 1, subplots: subplotGrid, roworder: "top to bottom" };

    // total coefficients annotation
    var total = 0;
    for (var k = 0; k < levels; k++) total += res.d[k].length;
    total += res.a[levels].length;

    var height = 100 + nRows * 110;
    Plotly.react("q6b-plot", traces, layout, { responsive: true });
    document.getElementById("q6b-plot").style.height = height + "px";

    // update coefficient count
    var countEl = document.getElementById("q6b-count");
    if (countEl) {
      var parts = [];
      for (var k = 0; k < levels; k++) parts.push("|d" + (k+1) + "| = " + res.d[k].length);
      parts.push("|a" + levels + "| = " + res.a[levels].length);
      countEl.textContent = parts.join(" + ") + " = " + total + " (N = " + sig.N + ")";
    }
  }

  function updateAll() {
    updateQ6a();
    updateQ6b();
  }

  function init() {
    document.getElementById("q6a-levels").value = 3;

    var ids = [
      "q6a-levels",
      "q1a-window-type", "q1a-sigma", "q1a-fermi-width", "q1a-fermi-edge",
      "q1a-center", "q1a-omega1", "q1a-omega2", "q1a-amp1", "q1a-amp2",
      "q1a-tmax", "q1a-N",
    ];
    var fLenEl = document.getElementById("q5a-filter-length");
    if (fLenEl) ids.push("q5a-filter-length");

    ids.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) {
        el.addEventListener("input", updateAll);
        el.addEventListener("change", updateAll);
      }
    });

    updateAll();
  }

  window.Q6 = { cascade: cascade, getSignal: getSignal, getFilterLength: getFilterLength };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
