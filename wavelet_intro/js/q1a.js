/**
 * Q1a — Interactive plot of h(t) = w(t) * [sin(ω₁t) + 10 cos(ω₂t)]
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q1a-plot";

  /* ---------- default parameter values ---------- */
  var defaults = {
    windowType: "gaussian",
    sigma: 3.0,
    fermiWidth: 6.0,
    fermiEdge: 0.3,
    center: 5.0,
    omega1: 2.0,
    omega2: 20.0,
    amp1: 1.0,
    amp2: 10.0,
    tMax: 10.0,
    N: 1024,
  };

  /* ---------- read current control values ---------- */
  function readControls() {
    return {
      windowType: document.getElementById("q1a-window-type").value,
      sigma: parseFloat(document.getElementById("q1a-sigma").value),
      fermiWidth: parseFloat(document.getElementById("q1a-fermi-width").value),
      fermiEdge: parseFloat(document.getElementById("q1a-fermi-edge").value),
      center: parseFloat(document.getElementById("q1a-center").value),
      omega1: parseFloat(document.getElementById("q1a-omega1").value),
      omega2: parseFloat(document.getElementById("q1a-omega2").value),
      amp1: parseFloat(document.getElementById("q1a-amp1").value),
      amp2: parseFloat(document.getElementById("q1a-amp2").value),
      tMax: parseFloat(document.getElementById("q1a-tmax").value),
      N: parseInt(document.getElementById("q1a-N").value, 10),
    };
  }

  /* ---------- show / hide Gaussian vs Fermi params ---------- */
  function syncWindowControls(type) {
    var gEl = document.getElementById("q1a-gaussian-params");
    var fEl = document.getElementById("q1a-fermi-params");
    if (type === "gaussian") {
      gEl.style.display = "";
      fEl.style.display = "none";
    } else {
      gEl.style.display = "none";
      fEl.style.display = "";
    }
  }

  /* ---------- update the plot ---------- */
  function update() {
    var p = readControls();
    syncWindowControls(p.windowType);

    var t = lib.linspace(0, p.tMax, p.N);
    var windowFn =
      p.windowType === "gaussian" ? lib.gaussianWindow : lib.fermiWindow;
    var wp =
      p.windowType === "gaussian"
        ? { center: p.center, sigma: p.sigma }
        : { center: p.center, width: p.fermiWidth, edge: p.fermiEdge };

    var sig = lib.buildSignal(t, windowFn, wp, p.omega1, p.omega2, p.amp1, p.amp2);

    var traces = [
      {
        x: t,
        y: sig.h,
        name: "h(t)",
        line: { color: "#870000", width: 1.5 },
      },
      {
        x: t,
        y: sig.w,
        name: "window",
        line: { color: "#999", width: 1, dash: "dot" },
      },
      {
        x: t,
        y: sig.w.map(function (v) { return -v; }),
        name: "",
        line: { color: "#999", width: 1, dash: "dot" },
        showlegend: false,
      },
    ];

    var layout = {
      title: { text: "h(t) = w(t) · [sin(ω₁t) + 10 cos(ω₂t)]", font: { family: "STIX Two Text", size: 16 } },
      xaxis: { title: "t", zeroline: true },
      yaxis: { title: "h(t)", zeroline: true },
      font: { family: "STIX Two Text" },
      margin: { t: 50, b: 50, l: 60, r: 20 },
      legend: { x: 1, xanchor: "right", y: 1 },
      hovermode: "x unified",
    };

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  /* ---------- wire up controls ---------- */
  function init() {
    // set defaults into the DOM
    document.getElementById("q1a-window-type").value = defaults.windowType;
    document.getElementById("q1a-sigma").value = defaults.sigma;
    document.getElementById("q1a-fermi-width").value = defaults.fermiWidth;
    document.getElementById("q1a-fermi-edge").value = defaults.fermiEdge;
    document.getElementById("q1a-center").value = defaults.center;
    document.getElementById("q1a-omega1").value = defaults.omega1;
    document.getElementById("q1a-omega2").value = defaults.omega2;
    document.getElementById("q1a-amp1").value = defaults.amp1;
    document.getElementById("q1a-amp2").value = defaults.amp2;
    document.getElementById("q1a-tmax").value = defaults.tMax;
    document.getElementById("q1a-N").value = defaults.N;

    syncWindowControls(defaults.windowType);

    // attach listeners to every control
    var ids = [
      "q1a-window-type", "q1a-sigma", "q1a-fermi-width", "q1a-fermi-edge",
      "q1a-center", "q1a-omega1", "q1a-omega2", "q1a-amp1", "q1a-amp2",
      "q1a-tmax", "q1a-N",
    ];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      el.addEventListener("input", update);
      el.addEventListener("change", update);
    });

    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
