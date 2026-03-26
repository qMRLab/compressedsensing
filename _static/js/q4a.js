/**
 * Q4a — Interactive plot of the Haar wavelet at a controllable scale.
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q4a-plot";

  var defaults = {
    haarWidth: 32,
  };

  function readControls() {
    return {
      haarWidth: parseInt(document.getElementById("q4a-haar-width").value, 10),
    };
  }

  function update() {
    var p = readControls();
    var kernel = lib.buildHaarKernel(p.haarWidth);
    var x = [];
    for (var i = 0; i < p.haarWidth; i++) x.push(i);

    var traces = [
      {
        x: x,
        y: kernel,
        name: "ψ(x) Haar",
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#870000", width: 2, shape: "hv" },
        marker: { color: "#870000", size: 4 },
      },
    ];

    var layout = {
      title: { text: "Haar wavelet ψ(x) — width = " + p.haarWidth + " samples", font: { family: "STIX Two Text", size: 16 } },
      xaxis: { title: "sample", zeroline: true },
      yaxis: { title: "ψ(x)", zeroline: true, range: [-1.5, 1.5] },
      font: { family: "STIX Two Text" },
      margin: { t: 50, b: 50, l: 60, r: 20 },
      hovermode: "x unified",
    };

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  function init() {
    document.getElementById("q4a-haar-width").value = defaults.haarWidth;

    var el = document.getElementById("q4a-haar-width");
    el.addEventListener("input", update);
    el.addEventListener("change", update);

    update();
  }

  window.Q4a = { update: update, readControls: readControls };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
