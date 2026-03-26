/**
 * Q2a — Interactive plot of the positive kernel φ(x).
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q2a-plot";

  var defaults = {
    kernelLength: 32,
    sigma: 5.0,
  };

  function readControls() {
    return {
      kernelLength: parseInt(document.getElementById("q2a-kernel-length").value, 10),
      sigma: parseFloat(document.getElementById("q2a-sigma").value),
    };
  }

  function update() {
    var p = readControls();
    var kernel = lib.buildGaussianKernel(p.kernelLength, p.sigma);
    var x = [];
    for (var i = 0; i < p.kernelLength; i++) x.push(i);

    var traces = [
      {
        x: x,
        y: kernel,
        name: "φ(x)",
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#870000", width: 2 },
        marker: { color: "#870000", size: 4 },
      },
    ];

    var layout = {
      title: { text: "Kernel φ(x) — Gaussian, positive", font: { family: "STIX Two Text", size: 16 } },
      xaxis: { title: "sample", zeroline: true },
      yaxis: { title: "φ(x)", zeroline: true },
      font: { family: "STIX Two Text" },
      margin: { t: 50, b: 50, l: 60, r: 20 },
      hovermode: "x unified",
    };

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  function init() {
    document.getElementById("q2a-kernel-length").value = defaults.kernelLength;
    document.getElementById("q2a-sigma").value = defaults.sigma;

    var ids = ["q2a-kernel-length", "q2a-sigma"];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      el.addEventListener("input", update);
      el.addEventListener("change", update);
    });

    update();
  }

  // expose update so convolution plot can listen for changes
  window.Q2a = { update: update, readControls: readControls };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
