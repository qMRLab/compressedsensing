/**
 * Q3a — Interactive plot of a rough, zero-mean kernel.
 * Options: Sawtooth (rough) or Difference-of-Gaussians (smoother, for comparison).
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q3a-plot";

  var defaults = {
    kernelType: "bipolar",
    kernelLength: 32,
    bipolarSigma: 4.0,
    bipolarSep: 12.0,
    dogSigma1: 3.0,
    dogSigma2: 6.0,
  };

  function readControls() {
    return {
      kernelType: document.getElementById("q3a-kernel-type").value,
      kernelLength: parseInt(document.getElementById("q3a-kernel-length").value, 10),
      bipolarSigma: parseFloat(document.getElementById("q3a-bipolar-sigma").value),
      bipolarSep: parseFloat(document.getElementById("q3a-bipolar-sep").value),
      dogSigma1: parseFloat(document.getElementById("q3a-dog-sigma1").value),
      dogSigma2: parseFloat(document.getElementById("q3a-dog-sigma2").value),
    };
  }

  function getKernel(p) {
    if (p.kernelType === "bipolar") {
      return lib.buildBipolarGaussianKernel(p.kernelLength, p.bipolarSigma, p.bipolarSep);
    } else {
      return lib.buildDoGKernel(p.kernelLength, p.dogSigma1, p.dogSigma2);
    }
  }

  function syncControls(type) {
    document.getElementById("q3a-bipolar-params").style.display = type === "bipolar" ? "" : "none";
    document.getElementById("q3a-dog-params").style.display = type === "dog" ? "" : "none";
  }

  function update() {
    var p = readControls();
    syncControls(p.kernelType);
    var kernel = getKernel(p);
    var x = [];
    for (var i = 0; i < kernel.length; i++) x.push(i);

    var traces = [
      {
        x: x,
        y: kernel,
        name: "ψ(x)",
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#870000", width: 2 },
        marker: { color: "#870000", size: 4 },
      },
    ];

    var layout = {
      title: { text: "Zero-mean kernel ψ(x) — " + (p.kernelType === "bipolar" ? "Bipolar Gaussian" : "Difference of Gaussians"), font: { family: "STIX Two Text", size: 16 } },
      xaxis: { title: "sample", zeroline: true },
      yaxis: { title: "ψ(x)", zeroline: true },
      font: { family: "STIX Two Text" },
      margin: { t: 50, b: 50, l: 60, r: 20 },
      hovermode: "x unified",
    };

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  function init() {
    document.getElementById("q3a-kernel-type").value = defaults.kernelType;
    document.getElementById("q3a-kernel-length").value = defaults.kernelLength;
    document.getElementById("q3a-bipolar-sigma").value = defaults.bipolarSigma;
    document.getElementById("q3a-bipolar-sep").value = defaults.bipolarSep;
    document.getElementById("q3a-dog-sigma1").value = defaults.dogSigma1;
    document.getElementById("q3a-dog-sigma2").value = defaults.dogSigma2;
    syncControls(defaults.kernelType);

    var ids = ["q3a-kernel-type", "q3a-kernel-length", "q3a-bipolar-sigma", "q3a-bipolar-sep", "q3a-dog-sigma1", "q3a-dog-sigma2"];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      el.addEventListener("input", update);
      el.addEventListener("change", update);
    });

    update();
  }

  window.Q3a = { update: update, readControls: readControls, getKernel: getKernel };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
