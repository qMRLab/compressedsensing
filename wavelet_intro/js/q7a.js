/**
 * Q7a — Reconstruction from partial information.
 *
 * Shows step-by-step reconstruction as top-down subplots.
 * User can zero out coefficients via checkboxes.
 * Default: d₁=d₂=d₃=0 (keep only a₃).
 */
(function () {
  "use strict";

  var lib = window.WaveletLib;
  var plotDiv = "q7a-plot";

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

  function getLevels() {
    var el = document.getElementById("q6a-levels");
    return el ? parseInt(el.value, 10) || 3 : 3;
  }

  function zeros(n) {
    var a = new Array(n);
    for (var i = 0; i < n; i++) a[i] = 0;
    return a;
  }

  function update() {
    var sig = getSignal();
    var levels = getLevels();
    var fLen = getFilterLength();

    // decompose using standard Haar DWT
    var a = [sig.h];
    var d = [];
    for (var k = 0; k < levels; k++) {
      var result = lib.haarDecompose(a[k]);
      a.push(result.a);
      d.push(result.d);
    }

    // read which coefficients to zero
    var keepA = document.getElementById("q7a-keep-a").checked;
    var keepD = [];
    for (var k = 0; k < levels; k++) {
      var el = document.getElementById("q7a-keep-d" + (k + 1));
      keepD.push(el ? el.checked : false);
    }

    // apply zeroing
    var recA = keepA ? a[levels].slice() : zeros(a[levels].length);
    var recD = [];
    for (var k = 0; k < levels; k++) {
      recD.push(keepD[k] ? d[k].slice() : zeros(d[k].length));
    }

    // reconstruct step by step, collecting intermediates
    var steps = []; // each: { label, data, len }
    steps.push({ label: "a" + levels + (keepA ? "" : " (zeroed)"), data: recA });

    var current = recA;
    for (var k = levels - 1; k >= 0; k--) {
      var reconstructed = lib.haarReconstruct(current, recD[k]);
      var dLabel = "d" + (k + 1) + (keepD[k] ? "" : " (zeroed)");
      steps.push({ label: dLabel, data: recD[k] });
      steps.push({ label: "a" + k + " (reconstructed)", data: reconstructed });
      current = reconstructed;
    }

    // final step: overlay reconstruction vs original
    var finalRecon = current;

    // build subplots
    // Layout: one row per step, plus a final row for comparison
    var nRows = steps.length + 1;
    var traces = [];
    var subplotGrid = [];

    for (var r = 0; r < steps.length; r++) {
      var axIdx = r === 0 ? "" : (r + 1).toString();
      var xax = "x" + axIdx;
      var yax = "y" + axIdx;

      // generate t-array for this data length
      var dataLen = steps[r].data.length;
      var tArr = lib.linspace(0, sig.tMax, dataLen);

      var color = steps[r].label.indexOf("(zeroed)") >= 0 ? "#ccc" :
                  steps[r].label.indexOf("d") === 0 ? "#870000" : "#2c5f8a";

      traces.push({
        x: tArr, y: steps[r].data,
        name: steps[r].label,
        mode: dataLen < sig.N ? "lines+markers" : "lines",
        line: { color: color, width: dataLen < sig.N ? 0.5 : 1.5 },
        marker: { color: color, size: 3 },
        xaxis: xax, yaxis: yax,
      });

      subplotGrid.push([xax + yax]);
    }

    // final comparison row: original vs reconstruction
    // The last step's data IS the reconstruction — plot it against the original.
    // Trim or pad reconstruction to match original length.
    var reconLen = finalRecon.length;
    var origLen = sig.h.length;
    var plotLen = Math.min(reconLen, origLen);
    var reconTrimmed = finalRecon.slice(0, plotLen);
    var origTrimmed = sig.h.slice(0, plotLen);
    var tTrimmed = sig.t.slice(0, plotLen);

    var lastAxIdx = (steps.length + 1).toString();
    var xaxLast = "x" + lastAxIdx;
    var yaxLast = "y" + lastAxIdx;

    traces.push({
      x: tTrimmed, y: origTrimmed,
      name: "original h(t)",
      line: { color: "#870000", width: 1.5 },
      xaxis: xaxLast, yaxis: yaxLast,
    });
    traces.push({
      x: tTrimmed, y: reconTrimmed,
      name: "reconstruction",
      line: { color: "#2c5f8a", width: 2 },
      xaxis: xaxLast, yaxis: yaxLast,
    });
    subplotGrid.push([xaxLast + yaxLast]);

    // layout
    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 30, b: 50, l: 60, r: 120 },
      hovermode: "x unified",
      showlegend: true,
      legend: { x: 1.02, xanchor: "left", y: 1, font: { size: 10 } },
      grid: { rows: nRows, columns: 1, subplots: subplotGrid, roworder: "top to bottom" },
    };

    // axis configs
    for (var r = 0; r < nRows; r++) {
      var axIdx = r === 0 ? "" : (r + 1).toString();
      var xKey = "xaxis" + axIdx;
      var yKey = "yaxis" + axIdx;
      layout[xKey] = { title: r === nRows - 1 ? "t" : "", zeroline: true, range: [0, sig.tMax] };

      if (r < steps.length) {
        layout[yKey] = { title: steps[r].label.split(" ")[0], zeroline: true };
      } else {
        layout[yKey] = { title: "orig vs recon", zeroline: true };
      }
    }

    var height = 80 + nRows * 100;
    document.getElementById(plotDiv).style.height = height + "px";
    Plotly.react(plotDiv, traces, layout, { responsive: true });

    // update memory/coefficient count
    var memEl = document.getElementById("q7a-memory");
    if (memEl) {
      var kept = 0;
      var parts = [];

      if (keepA) {
        parts.push("a" + levels + ": " + a[levels].length);
        kept += a[levels].length;
      }
      for (var k = 0; k < levels; k++) {
        if (keepD[k]) {
          parts.push("d" + (k + 1) + ": " + d[k].length);
          kept += d[k].length;
        }
      }

      var totalAll = 0;
      totalAll += a[levels].length;
      for (var k = 0; k < levels; k++) totalAll += d[k].length;

      var html = "<table style='font-size:0.85rem; border-collapse:collapse;'>";
      html += "<tr><th style='text-align:left; padding:0.2rem 0.8rem;'>Component</th><th style='padding:0.2rem 0.8rem;'>Coefficients</th></tr>";

      for (var k = 0; k < levels; k++) {
        var checked = keepD[k];
        html += "<tr style='color:" + (checked ? "#870000" : "#ccc") + "'>";
        html += "<td style='padding:0.1rem 0.8rem;'>d" + (k + 1) + "</td>";
        html += "<td style='padding:0.1rem 0.8rem; text-align:right;'>" + d[k].length + "</td></tr>";
      }
      html += "<tr style='color:" + (keepA ? "#2c5f8a" : "#ccc") + "'>";
      html += "<td style='padding:0.1rem 0.8rem;'>a" + levels + "</td>";
      html += "<td style='padding:0.1rem 0.8rem; text-align:right;'>" + a[levels].length + "</td></tr>";

      html += "<tr style='border-top:1px solid #999;'><td style='padding:0.2rem 0.8rem;'><strong>Kept</strong></td>";
      html += "<td style='padding:0.2rem 0.8rem; text-align:right;'><strong>" + kept + "</strong></td></tr>";
      html += "<tr><td style='padding:0.1rem 0.8rem;'>All coefficients</td>";
      html += "<td style='padding:0.1rem 0.8rem; text-align:right;'>" + totalAll + "</td></tr>";
      html += "<tr><td style='padding:0.1rem 0.8rem;'>Original signal</td>";
      html += "<td style='padding:0.1rem 0.8rem; text-align:right;'>" + sig.N + "</td></tr>";
      html += "<tr><td style='padding:0.1rem 0.8rem;'>Compression ratio</td>";
      html += "<td style='padding:0.1rem 0.8rem; text-align:right;'>" + (kept > 0 ? (sig.N / kept).toFixed(1) + "×" : "∞") + "</td></tr>";
      html += "</table>";
      memEl.innerHTML = html;
    }
  }

  function buildCheckboxes() {
    var levels = getLevels();
    var container = document.getElementById("q7a-checkboxes");
    container.innerHTML = "";

    // a_L checkbox
    var aLabel = document.createElement("label");
    var aCb = document.createElement("input");
    aCb.type = "checkbox";
    aCb.id = "q7a-keep-a";
    aCb.checked = true;
    aCb.addEventListener("change", update);
    aLabel.appendChild(aCb);
    aLabel.appendChild(document.createTextNode(" keep a" + levels));
    container.appendChild(aLabel);

    // d_k checkboxes
    for (var k = 0; k < levels; k++) {
      var label = document.createElement("label");
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = "q7a-keep-d" + (k + 1);
      cb.checked = false;
      cb.addEventListener("change", update);
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" keep d" + (k + 1)));
      container.appendChild(label);
    }
  }

  function init() {
    buildCheckboxes();

    var ids = [
      "q1a-window-type", "q1a-sigma", "q1a-fermi-width", "q1a-fermi-edge",
      "q1a-center", "q1a-omega1", "q1a-omega2", "q1a-amp1", "q1a-amp2",
      "q1a-tmax", "q1a-N",
    ];
    var fLenEl = document.getElementById("q5a-filter-length");
    if (fLenEl) ids.push("q5a-filter-length");
    var levelsEl = document.getElementById("q6a-levels");
    if (levelsEl) {
      levelsEl.addEventListener("input", function () { buildCheckboxes(); update(); });
      levelsEl.addEventListener("change", function () { buildCheckboxes(); update(); });
    }

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
