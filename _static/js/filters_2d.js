/**
 * filters_2d.js — Plot the Haar low-pass and high-pass filters
 * used for the 2D wavelet decomposition.
 */
(function () {
  "use strict";

  var s2 = 1 / Math.sqrt(2);
  var lo = [s2, s2];
  var hi = [s2, -s2];

  // stem plot: vertical lines from zero + markers at the tips
  var traces = [
    // L stems
    { x: [0, 0], y: [0, lo[0]], mode: "lines", line: { color: "#2c5f8a", width: 3 }, showlegend: false, xaxis: "x", yaxis: "y" },
    { x: [1, 1], y: [0, lo[1]], mode: "lines", line: { color: "#2c5f8a", width: 3 }, showlegend: false, xaxis: "x", yaxis: "y" },
    { x: [0, 1], y: lo, name: "L (low-pass)", mode: "markers", marker: { color: "#2c5f8a", size: 12 }, xaxis: "x", yaxis: "y" },
    // H stems
    { x: [0, 0], y: [0, hi[0]], mode: "lines", line: { color: "#870000", width: 3 }, showlegend: false, xaxis: "x2", yaxis: "y2" },
    { x: [1, 1], y: [0, hi[1]], mode: "lines", line: { color: "#870000", width: 3 }, showlegend: false, xaxis: "x2", yaxis: "y2" },
    { x: [0, 1], y: hi, name: "H (high-pass)", mode: "markers", marker: { color: "#870000", size: 12 }, xaxis: "x2", yaxis: "y2" },
  ];

  var layout = {
    font: { family: "STIX Two Text" },
    margin: { t: 30, b: 40, l: 60, r: 20 },
    showlegend: false,
    grid: { rows: 1, columns: 2, subplots: [["xy", "x2y2"]] },
    xaxis: { title: "n", zeroline: true, dtick: 1, range: [-0.5, 1.5] },
    yaxis: {
      title: "L[n] = [1/√2, 1/√2]",
      zeroline: true,
      range: [-1, 1],
      titlefont: { color: "#2c5f8a" },
    },
    xaxis2: { title: "n", zeroline: true, dtick: 1, range: [-0.5, 1.5] },
    yaxis2: {
      title: "H[n] = [1/√2, −1/√2]",
      zeroline: true,
      range: [-1, 1],
      titlefont: { color: "#870000" },
    },
    annotations: [
      {
        text: "≈ 0.707",
        x: 0.5, y: s2,
        xref: "x", yref: "y",
        showarrow: false,
        font: { size: 11, color: "#2c5f8a" },
        yshift: 14,
      },
      {
        text: "≈ 0.707",
        x: 0, y: s2,
        xref: "x2", yref: "y2",
        showarrow: false,
        font: { size: 11, color: "#870000" },
        yshift: 14,
      },
      {
        text: "≈ −0.707",
        x: 1, y: -s2,
        xref: "x2", yref: "y2",
        showarrow: false,
        font: { size: 11, color: "#870000" },
        yshift: -14,
      },
    ],
  };

  Plotly.react("filters-2d-plot", traces, layout, { responsive: true });

  // ── 2D separable kernels ──
  var kernels = {
    LL: [[ s2*s2,  s2*s2], [ s2*s2,  s2*s2]],
    LH: [[ s2*s2, -s2*s2], [ s2*s2, -s2*s2]],
    HL: [[ s2*s2,  s2*s2], [-s2*s2, -s2*s2]],
    HH: [[ s2*s2, -s2*s2], [-s2*s2,  s2*s2]],
  };

  var names = ["LL", "LH", "HL", "HH"];
  var colors = ["#2c5f8a", "#870000", "#870000", "#870000"];
  var traces2d = [];
  var annotations2d = [];

  for (var k = 0; k < 4; k++) {
    var axIdx = k === 0 ? "" : (k + 1).toString();
    var xax = "x" + axIdx;
    var yax = "y" + axIdx;
    var kern = kernels[names[k]];

    traces2d.push({
      z: kern,
      type: "heatmap",
      colorscale: "RdBu",
      reversescale: true,
      zmid: 0,
      zmin: -0.5,
      zmax: 0.5,
      showscale: false,
      xaxis: xax,
      yaxis: yax,
      text: kern.map(function(row) { return row.map(function(v) { return v.toFixed(3); }); }),
      texttemplate: "%{text}",
      hoverinfo: "z",
    });

    var rowF = k < 2 ? "L" : "H";
    var colF = k % 2 === 0 ? "L" : "H";
    annotations2d.push({
      text: names[k] + " = " + rowF + "\u2297" + colF,
      x: 0.5, y: 1.2,
      xref: xax + " domain", yref: yax + " domain",
      showarrow: false,
      font: { size: 13, color: colors[k], family: "STIX Two Text" },
    });
  }

  var layout2d = {
    font: { family: "STIX Two Text" },
    margin: { t: 40, b: 20, l: 20, r: 20 },
    showlegend: false,
    grid: { rows: 1, columns: 4, subplots: [["xy", "x2y2", "x3y3", "x4y4"]] },
    xaxis:  { showticklabels: false, constrain: "domain", scaleanchor: "y" },
    yaxis:  { autorange: "reversed", showticklabels: false },
    xaxis2: { showticklabels: false, constrain: "domain", scaleanchor: "y2" },
    yaxis2: { autorange: "reversed", showticklabels: false },
    xaxis3: { showticklabels: false, constrain: "domain", scaleanchor: "y3" },
    yaxis3: { autorange: "reversed", showticklabels: false },
    xaxis4: { showticklabels: false, constrain: "domain", scaleanchor: "y4" },
    yaxis4: { autorange: "reversed", showticklabels: false },
    annotations: annotations2d,
  };

  Plotly.react("filters-2d-kernels-plot", traces2d, layout2d, { responsive: true });
})();
