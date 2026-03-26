/**
 * decompose_2d.js — Two-pass separable 2D Haar decomposition visualizer.
 *
 * Pass 1 (rows): slide L/H filter across each row → L-rows and H-rows intermediates
 * Pass 2 (columns): slide L/H filter down each column → LL, LH, HL, HH
 *
 * Each pass has a "which line" slider and a "filter position" slider.
 */
(function () {
  "use strict";

  var s2 = 1 / Math.sqrt(2);
  var LO = [s2, s2];
  var HI = [s2, -s2];

  function conv1dSameRow(row, filter) {
    // convolveSame for a 1D row with a length-2 filter
    var n = row.length;
    var out = new Array(n);
    for (var i = 0; i < n; i++) {
      var sum = 0;
      for (var j = 0; j < filter.length; j++) {
        var idx = i - Math.floor(filter.length / 2) + j;
        if (idx >= 0 && idx < n) sum += row[idx] * filter[j];
        // zero-pad outside
      }
      out[i] = sum;
    }
    return out;
  }

  function downsample(arr) {
    var out = [];
    for (var i = 0; i < arr.length; i += 2) out.push(arr[i]);
    return out;
  }

  // ── PASS 1: Row filtering ──
  function updatePass1() {
    var img = window.ImageData2D;
    if (!img) return;

    var H = img.height;
    var W = img.width;

    var rowSlider = document.getElementById("dec2d-p1-row");
    var posSlider = document.getElementById("dec2d-p1-pos");
    rowSlider.max = H - 1;
    posSlider.max = W - 1;

    var curRow = parseInt(rowSlider.value, 10);
    var curPos = parseInt(posSlider.value, 10);
    if (curRow >= H) { curRow = H - 1; rowSlider.value = curRow; }
    if (curPos >= W) { curPos = W - 1; posSlider.value = curPos; }

    // compute L-rows and H-rows for all rows up to curRow (full row),
    // and partial for curRow up to curPos
    var Lrows = [];
    var Hrows = [];
    for (var r = 0; r < H; r++) {
      if (r < curRow) {
        Lrows.push(downsample(conv1dSameRow(img.pixels[r], LO)));
        Hrows.push(downsample(conv1dSameRow(img.pixels[r], HI)));
      } else if (r === curRow) {
        var lFull = conv1dSameRow(img.pixels[r], LO);
        var hFull = conv1dSameRow(img.pixels[r], HI);
        // partial: only show downsampled values up to curPos
        var lPart = [];
        var hPart = [];
        for (var c = 0; c < W; c += 2) {
          if (c <= curPos) {
            lPart.push(lFull[c]);
            hPart.push(hFull[c]);
          } else {
            lPart.push(null);
            hPart.push(null);
          }
        }
        Lrows.push(lPart);
        Hrows.push(hPart);
      } else {
        Lrows.push(new Array(Math.floor(W / 2)).fill(null));
        Hrows.push(new Array(Math.floor(W / 2)).fill(null));
      }
    }

    // highlight: the filter covers [curPos-1, curPos] on row curRow (length-2 filter)
    var fStart = curPos;
    var fEnd = curPos + 1;

    var traces = [
      // original with highlight line
      {
        z: img.pixels, type: "heatmap", colorscale: "Greys", reversescale: false,
        showscale: false, zmin: 0, zmax: 65535,
        xaxis: "x", yaxis: "y",
      },
      // horizontal line at curRow
      {
        x: [0, W - 1], y: [curRow, curRow],
        mode: "lines", line: { color: "rgba(255,0,0,0.4)", width: 1 },
        showlegend: false, xaxis: "x", yaxis: "y",
      },
      // filter position marker
      {
        x: [fStart - 0.5, fEnd + 0.5, fEnd + 0.5, fStart - 0.5, fStart - 0.5],
        y: [curRow - 0.5, curRow - 0.5, curRow + 0.5, curRow + 0.5, curRow - 0.5],
        mode: "lines", line: { color: "red", width: 2 },
        showlegend: false, xaxis: "x", yaxis: "y",
      },
      // L-rows intermediate
      {
        z: Lrows, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x2", yaxis: "y2",
      },
      // H-rows intermediate (show colorbar on this one)
      {
        z: Hrows, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: true, zmid: 0,
        colorbar: { title: "value", len: 0.8, thickness: 12, x: 1.02 },
        xaxis: "x3", yaxis: "y3",
      },
    ];

    // layout: equal-width columns, all same visual size
    var gap = 0.03;
    var colW = (1 - 2 * gap) / 3;

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 40, b: 30, l: 30, r: 60 },
      showlegend: false,
      xaxis:  { domain: [0, colW], showticklabels: false },
      yaxis:  { autorange: "reversed", showticklabels: false },
      xaxis2: { domain: [colW + gap, 2 * colW + gap], showticklabels: false },
      yaxis2: { autorange: "reversed", showticklabels: false, anchor: "x2" },
      xaxis3: { domain: [2 * colW + 2 * gap, 1], showticklabels: false },
      yaxis3: { autorange: "reversed", showticklabels: false, anchor: "x3" },
      annotations: [
        { text: "Original", x: 0.5, y: 1.07, xref: "x domain", yref: "y domain",
          showarrow: false, font: { size: 12 } },
        { text: "L\u2192 (rows)", x: 0.5, y: 1.07, xref: "x2 domain", yref: "y2 domain",
          showarrow: false, font: { size: 12, color: "#2c5f8a" } },
        { text: "H\u2192 (rows)", x: 0.5, y: 1.07, xref: "x3 domain", yref: "y3 domain",
          showarrow: false, font: { size: 12, color: "#870000" } },
      ],
    };

    var infoEl = document.getElementById("dec2d-p1-info");
    if (infoEl) {
      infoEl.textContent = "Row " + curRow + " / " + (H - 1) + ", position " + curPos + " / " + (W - 1);
    }

    Plotly.react("dec2d-p1-plot", traces, layout, { responsive: true });

    // store intermediates for pass 2
    // compute full intermediates (all rows, all columns)
    var fullL = [];
    var fullH = [];
    for (var r = 0; r < H; r++) {
      fullL.push(downsample(conv1dSameRow(img.pixels[r], LO)));
      fullH.push(downsample(conv1dSameRow(img.pixels[r], HI)));
    }
    window.DecompIntermediate = { Lrows: fullL, Hrows: fullH, height: H, width: Math.floor(W / 2) };
  }

  // ── PASS 2: Column filtering ──
  function updatePass2() {
    var inter = window.DecompIntermediate;
    if (!inter) return;

    var H = inter.height;
    var W = inter.width;
    var outH = Math.floor(H / 2);

    var colSlider = document.getElementById("dec2d-p2-col");
    var posSlider = document.getElementById("dec2d-p2-pos");
    colSlider.max = W - 1;
    posSlider.max = H - 1;

    var curCol = parseInt(colSlider.value, 10);
    var curPos = parseInt(posSlider.value, 10);
    if (curCol >= W) { curCol = W - 1; colSlider.value = curCol; }
    if (curPos >= H) { curPos = H - 1; posSlider.value = curPos; }

    // extract column from L-rows and H-rows, filter vertically, downsample
    function filterCol(matrix, filter, col, maxPos) {
      var out = new Array(outH);
      for (var r = 0; r < outH; r++) {
        var srcR = r * 2;
        if (srcR <= maxPos && srcR + 1 < H) {
          out[r] = filter[0] * matrix[srcR][col] + filter[1] * matrix[srcR + 1][col];
        } else {
          out[r] = null;
        }
      }
      return out;
    }

    // build LL, LH, HL, HH
    var LL = [], LH = [], HL = [], HH = [];
    for (var c = 0; c < W; c++) {
      var mp = c < curCol ? H - 1 : (c === curCol ? curPos : -1);
      LL.push(filterCol(inter.Lrows, LO, c, mp));
      LH.push(filterCol(inter.Lrows, HI, c, mp));
      HL.push(filterCol(inter.Hrows, LO, c, mp));
      HH.push(filterCol(inter.Hrows, HI, c, mp));
    }

    // transpose: LL[col][row] → LL[row][col]
    function transpose(arr, rows, cols) {
      var out = [];
      for (var r = 0; r < rows; r++) {
        out.push(new Array(cols));
        for (var c = 0; c < cols; c++) {
          out[r][c] = arr[c][r];
        }
      }
      return out;
    }

    var llT = transpose(LL, outH, W);
    var lhT = transpose(LH, outH, W);
    var hlT = transpose(HL, outH, W);
    var hhT = transpose(HH, outH, W);

    // highlight on L-rows intermediate
    var fStart = curPos;
    var fEnd = curPos + 1;

    var traces = [
      // L-rows with highlight
      {
        z: inter.Lrows, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x", yaxis: "y",
      },
      // vertical line at curCol
      {
        x: [curCol, curCol], y: [0, H - 1],
        mode: "lines", line: { color: "rgba(255,0,0,0.4)", width: 1 },
        showlegend: false, xaxis: "x", yaxis: "y",
      },
      // filter position
      {
        x: [curCol - 0.5, curCol + 0.5, curCol + 0.5, curCol - 0.5, curCol - 0.5],
        y: [fStart - 0.5, fStart - 0.5, fEnd + 0.5, fEnd + 0.5, fStart - 0.5],
        mode: "lines", line: { color: "red", width: 2 },
        showlegend: false, xaxis: "x", yaxis: "y",
      },
      // LL
      { z: llT, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x2", yaxis: "y2" },
      // LH
      { z: lhT, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x3", yaxis: "y3" },
      // H-rows with highlight
      {
        z: inter.Hrows, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x4", yaxis: "y4",
      },
      {
        x: [curCol, curCol], y: [0, H - 1],
        mode: "lines", line: { color: "rgba(255,0,0,0.4)", width: 1 },
        showlegend: false, xaxis: "x4", yaxis: "y4",
      },
      {
        x: [curCol - 0.5, curCol + 0.5, curCol + 0.5, curCol - 0.5, curCol - 0.5],
        y: [fStart - 0.5, fStart - 0.5, fEnd + 0.5, fEnd + 0.5, fStart - 0.5],
        mode: "lines", line: { color: "red", width: 2 },
        showlegend: false, xaxis: "x4", yaxis: "y4",
      },
      // HL
      { z: hlT, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x5", yaxis: "y5" },
      // HH (show colorbar)
      { z: hhT, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: true, zmid: 0,
        colorbar: { title: "value", len: 0.4, thickness: 12, x: 1.02, y: 0.25 },
        xaxis: "x6", yaxis: "y6" },
    ];

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 40, b: 30, l: 30, r: 20 },
      showlegend: false,
      // 2x3 grid, all equal size, no aspect ratio lock
      grid: { rows: 2, columns: 3,
        subplots: [["xy", "x2y2", "x3y3"], ["x4y4", "x5y5", "x6y6"]],
        roworder: "top to bottom" },
      xaxis:  { showticklabels: false },
      yaxis:  { autorange: "reversed", showticklabels: false },
      xaxis2: { showticklabels: false },
      yaxis2: { autorange: "reversed", showticklabels: false },
      xaxis3: { showticklabels: false },
      yaxis3: { autorange: "reversed", showticklabels: false },
      xaxis4: { showticklabels: false },
      yaxis4: { autorange: "reversed", showticklabels: false },
      xaxis5: { showticklabels: false },
      yaxis5: { autorange: "reversed", showticklabels: false },
      xaxis6: { showticklabels: false },
      yaxis6: { autorange: "reversed", showticklabels: false },
      annotations: [
        { text: "L\u2192 (from pass 1)", x: 0.5, y: 1.07, xref: "x domain", yref: "y domain",
          showarrow: false, font: { size: 11, color: "#2c5f8a" } },
        { text: "LL", x: 0.5, y: 1.07, xref: "x2 domain", yref: "y2 domain",
          showarrow: false, font: { size: 12, color: "#2c5f8a" } },
        { text: "LH", x: 0.5, y: 1.07, xref: "x3 domain", yref: "y3 domain",
          showarrow: false, font: { size: 12, color: "#870000" } },
        { text: "H\u2192 (from pass 1)", x: 0.5, y: 1.07, xref: "x4 domain", yref: "y4 domain",
          showarrow: false, font: { size: 11, color: "#870000" } },
        { text: "HL", x: 0.5, y: 1.07, xref: "x5 domain", yref: "y5 domain",
          showarrow: false, font: { size: 12, color: "#870000" } },
        { text: "HH", x: 0.5, y: 1.07, xref: "x6 domain", yref: "y6 domain",
          showarrow: false, font: { size: 12, color: "#870000" } },
      ],
    };

    var infoEl = document.getElementById("dec2d-p2-info");
    if (infoEl) {
      infoEl.textContent = "Column " + curCol + " / " + (W - 1) + ", position " + curPos + " / " + (H - 1);
    }

    Plotly.react("dec2d-p2-plot", traces, layout, { responsive: true });
  }

  function init() {
    // pass 1 sliders
    var p1Row = document.getElementById("dec2d-p1-row");
    var p1Pos = document.getElementById("dec2d-p1-pos");
    p1Row.addEventListener("input", updatePass1);
    p1Pos.addEventListener("input", updatePass1);

    // pass 2 sliders
    var p2Col = document.getElementById("dec2d-p2-col");
    var p2Pos = document.getElementById("dec2d-p2-pos");
    p2Col.addEventListener("input", function () { updatePass2(); });
    p2Pos.addEventListener("input", function () { updatePass2(); });

    // poll for image
    if (!window.ImageData2D) {
      var poll = setInterval(function () {
        if (window.ImageData2D) {
          clearInterval(poll);
          updatePass1();
          updatePass2();
        }
      }, 300);
    } else {
      updatePass1();
      updatePass2();
    }

    // re-run when a new image is loaded
    window.addEventListener("imagedata2d-changed", function () {
      updatePass1();
      updatePass2();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
