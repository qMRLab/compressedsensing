/**
 * threshold_2d.js — Wavelet thresholding and compression.
 *
 * Full pipeline:
 *   1. Multi-level 2D Haar decomposition
 *   2. Threshold: zero out coefficients with |value| < threshold
 *   3. Count non-zero coefficients → compute storage cost
 *   4. Reconstruct from thresholded coefficients
 *   5. Show: original, thresholded coefficient map, reconstruction, difference
 */
(function () {
  "use strict";

  var s2 = 1 / Math.sqrt(2);

  function decompose2d(pixels) {
    var H = pixels.length;
    var W = pixels[0].length;
    var halfH = Math.floor(H / 2);
    var halfW = Math.floor(W / 2);
    var Lrows = [], Hrows = [];
    for (var r = 0; r < H; r++) {
      var lr = new Array(halfW), hr = new Array(halfW);
      for (var c = 0; c < halfW; c++) {
        lr[c] = (pixels[r][2*c] + pixels[r][2*c+1]) * s2;
        hr[c] = (pixels[r][2*c] - pixels[r][2*c+1]) * s2;
      }
      Lrows.push(lr); Hrows.push(hr);
    }
    var LL = [], LH = [], HL = [], HH = [];
    for (var r = 0; r < halfH; r++) {
      var llr = new Array(halfW), lhr = new Array(halfW);
      var hlr = new Array(halfW), hhr = new Array(halfW);
      for (var c = 0; c < halfW; c++) {
        llr[c] = (Lrows[2*r][c] + Lrows[2*r+1][c]) * s2;
        lhr[c] = (Lrows[2*r][c] - Lrows[2*r+1][c]) * s2;
        hlr[c] = (Hrows[2*r][c] + Hrows[2*r+1][c]) * s2;
        hhr[c] = (Hrows[2*r][c] - Hrows[2*r+1][c]) * s2;
      }
      LL.push(llr); LH.push(lhr); HL.push(hlr); HH.push(hhr);
    }
    return { LL: LL, LH: LH, HL: HL, HH: HH };
  }

  function reconstruct2d(sub) {
    var halfH = sub.LL.length, halfW = sub.LL[0].length;
    var H = halfH * 2, W = halfW * 2;
    var Lrows = [], Hrows = [];
    for (var r = 0; r < H; r++) {
      Lrows.push(new Array(halfW)); Hrows.push(new Array(halfW));
    }
    for (var r = 0; r < halfH; r++) {
      for (var c = 0; c < halfW; c++) {
        Lrows[2*r][c]   = (sub.LL[r][c] + sub.LH[r][c]) * s2;
        Lrows[2*r+1][c] = (sub.LL[r][c] - sub.LH[r][c]) * s2;
        Hrows[2*r][c]   = (sub.HL[r][c] + sub.HH[r][c]) * s2;
        Hrows[2*r+1][c] = (sub.HL[r][c] - sub.HH[r][c]) * s2;
      }
    }
    var out = [];
    for (var r = 0; r < H; r++) {
      var row = new Array(W);
      for (var c = 0; c < halfW; c++) {
        row[2*c]   = (Lrows[r][c] + Hrows[r][c]) * s2;
        row[2*c+1] = (Lrows[r][c] - Hrows[r][c]) * s2;
      }
      out.push(row);
    }
    return out;
  }

  // Multi-level decompose: returns array of { LH, HL, HH } per level + final LL
  function decomposeMulti(pixels, levels) {
    var details = [];
    var current = pixels;
    for (var lev = 0; lev < levels; lev++) {
      var d = decompose2d(current);
      details.push({ LH: d.LH, HL: d.HL, HH: d.HH });
      current = d.LL;
    }
    return { details: details, LL: current };
  }

  // Multi-level reconstruct from thresholded coefficients
  function reconstructMulti(decomp, levels) {
    var current = decomp.LL;
    for (var lev = levels - 1; lev >= 0; lev--) {
      current = reconstruct2d({
        LL: current,
        LH: decomp.details[lev].LH,
        HL: decomp.details[lev].HL,
        HH: decomp.details[lev].HH,
      });
    }
    return current;
  }

  // Deep copy a 2D array
  function copy2d(arr) {
    return arr.map(function(row) { return row.slice(); });
  }

  // Apply threshold: zero coefficients with |value| < thr
  // Returns { decomp (thresholded), nonZeroCount, totalCount }
  function applyThreshold(decomp, levels, thr) {
    var result = {
      details: [],
      LL: copy2d(decomp.LL),
    };
    var nonZero = 0;
    var total = 0;

    // count LL (never threshold LL — always keep it)
    for (var r = 0; r < result.LL.length; r++) {
      for (var c = 0; c < result.LL[0].length; c++) {
        total++;
        nonZero++; // always keep LL
      }
    }

    for (var lev = 0; lev < levels; lev++) {
      var d = decomp.details[lev];
      var tLH = copy2d(d.LH);
      var tHL = copy2d(d.HL);
      var tHH = copy2d(d.HH);
      var h = tLH.length, w = tLH[0].length;

      for (var r = 0; r < h; r++) {
        for (var c = 0; c < w; c++) {
          total += 3;
          if (Math.abs(tLH[r][c]) < thr) tLH[r][c] = 0; else nonZero++;
          if (Math.abs(tHL[r][c]) < thr) tHL[r][c] = 0; else nonZero++;
          if (Math.abs(tHH[r][c]) < thr) tHH[r][c] = 0; else nonZero++;
        }
      }

      result.details.push({ LH: tLH, HL: tHL, HH: tHH });
    }

    return { decomp: result, nonZero: nonZero, total: total };
  }

  // Build tiled image from decomposition (for visualization)
  function buildTiled(decomp, levels, origH, origW) {
    var canvas = [];
    for (var r = 0; r < origH; r++) {
      canvas.push(new Array(origW));
      for (var c = 0; c < origW; c++) canvas[r][c] = 0;
    }

    // place LL in top-left
    var ll = decomp.LL;
    for (var r = 0; r < ll.length; r++) {
      for (var c = 0; c < ll[0].length; c++) {
        canvas[r][c] = ll[r][c];
      }
    }

    // place details
    var curH = origH, curW = origW;
    for (var lev = 0; lev < levels; lev++) {
      var hh = Math.floor(curH / 2);
      var hw = Math.floor(curW / 2);
      var d = decomp.details[lev];
      for (var r = 0; r < hh; r++) {
        for (var c = 0; c < hw; c++) {
          canvas[r][c + hw] = d.HL[r][c];
          canvas[r + hh][c] = d.LH[r][c];
          canvas[r + hh][c + hw] = d.HH[r][c];
        }
      }
      curH = hh;
      curW = hw;
    }
    return canvas;
  }

  function update() {
    var img = window.ImageData2D;
    if (!img) return;

    var H = img.height;
    var W = img.width;
    var levels = parseInt(document.getElementById("cascade2d-levels").value, 10) || 3;
    var thr = parseFloat(document.getElementById("thr2d-threshold").value);

    // decompose
    var decomp = decomposeMulti(img.pixels, levels);

    // find max coefficient magnitude for slider scaling
    var maxCoeff = 0;
    for (var lev = 0; lev < levels; lev++) {
      var subs = [decomp.details[lev].LH, decomp.details[lev].HL, decomp.details[lev].HH];
      for (var s = 0; s < 3; s++) {
        for (var r = 0; r < subs[s].length; r++) {
          for (var c = 0; c < subs[s][0].length; c++) {
            var a = Math.abs(subs[s][r][c]);
            if (a > maxCoeff) maxCoeff = a;
          }
        }
      }
    }

    // update slider max
    var thrSlider = document.getElementById("thr2d-threshold");
    thrSlider.max = Math.ceil(maxCoeff);
    if (thr > maxCoeff) { thr = maxCoeff; thrSlider.value = thr; }

    // threshold
    var result = applyThreshold(decomp, levels, thr);
    var threshDecomp = result.decomp;
    var nonZero = result.nonZero;
    var total = result.total;
    var zeroed = total - nonZero;
    var pctKept = (100 * nonZero / total).toFixed(1);

    // storage costs (coordinate list: row uint16 + col uint16 + value float32 = 8 bytes per nonzero)
    var origBytes = W * H * 2; // uint16
    var sparseBytes = nonZero * 8; // (row, col, value) per nonzero
    var origKB = (origBytes / 1024).toFixed(1);
    var sparseKB = (sparseBytes / 1024).toFixed(1);
    var ratio = sparseBytes > 0 ? (origBytes / sparseBytes).toFixed(1) : "\u221e";

    // reconstruct
    var recon = reconstructMulti(threshDecomp, levels);

    // compute error
    var mse = 0;
    for (var r = 0; r < H; r++) {
      for (var c = 0; c < W; c++) {
        var diff = img.pixels[r][c] - recon[r][c];
        mse += diff * diff;
      }
    }
    mse /= (H * W);
    var psnr = mse > 0 ? (10 * Math.log10(65535 * 65535 / mse)).toFixed(1) : "\u221e";

    // build tiled views
    var tiledOrig = buildTiled(decomp, levels, H, W);
    var tiledThresh = buildTiled(threshDecomp, levels, H, W);

    var traces = [
      // row 1: original image
      { z: img.pixels, type: "heatmap", colorscale: "Greys", reversescale: false,
        showscale: false, zmin: 0, zmax: 65535,
        xaxis: "x", yaxis: "y" },
      // row 1: thresholded coefficients (tiled)
      { z: tiledThresh, type: "heatmap", colorscale: "RdGy", reversescale: true,
        showscale: false, zmid: 0,
        xaxis: "x2", yaxis: "y2" },
      // row 1: reconstruction
      { z: recon, type: "heatmap", colorscale: "Greys", reversescale: false,
        showscale: false, zmin: 0, zmax: 65535,
        xaxis: "x3", yaxis: "y3" },
    ];

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 50, b: 30, l: 20, r: 20 },
      showlegend: false,
      grid: { rows: 1, columns: 3, subplots: [["xy", "x2y2", "x3y3"]] },
      xaxis:  { showticklabels: false },
      yaxis:  { autorange: "reversed", showticklabels: false },
      xaxis2: { showticklabels: false },
      yaxis2: { autorange: "reversed", showticklabels: false },
      xaxis3: { showticklabels: false },
      yaxis3: { autorange: "reversed", showticklabels: false },
      annotations: [
        { text: "Original (" + origKB + " KB)",
          x: 0.5, y: 1.1, xref: "x domain", yref: "y domain",
          showarrow: false, font: { size: 11 } },
        { text: "Thresholded coefficients",
          x: 0.5, y: 1.1, xref: "x2 domain", yref: "y2 domain",
          showarrow: false, font: { size: 11, color: "#870000" } },
        { text: "Reconstruction (" + sparseKB + " KB, " + ratio + "\u00d7)",
          x: 0.5, y: 1.1, xref: "x3 domain", yref: "y3 domain",
          showarrow: false, font: { size: 11, color: "#2c5f8a" } },
      ],
    };

    Plotly.react("thr2d-plot", traces, layout, { responsive: true });

    // info
    var infoEl = document.getElementById("thr2d-info");
    if (infoEl) {
      var sizeMsg;
      if (sparseBytes <= origBytes) {
        var pctReduction = (100 - 100 * sparseBytes / origBytes).toFixed(1);
        sizeMsg = "<strong>" + sparseKB + " KB</strong> (" +
          "<strong>" + pctReduction + "% smaller</strong> than " + origKB + " KB original)";
      } else {
        var pctBigger = (100 * sparseBytes / origBytes - 100).toFixed(1);
        sizeMsg = "<strong>" + sparseKB + " KB</strong> (" +
          pctBigger + "% <em>larger</em> than " + origKB + " KB original \u2014 need more thresholding)";
      }

      infoEl.innerHTML =
        "Threshold: <strong>" + thr.toFixed(0) + "</strong> &mdash; " +
        "Non-zero: <strong>" + nonZero + "</strong> / " + total +
        " (" + pctKept + "% kept) &mdash; " +
        "Storage: " + sizeMsg + " &mdash; " +
        "PSNR: <strong>" + psnr + " dB</strong>";
    }
  }

  function init() {
    document.getElementById("thr2d-threshold").addEventListener("input", update);

    // also update when levels change
    var levEl = document.getElementById("cascade2d-levels");
    if (levEl) {
      levEl.addEventListener("input", update);
      levEl.addEventListener("change", update);
    }

    if (!window.ImageData2D) {
      var poll = setInterval(function () {
        if (window.ImageData2D) { clearInterval(poll); update(); }
      }, 300);
    } else {
      update();
    }

    window.addEventListener("imagedata2d-changed", update);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
