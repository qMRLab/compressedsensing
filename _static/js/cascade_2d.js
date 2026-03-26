/**
 * cascade_2d.js — Multi-level 2D Haar wavelet decomposition.
 *
 * Shows the classic wavelet layout: LL gets recursively decomposed.
 * One slider controls number of decomposition levels.
 * Displays as the standard tiled layout:
 *   Level 1: [LL1 | HL1]   (top-left quadrant gets subdivided at level 2)
 *            [LH1 | HH1]
 */
(function () {
  "use strict";

  var s2 = 1 / Math.sqrt(2);

  // 1-level 2D Haar decompose: returns { LL, LH, HL, HH }
  function decompose2d(pixels) {
    var H = pixels.length;
    var W = pixels[0].length;
    var halfH = Math.floor(H / 2);
    var halfW = Math.floor(W / 2);

    // pass 1: filter rows
    var Lrows = [];
    var Hrows = [];
    for (var r = 0; r < H; r++) {
      var lr = new Array(halfW);
      var hr = new Array(halfW);
      for (var c = 0; c < halfW; c++) {
        lr[c] = (pixels[r][2*c] + pixels[r][2*c+1]) * s2;
        hr[c] = (pixels[r][2*c] - pixels[r][2*c+1]) * s2;
      }
      Lrows.push(lr);
      Hrows.push(hr);
    }

    // pass 2: filter columns
    var LL = [], LH = [], HL = [], HH = [];
    for (var r = 0; r < halfH; r++) {
      var llr = new Array(halfW);
      var lhr = new Array(halfW);
      var hlr = new Array(halfW);
      var hhr = new Array(halfW);
      for (var c = 0; c < halfW; c++) {
        llr[c] = (Lrows[2*r][c] + Lrows[2*r+1][c]) * s2;
        lhr[c] = (Lrows[2*r][c] - Lrows[2*r+1][c]) * s2;
        hlr[c] = (Hrows[2*r][c] + Hrows[2*r+1][c]) * s2;
        hhr[c] = (Hrows[2*r][c] - Hrows[2*r+1][c]) * s2;
      }
      LL.push(llr);
      LH.push(lhr);
      HL.push(hlr);
      HH.push(hhr);
    }

    return { LL: LL, LH: LH, HL: HL, HH: HH };
  }

  // 1-level 2D Haar reconstruct from { LL, LH, HL, HH }
  function reconstruct2d(sub) {
    var halfH = sub.LL.length;
    var halfW = sub.LL[0].length;
    var H = halfH * 2;
    var W = halfW * 2;

    // inverse pass 2: columns
    var Lrows = [];
    var Hrows = [];
    for (var r = 0; r < H; r++) {
      Lrows.push(new Array(halfW));
      Hrows.push(new Array(halfW));
    }
    for (var r = 0; r < halfH; r++) {
      for (var c = 0; c < halfW; c++) {
        Lrows[2*r][c]   = (sub.LL[r][c] + sub.LH[r][c]) * s2;
        Lrows[2*r+1][c] = (sub.LL[r][c] - sub.LH[r][c]) * s2;
        Hrows[2*r][c]   = (sub.HL[r][c] + sub.HH[r][c]) * s2;
        Hrows[2*r+1][c] = (sub.HL[r][c] - sub.HH[r][c]) * s2;
      }
    }

    // inverse pass 1: rows
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

  // Multi-level reconstruct from only LL (zero all details)
  function reconstructFromLL(llPixels, levels, origPixels) {
    // Decompose fully to get sizes at each level
    var decomps = [];
    var current = origPixels;
    for (var lev = 0; lev < levels; lev++) {
      decomps.push(decompose2d(current));
      current = decomps[lev].LL;
    }

    // Reconstruct from LL upward, zeroing all detail at each level
    current = llPixels;
    for (var lev = levels - 1; lev >= 0; lev--) {
      var hh = current.length;
      var hw = current[0].length;
      var zeros = [];
      for (var r = 0; r < hh; r++) {
        var zrow = new Array(hw);
        for (var c = 0; c < hw; c++) zrow[c] = 0;
        zeros.push(zrow);
      }
      current = reconstruct2d({ LL: current, LH: zeros, HL: zeros, HH: zeros });
    }
    return current;
  }

  // Assemble the classic tiled layout into a single 2D array
  function buildTiledImage(pixels, levels) {
    var H = pixels.length;
    var W = pixels[0].length;

    // start with a copy of the original sized canvas
    var canvas = [];
    for (var r = 0; r < H; r++) {
      canvas.push(new Array(W));
      for (var c = 0; c < W; c++) canvas[r][c] = pixels[r][c];
    }

    var current = pixels;
    for (var lev = 0; lev < levels; lev++) {
      var d = decompose2d(current);
      var hh = Math.floor(current.length / 2);
      var hw = Math.floor(current[0].length / 2);

      // place into canvas:
      // top-left: LL (will be overwritten next level)
      // top-right: HL
      // bottom-left: LH
      // bottom-right: HH
      var rOff = 0, cOff = 0;
      // compute offset: for level 0 it's (0,0), for level 1 it's still (0,0) since we subdivide top-left
      // Actually the offset is always (0,0) because LL always occupies top-left
      for (var r = 0; r < hh; r++) {
        for (var c = 0; c < hw; c++) {
          canvas[r][c] = d.LL[r][c];             // top-left
          canvas[r][c + hw] = d.HL[r][c];        // top-right
          canvas[r + hh][c] = d.LH[r][c];        // bottom-left
          canvas[r + hh][c + hw] = d.HH[r][c];   // bottom-right
        }
      }

      current = d.LL;
    }

    return canvas;
  }

  function update() {
    var img = window.ImageData2D;
    if (!img) return;

    var levels = parseInt(document.getElementById("cascade2d-levels").value, 10) || 1;
    var tiled = buildTiledImage(img.pixels, levels);

    var traces = [{
      z: tiled,
      type: "heatmap",
      colorscale: "RdGy",
      reversescale: true,
      showscale: true,
      zmid: 0,
      colorbar: { title: "value", thickness: 12 },
    }];

    // draw grid lines at subdivision boundaries
    var shapes = [];
    var H = img.height;
    var W = img.width;
    for (var lev = 0; lev < levels; lev++) {
      var hh = Math.floor(H / Math.pow(2, lev + 1));
      var hw = Math.floor(W / Math.pow(2, lev + 1));
      // horizontal line
      shapes.push({
        type: "line",
        x0: -0.5, x1: Math.floor(W / Math.pow(2, lev)) - 0.5,
        y0: hh - 0.5, y1: hh - 0.5,
        line: { color: "rgba(255,0,0,0.6)", width: 1 },
      });
      // vertical line
      shapes.push({
        type: "line",
        x0: hw - 0.5, x1: hw - 0.5,
        y0: -0.5, y1: Math.floor(H / Math.pow(2, lev)) - 0.5,
        line: { color: "rgba(255,0,0,0.6)", width: 1 },
      });
    }

    // annotations labeling each quadrant at finest decomposition level
    var annotations = [];
    var lastH = Math.floor(H / Math.pow(2, levels));
    var lastW = Math.floor(W / Math.pow(2, levels));
    annotations.push({ text: "LL" + levels, x: lastW/2, y: lastH/2, showarrow: false,
      font: { size: 11, color: "red" } });
    for (var lev = levels; lev >= 1; lev--) {
      var hh = Math.floor(H / Math.pow(2, lev));
      var hw = Math.floor(W / Math.pow(2, lev));
      annotations.push({ text: "HL" + lev, x: hw + hw/2, y: hh/2 - hh, showarrow: false,
        font: { size: Math.min(13, 9 + lev), color: "red" } });
      annotations.push({ text: "LH" + lev, x: hw/2, y: hh + hh/2 - hh, showarrow: false,
        font: { size: Math.min(13, 9 + lev), color: "red" } });
      annotations.push({ text: "HH" + lev, x: hw + hw/2, y: hh + hh/2 - hh, showarrow: false,
        font: { size: Math.min(13, 9 + lev), color: "red" } });
    }

    var layout = {
      font: { family: "STIX Two Text" },
      margin: { t: 30, b: 30, l: 30, r: 20 },
      yaxis: { autorange: "reversed", showticklabels: false },
      xaxis: { showticklabels: false },
      shapes: shapes,
    };

    // info
    var infoEl = document.getElementById("cascade2d-info");
    if (infoEl) {
      var totalCoeffs = H * W;
      var llSize = Math.floor(H / Math.pow(2, levels)) * Math.floor(W / Math.pow(2, levels));
      infoEl.textContent = levels + " level" + (levels > 1 ? "s" : "") +
        " \u2014 LL" + levels + " is " + Math.floor(W / Math.pow(2, levels)) +
        "\u00d7" + Math.floor(H / Math.pow(2, levels)) +
        " (" + llSize + " coefficients out of " + totalCoeffs + " total)";
    }

    Plotly.react("cascade2d-plot", traces, layout, { responsive: true });

    // ── Side-by-side: Original vs LL vs Reconstruction ──
    var llPixels = img.pixels;
    for (var lev = 0; lev < levels; lev++) {
      llPixels = decompose2d(llPixels).LL;
    }

    var reconPixels = reconstructFromLL(llPixels, levels, img.pixels);

    var llW = Math.floor(W / Math.pow(2, levels));
    var llH = Math.floor(H / Math.pow(2, levels));

    var compTraces = [
      {
        z: img.pixels, type: "heatmap", colorscale: "Greys", reversescale: false,
        showscale: false, zmin: 0, zmax: 65535,
        xaxis: "x", yaxis: "y",
      },
      {
        z: llPixels, type: "heatmap", colorscale: "Greys", reversescale: false,
        showscale: false, zmid: 0,
        xaxis: "x2", yaxis: "y2",
      },
      {
        z: reconPixels, type: "heatmap", colorscale: "Greys", reversescale: false,
        showscale: false, zmin: 0, zmax: 65535,
        xaxis: "x3", yaxis: "y3",
      },
    ];

    // KB calculations (uint16 = 2 bytes per pixel)
    var origKB = (W * H * 2 / 1024).toFixed(1);
    var llKB = (llW * llH * 2 / 1024).toFixed(1);
    var ratio = Math.pow(4, levels);

    var compLayout = {
      font: { family: "STIX Two Text" },
      margin: { t: 50, b: 30, l: 20, r: 20 },
      showlegend: false,
      xaxis:  { domain: [0, 0.31], showticklabels: false },
      yaxis:  { autorange: "reversed", showticklabels: false },
      xaxis2: { domain: [0.345, 0.655], showticklabels: false },
      yaxis2: { autorange: "reversed", showticklabels: false, anchor: "x2" },
      xaxis3: { domain: [0.69, 1], showticklabels: false },
      yaxis3: { autorange: "reversed", showticklabels: false, anchor: "x3" },
      annotations: [
        { text: "Original (" + W + "\u00d7" + H + ", " + origKB + " KB)",
          x: 0.5, y: 1.1, xref: "x domain", yref: "y domain",
          showarrow: false, font: { size: 11 } },
        { text: "LL" + levels + " (" + llW + "\u00d7" + llH + ", " + llKB + " KB, " + ratio + "\u00d7 smaller)",
          x: 0.5, y: 1.1, xref: "x2 domain", yref: "y2 domain",
          showarrow: false, font: { size: 11, color: "#2c5f8a" } },
        { text: "Reconstructed from LL" + levels + " only",
          x: 0.5, y: 1.1, xref: "x3 domain", yref: "y3 domain",
          showarrow: false, font: { size: 11, color: "#870000" } },
      ],
    };

    Plotly.react("cascade2d-compare-plot", compTraces, compLayout, { responsive: true });
  }

  function init() {
    document.getElementById("cascade2d-levels").addEventListener("input", update);
    document.getElementById("cascade2d-levels").addEventListener("change", update);

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
