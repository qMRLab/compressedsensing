/**
 * img_loader.js — Load a user-selected image, convert to grayscale,
 * downscale so the largest dimension is ≤ 512, and store as a 2D array
 * of uint16 values on window.ImageData2D.
 *
 * Exposes:
 *   window.ImageData2D = {
 *     pixels: number[][],   // [row][col], values 0–65535
 *     width: number,
 *     height: number,
 *     name: string,         // original filename
 *   }
 */
(function () {
  "use strict";

  var MAX_DIM = 1024;

  var plotDiv = "img-preview-plot";

  function handleFile(file) {
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function (e) {
      var img = new Image();
      img.onload = function () {
        // downscale if needed, ensure even dimensions for wavelet decomposition
        var w = img.width;
        var h = img.height;
        var scale = 1;
        if (Math.max(w, h) > MAX_DIM) {
          scale = MAX_DIM / Math.max(w, h);
        }
        var newW = Math.round(w * scale);
        var newH = Math.round(h * scale);
        newW -= newW % 2;
        newH -= newH % 2;

        // draw to canvas at new size
        var canvas = document.createElement("canvas");
        canvas.width = newW;
        canvas.height = newH;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, newW, newH);

        // extract pixel data and convert to grayscale uint16
        var imageData = ctx.getImageData(0, 0, newW, newH);
        var data = imageData.data; // RGBA flat array
        var pixels = [];
        for (var row = 0; row < newH; row++) {
          var rowArr = new Array(newW);
          for (var col = 0; col < newW; col++) {
            var idx = (row * newW + col) * 4;
            // luminance: 0.299R + 0.587G + 0.114B, scaled to 0–65535
            var gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            rowArr[col] = Math.round(gray * 257); // 0–255 → 0–65535
          }
          pixels.push(rowArr);
        }

        window.ImageData2D = {
          pixels: pixels,
          width: newW,
          height: newH,
          name: file.name,
        };

        // update info text
        var infoEl = document.getElementById("img-info");
        if (infoEl) {
          infoEl.textContent = file.name + " — " + newW + " × " + newH + " (grayscale uint16)";
        }

        // plot preview with Plotly heatmap
        if (typeof Plotly !== "undefined" && document.getElementById(plotDiv)) {
          var traces = [{
            z: pixels,
            type: "heatmap",
            colorscale: "Greys",
            reversescale: false,
            showscale: false,
            zmin: 0,
            zmax: 65535,
          }];
          var layout = {
            title: { text: file.name, font: { family: "STIX Two Text", size: 14 } },
            font: { family: "STIX Two Text" },
            margin: { t: 40, b: 30, l: 40, r: 20 },
            yaxis: { autorange: "reversed", scaleanchor: "x", scaleratio: 1 },
            xaxis: { constrain: "domain" },
          };
          Plotly.react(plotDiv, traces, layout, { responsive: true });
        }

        // notify other scripts that a new image was loaded
        window.dispatchEvent(new Event("imagedata2d-changed"));
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function init() {
    var input = document.getElementById("img-file-input");
    if (input) {
      input.addEventListener("change", function () {
        if (this.files && this.files[0]) {
          handleFile(this.files[0]);
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
