(function () {
  "use strict";

  var STORAGE_KEY = "ink-calculator-state-v1";

  var defaultState = {
    mode: "area",
    length: 5,
    width: 5,
    loading: 0.5,
    avail_cat: 12.5,
    ink_density: 2,
    ionomers: [
      { ratio: 20, conc: 5, density: 0.78 }
    ],
    extras: []
  };

  var state = loadState();

  var form = document.getElementById("calculator");
  var modeButtons = Array.prototype.slice.call(document.querySelectorAll("[data-mode-button]"));
  var ionomerRows = document.getElementById("ionomerRows");
  var extraRows = document.getElementById("extraRows");
  var extraEmpty = document.getElementById("extraEmpty");
  var notice = document.getElementById("notice");

  function finiteNumber(value, fallback) {
    var next = Number(value);
    return Number.isFinite(next) ? next : fallback;
  }

  function roundInput(value) {
    if (!Number.isFinite(value)) {
      return "";
    }
    return String(Number.parseFloat(value.toFixed(6)));
  }

  function loadState() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return structuredClone(defaultState);
      }
      var parsed = JSON.parse(raw);
      return Object.assign(structuredClone(defaultState), parsed, {
        ionomers: Array.isArray(parsed.ionomers) ? parsed.ionomers : defaultState.ionomers,
        extras: Array.isArray(parsed.extras) ? parsed.extras : defaultState.extras
      });
    } catch (error) {
      return structuredClone(defaultState);
    }
  }

  function saveState() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      return;
    }
  }

  function readStateFromDom() {
    var next = Object.assign({}, state);
    document.querySelectorAll("[data-field]").forEach(function (input) {
      next[input.dataset.field] = finiteNumber(input.value, 0);
    });

    next.ionomers = Array.prototype.slice.call(ionomerRows.querySelectorAll(".calc-row")).map(function (row) {
      return {
        ratio: finiteNumber(row.querySelector("[data-key='ratio']").value, 0),
        conc: finiteNumber(row.querySelector("[data-key='conc']").value, 0),
        density: finiteNumber(row.querySelector("[data-key='density']").value, 0)
      };
    });

    next.extras = Array.prototype.slice.call(extraRows.querySelectorAll(".calc-row")).map(function (row) {
      return {
        name: row.querySelector("[data-key='name']").value.trim() || "Material",
        ratio: finiteNumber(row.querySelector("[data-key='ratio']").value, 0)
      };
    });

    state = next;
  }

  function syncStaticFields() {
    document.querySelectorAll("[data-field]").forEach(function (input) {
      input.value = roundInput(state[input.dataset.field]);
    });
  }

  function setMode(mode) {
    state.mode = mode;
    modeButtons.forEach(function (button) {
      var active = button.dataset.modeButton === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll(".mode-area").forEach(function (node) {
      node.hidden = mode !== "area";
    });
    document.querySelectorAll(".mode-catalyst").forEach(function (node) {
      node.hidden = mode !== "catalyst";
    });
  }

  function renderRows() {
    ionomerRows.innerHTML = state.ionomers.map(function (row, index) {
      return [
        '<div class="calc-row" data-index="' + index + '">',
        fieldMarkup("Ratio", "ratio", row.ratio, "%"),
        fieldMarkup("Conc", "conc", row.conc, "wt%"),
        fieldMarkup("Density", "density", row.density, "g/mL"),
        removeButton("ionomer", index),
        "</div>"
      ].join("");
    }).join("");

    extraRows.innerHTML = state.extras.map(function (row, index) {
      return [
        '<div class="calc-row extra-row" data-index="' + index + '">',
        textFieldMarkup("Material", "name", row.name || "Material"),
        fieldMarkup("Ratio", "ratio", row.ratio, "%"),
        removeButton("extra", index),
        "</div>"
      ].join("");
    }).join("");

    extraEmpty.hidden = state.extras.length > 0;
  }

  function fieldMarkup(label, key, value, unit) {
    return [
      '<label class="field">',
      "<span>" + label + "</span>",
      '<span class="input-shell">',
      '<input type="number" inputmode="decimal" min="0" step="0.01" data-key="' + key + '" value="' + roundInput(finiteNumber(value, 0)) + '">',
      "<span>" + unit + "</span>",
      "</span>",
      "</label>"
    ].join("");
  }

  function textFieldMarkup(label, key, value) {
    return [
      '<label class="field">',
      "<span>" + label + "</span>",
      '<span class="input-shell">',
      '<input type="text" data-key="' + key + '" value="' + escapeAttribute(value) + '">',
      "<span>name</span>",
      "</span>",
      "</label>"
    ].join("");
  }

  function removeButton(type, index) {
    var label = type === "ionomer" ? "Remove ionomer" : "Remove solid";
    return [
      '<button class="icon-button danger" type="button" data-remove="' + type + '" data-index="' + index + '" aria-label="' + label + '" title="' + label + '">',
      '<svg aria-hidden="true" viewBox="0 0 24 24">',
      '<path d="M3 6h18"></path>',
      '<path d="M8 6V4h8v2"></path>',
      '<path d="M6 6l1 14h10l1-14"></path>',
      '<path d="M10 11v5"></path>',
      '<path d="M14 11v5"></path>',
      "</svg>",
      "</button>"
    ].join("");
  }

  function escapeAttribute(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function calculateInkRecipe(params) {
    var mode = params.mode || "area";
    var loading = finiteNumber(params.loading, 0);
    var inkDensity = finiteNumber(params.ink_density, 0);
    var length = finiteNumber(params.length, 0);
    var width = finiteNumber(params.width, 0);
    var availableCatalyst = finiteNumber(params.avail_cat, 0);
    var warnings = [];

    var catalystMg = mode === "area" ? loading * length * width : availableCatalyst;
    if (mode === "area" && (length <= 0 || width <= 0 || loading <= 0)) {
      warnings.push("Area mode needs length, width, and loading above zero.");
    }
    if (mode === "catalyst" && availableCatalyst <= 0) {
      warnings.push("Catalyst mode needs available catalyst above zero.");
    }
    if (inkDensity <= 0) {
      warnings.push("Ink density must be above zero.");
    }

    var totalIonomerMg = 0;
    var totalDispersionMassMg = 0;
    var totalDispersionVolumeUl = 0;

    var ionomers = (params.ionomers || []).map(function (ionomer, index) {
      var pct = finiteNumber(ionomer.ratio, 0) / 100;
      var conc = finiteNumber(ionomer.conc, 0) / 100;
      var density = finiteNumber(ionomer.density, 0);
      var activeMg = pct < 1 ? (catalystMg * pct) / (1 - pct) : 0;
      var dispersionMg = conc > 0 ? activeMg / conc : 0;
      var volumeUl = density > 0 ? (dispersionMg / 1000 / density) * 1000 : 0;

      if (pct >= 1) {
        warnings.push("Ionomer " + (index + 1) + " ratio must stay below 100%.");
      }
      if (pct > 0 && conc <= 0) {
        warnings.push("Ionomer " + (index + 1) + " concentration must be above zero.");
      }
      if (pct > 0 && density <= 0) {
        warnings.push("Ionomer " + (index + 1) + " density must be above zero.");
      }

      totalIonomerMg += activeMg;
      totalDispersionMassMg += dispersionMg;
      totalDispersionVolumeUl += volumeUl;

      return {
        type: index + 1,
        active_mg: activeMg,
        dispersion_mg: dispersionMg,
        volume_ul: volumeUl
      };
    });

    var extrasInput = params.extras || [];
    var sumRatio = extrasInput.reduce(function (sum, extra) {
      return sum + finiteNumber(extra.ratio, 0) / 100;
    }, 0);
    var baseSolids = catalystMg + totalIonomerMg;
    var totalSolids = sumRatio < 1 ? baseSolids / (1 - sumRatio) : 0;
    if (sumRatio >= 1) {
      warnings.push("Other solids must total less than 100%.");
    }

    var extras = extrasInput.map(function (extra) {
      var ratio = finiteNumber(extra.ratio, 0) / 100;
      return {
        name: extra.name || "Material",
        mass_mg: totalSolids * ratio
      };
    });

    var adjusted = null;
    if (mode === "area" && length > 0 && width > 0) {
      var lossFactor = Math.min(0.212 * ((1 / length) + (1 / width)), 0.99);
      var scaleFactor = 1 / (1 - lossFactor);
      adjusted = {
        loss_factor: lossFactor,
        scale_factor: scaleFactor,
        required_catalyst_mg: catalystMg * scaleFactor,
        ionomers: ionomers.map(function (row) {
          return {
            type: row.type,
            required_volume_ul: row.volume_ul * scaleFactor
          };
        })
      };
    }

    return {
      catalyst_mg: catalystMg,
      ink_volume_ml: inkDensity > 0 ? catalystMg / inkDensity : 0,
      total_ionomer_mg: totalIonomerMg,
      total_dispersion_mass_mg: totalDispersionMassMg,
      total_dispersion_volume_ul: totalDispersionVolumeUl,
      ionomers: ionomers,
      extras: extras,
      adjusted: adjusted,
      warnings: warnings
    };
  }

  function renderResults(result) {
    setOutput("catalyst_mg", format(result.catalyst_mg, 2) + " mg");
    setOutput("ink_volume_ml", format(result.ink_volume_ml, 3) + " mL");
    setOutput("total_ionomer_mg", format(result.total_ionomer_mg, 2) + " mg");
    setOutput("total_dispersion_volume_ul", format(result.total_dispersion_volume_ul, 1) + " uL");

    if (result.adjusted) {
      setOutput("loss_factor", format(result.adjusted.loss_factor * 100, 1) + "%");
      setOutput("scale_factor", format(result.adjusted.scale_factor, 3) + "x");
      setOutput("required_catalyst_mg", format(result.adjusted.required_catalyst_mg, 2) + " mg");
    } else {
      setOutput("loss_factor", "0.0%");
      setOutput("scale_factor", "1.000x");
      setOutput("required_catalyst_mg", "0.00 mg");
    }

    notice.hidden = result.warnings.length === 0;
    notice.textContent = result.warnings.join(" ");
    renderTables(result);
  }

  function renderTables(result) {
    document.getElementById("ionomerTable").innerHTML = tableMarkup(
      "Ionomers",
      ["Type", "Active mg", "Dispersion mg", "Volume uL", "Adjusted uL"],
      result.ionomers.map(function (row) {
        var adjusted = result.adjusted ? result.adjusted.ionomers[row.type - 1].required_volume_ul : 0;
        return [
          String(row.type),
          format(row.active_mg, 3),
          format(row.dispersion_mg, 3),
          format(row.volume_ul, 2),
          result.adjusted ? format(adjusted, 2) : "-"
        ];
      })
    );

    document.getElementById("extraTable").innerHTML = tableMarkup(
      "Other solids",
      ["Material", "Mass mg"],
      result.extras.map(function (row) {
        return [row.name, format(row.mass_mg, 3)];
      })
    );
  }

  function tableMarkup(caption, headers, rows) {
    if (!rows.length) {
      return "";
    }
    return [
      "<table>",
      "<caption>" + caption + "</caption>",
      "<thead><tr>" + headers.map(function (header) { return "<th>" + header + "</th>"; }).join("") + "</tr></thead>",
      "<tbody>" + rows.map(function (row) {
        return "<tr>" + row.map(function (cell) { return "<td>" + escapeHtml(cell) + "</td>"; }).join("") + "</tr>";
      }).join("") + "</tbody>",
      "</table>"
    ].join("");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function setOutput(name, value) {
    var node = document.querySelector("[data-output='" + name + "']");
    if (node) {
      node.textContent = value;
    }
  }

  function format(value, digits) {
    if (!Number.isFinite(value)) {
      return "0";
    }
    return new Intl.NumberFormat("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }).format(value);
  }

  function refresh() {
    setMode(state.mode);
    syncStaticFields();
    renderRows();
    renderResults(calculateInkRecipe(state));
  }

  function recalculateFromDom() {
    readStateFromDom();
    setMode(state.mode);
    renderResults(calculateInkRecipe(state));
    saveState();
  }

  modeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      readStateFromDom();
      state.mode = button.dataset.modeButton;
      setMode(state.mode);
      renderResults(calculateInkRecipe(state));
      saveState();
    });
  });

  form.addEventListener("input", recalculateFromDom);

  document.getElementById("addIonomer").addEventListener("click", function () {
    readStateFromDom();
    state.ionomers.push({ ratio: 0, conc: 5, density: 0.78 });
    renderRows();
    renderResults(calculateInkRecipe(state));
    saveState();
  });

  document.getElementById("addExtra").addEventListener("click", function () {
    readStateFromDom();
    state.extras.push({ name: "Material", ratio: 0 });
    renderRows();
    renderResults(calculateInkRecipe(state));
    saveState();
  });

  form.addEventListener("click", function (event) {
    var button = event.target.closest("[data-remove]");
    if (!button) {
      return;
    }
    readStateFromDom();
    var index = Number(button.dataset.index);
    if (button.dataset.remove === "ionomer") {
      state.ionomers.splice(index, 1);
    } else {
      state.extras.splice(index, 1);
    }
    renderRows();
    renderResults(calculateInkRecipe(state));
    saveState();
  });

  document.getElementById("resetButton").addEventListener("click", function () {
    state = structuredClone(defaultState);
    saveState();
    refresh();
  });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("sw.js").catch(function () {
        return;
      });
    });
  }

  refresh();
  window.calculateInkRecipe = calculateInkRecipe;
})();
