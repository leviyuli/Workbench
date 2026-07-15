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

  function cloneValue(value) {
    if (typeof structuredClone === "function") {
      return structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
  }

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
        return cloneValue(defaultState);
      }
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") {
        return cloneValue(defaultState);
      }
      var loaded = Object.assign(cloneValue(defaultState), parsed, {
        ionomers: Array.isArray(parsed.ionomers) ? parsed.ionomers : cloneValue(defaultState.ionomers),
        extras: Array.isArray(parsed.extras) ? parsed.extras : []
      });
      loaded.mode = loaded.mode === "catalyst" ? "catalyst" : "area";
      return loaded;
    } catch (error) {
      return cloneValue(defaultState);
    }
  }

  var state = loadState();
  var form = document.getElementById("calculator");
  var modeButtons = Array.prototype.slice.call(document.querySelectorAll("[data-mode-button]"));
  var ionomerRows = document.getElementById("ionomerRows");
  var extraRows = document.getElementById("extraRows");
  var extraEmpty = document.getElementById("extraEmpty");
  var notice = document.getElementById("notice");
  var adjustedPanel = document.getElementById("adjustedPanel");
  var resultAnnouncement = document.getElementById("resultAnnouncement");

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
      input.value = roundInput(finiteNumber(state[input.dataset.field], 0));
    });
  }

  function setMode(mode) {
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
      var number = index + 1;
      return [
        '<div class="calc-row" data-index="' + index + '" role="group" aria-label="Ionomer ' + number + '">',
        fieldMarkup("Ratio", "ratio", row.ratio, "%", "Ionomer " + number + " ratio", 0, 99.99),
        fieldMarkup("Conc", "conc", row.conc, "wt%", "Ionomer " + number + " concentration", 0, 100),
        fieldMarkup("Density", "density", row.density, "g/mL", "Ionomer " + number + " density", 0, null),
        removeButton("ionomer", index, "Remove ionomer " + number),
        "</div>"
      ].join("");
    }).join("");

    extraRows.innerHTML = state.extras.map(function (row, index) {
      var number = index + 1;
      return [
        '<div class="calc-row extra-row" data-index="' + index + '" role="group" aria-label="Other solid ' + number + '">',
        textFieldMarkup("Material", "name", row.name || "Material", "Other solid " + number + " name"),
        fieldMarkup("Ratio", "ratio", row.ratio, "%", "Other solid " + number + " ratio", 0, 99.99),
        removeButton("extra", index, "Remove other solid " + number),
        "</div>"
      ].join("");
    }).join("");

    extraEmpty.hidden = state.extras.length > 0;
  }

  function fieldMarkup(label, key, value, unit, ariaLabel, min, max) {
    var constraints = ' min="' + min + '"';
    if (max !== null) {
      constraints += ' max="' + max + '"';
    }
    return [
      '<label class="field">',
      "<span>" + label + "</span>",
      '<span class="input-shell">',
      '<input type="number" inputmode="decimal" step="0.01" data-key="' + key + '"' + constraints + ' aria-label="' + escapeAttribute(ariaLabel) + '" value="' + roundInput(finiteNumber(value, 0)) + '">',
      "<span>" + unit + "</span>",
      "</span>",
      "</label>"
    ].join("");
  }

  function textFieldMarkup(label, key, value, ariaLabel) {
    return [
      '<label class="field">',
      "<span>" + label + "</span>",
      '<span class="input-shell">',
      '<input type="text" data-key="' + key + '" aria-label="' + escapeAttribute(ariaLabel) + '" value="' + escapeAttribute(value) + '">',
      "<span>name</span>",
      "</span>",
      "</label>"
    ].join("");
  }

  function removeButton(type, index, label) {
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

  function numericValue(value, label, errors) {
    var number = Number(value);
    if (!Number.isFinite(number)) {
      errors.push(label + " must be a finite number.");
      return NaN;
    }
    return number;
  }

  function invalidResult(errors) {
    return {
      valid: false,
      errors: errors,
      warnings: errors.slice(),
      catalyst_mg: null,
      ink_volume_ml: null,
      total_ionomer_mg: null,
      total_dispersion_mass_mg: null,
      total_dispersion_volume_ul: null,
      total_extra_mg: null,
      total_solids_mg: null,
      ionomers: [],
      extras: [],
      adjusted: null
    };
  }

  function calculateInkRecipe(params) {
    params = params && typeof params === "object" ? params : {};
    var errors = [];
    var mode = params.mode || "area";
    if (mode !== "area" && mode !== "catalyst") {
      errors.push("Mode must be area or catalyst.");
    }

    var loading = numericValue(params.loading === undefined ? 0 : params.loading, "Loading", errors);
    var inkConcentration = numericValue(params.ink_density === undefined ? 0 : params.ink_density, "Ink concentration", errors);
    var length = numericValue(params.length === undefined ? 0 : params.length, "Length", errors);
    var width = numericValue(params.width === undefined ? 0 : params.width, "Width", errors);
    var availableCatalyst = numericValue(params.avail_cat === undefined ? 0 : params.avail_cat, "Available catalyst", errors);

    if (mode === "area") {
      if (length <= 0) {
        errors.push("Length must be above zero in area mode.");
      }
      if (width <= 0) {
        errors.push("Width must be above zero in area mode.");
      }
      if (loading <= 0) {
        errors.push("Loading must be above zero in area mode.");
      }
    } else if (mode === "catalyst" && availableCatalyst <= 0) {
      errors.push("Available catalyst must be above zero in catalyst mode.");
    }
    if (inkConcentration <= 0) {
      errors.push("Ink concentration must be above zero.");
    }

    var rawIonomers = Array.isArray(params.ionomers) ? params.ionomers : [];
    if (params.ionomers && !Array.isArray(params.ionomers)) {
      errors.push("Ionomers must be a list of rows.");
    }
    var normalizedIonomers = rawIonomers.map(function (ionomer, index) {
      if (!ionomer || typeof ionomer !== "object") {
        errors.push("Ionomer " + (index + 1) + " must be a recipe row.");
        return { ratio: 0, conc: 0, density: 0 };
      }
      var ratio = numericValue(ionomer.ratio, "Ionomer " + (index + 1) + " ratio", errors);
      var conc = numericValue(ionomer.conc, "Ionomer " + (index + 1) + " concentration", errors);
      var density = numericValue(ionomer.density, "Ionomer " + (index + 1) + " density", errors);
      if (!(ratio >= 0 && ratio < 100)) {
        errors.push("Ionomer " + (index + 1) + " ratio must be at least 0% and below 100%.");
      }
      if (conc < 0 || conc > 100) {
        errors.push("Ionomer " + (index + 1) + " concentration must be between 0% and 100%.");
      } else if (ratio > 0 && conc <= 0) {
        errors.push("Ionomer " + (index + 1) + " concentration must be above zero when used.");
      }
      if (density < 0) {
        errors.push("Ionomer " + (index + 1) + " density cannot be negative.");
      } else if (ratio > 0 && density <= 0) {
        errors.push("Ionomer " + (index + 1) + " density must be above zero when used.");
      }
      return { ratio: ratio, conc: conc, density: density };
    });

    var rawExtras = Array.isArray(params.extras) ? params.extras : [];
    if (params.extras && !Array.isArray(params.extras)) {
      errors.push("Other solids must be a list of rows.");
    }
    var normalizedExtras = rawExtras.map(function (extra, index) {
      if (!extra || typeof extra !== "object") {
        errors.push("Other solid " + (index + 1) + " must be a recipe row.");
        return { name: "Material", ratio: 0 };
      }
      var ratio = numericValue(extra.ratio, "Other solid " + (index + 1) + " ratio", errors);
      if (!(ratio >= 0 && ratio < 100)) {
        errors.push("Other solid " + (index + 1) + " ratio must be at least 0% and below 100%.");
      }
      return { name: String(extra.name || "Material"), ratio: ratio };
    });
    var extraRatioTotal = normalizedExtras.reduce(function (sum, extra) {
      return sum + extra.ratio;
    }, 0);
    if (extraRatioTotal >= 100) {
      errors.push("Other-solid ratios must total less than 100%.");
    }

    if (errors.length) {
      return invalidResult(errors);
    }

    var catalystMg = mode === "area" ? loading * length * width : availableCatalyst;
    var totalIonomerMg = 0;
    var totalDispersionMassMg = 0;
    var totalDispersionVolumeUl = 0;
    var ionomers = normalizedIonomers.map(function (ionomer, index) {
      var ratio = ionomer.ratio / 100;
      var activeMg = (catalystMg * ratio) / (1 - ratio);
      var dispersionMg = ratio > 0 ? activeMg / (ionomer.conc / 100) : 0;
      var volumeUl = ratio > 0 ? dispersionMg / ionomer.density : 0;
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

    var extraFraction = extraRatioTotal / 100;
    var totalSolidsMg = (catalystMg + totalIonomerMg) / (1 - extraFraction);
    var extras = normalizedExtras.map(function (extra) {
      return {
        name: extra.name,
        mass_mg: totalSolidsMg * extra.ratio / 100
      };
    });
    var totalExtraMg = extras.reduce(function (sum, extra) {
      return sum + extra.mass_mg;
    }, 0);
    var inkVolumeMl = catalystMg / inkConcentration;

    var adjusted = null;
    if (mode === "area") {
      var lossFactor = Math.min(0.212 * ((1 / length) + (1 / width)), 0.99);
      var scaleFactor = 1 / (1 - lossFactor);
      adjusted = {
        loss_factor: lossFactor,
        scale_factor: scaleFactor,
        required_catalyst_mg: catalystMg * scaleFactor,
        ink_volume_ml: inkVolumeMl * scaleFactor,
        total_ionomer_mg: totalIonomerMg * scaleFactor,
        total_dispersion_mass_mg: totalDispersionMassMg * scaleFactor,
        total_dispersion_volume_ul: totalDispersionVolumeUl * scaleFactor,
        total_extra_mg: totalExtraMg * scaleFactor,
        total_solids_mg: totalSolidsMg * scaleFactor,
        ionomers: ionomers.map(function (row) {
          return {
            type: row.type,
            active_mg: row.active_mg * scaleFactor,
            dispersion_mg: row.dispersion_mg * scaleFactor,
            volume_ul: row.volume_ul * scaleFactor,
            required_volume_ul: row.volume_ul * scaleFactor
          };
        }),
        extras: extras.map(function (row) {
          return { name: row.name, mass_mg: row.mass_mg * scaleFactor };
        })
      };
    }

    return {
      valid: true,
      errors: [],
      warnings: [],
      catalyst_mg: catalystMg,
      ink_volume_ml: inkVolumeMl,
      total_ionomer_mg: totalIonomerMg,
      total_dispersion_mass_mg: totalDispersionMassMg,
      total_dispersion_volume_ul: totalDispersionVolumeUl,
      total_extra_mg: totalExtraMg,
      total_solids_mg: totalSolidsMg,
      ionomers: ionomers,
      extras: extras,
      adjusted: adjusted
    };
  }

  function renderResults(result) {
    if (!result.valid) {
      [
        "catalyst_mg", "ink_volume_ml", "total_ionomer_mg",
        "total_dispersion_volume_ul", "loss_factor", "scale_factor",
        "required_catalyst_mg", "required_ink_volume_ml",
        "required_ionomer_mg", "required_dispersion_volume_ul",
        "required_extra_mg", "required_total_solids_mg"
      ].forEach(function (name) {
        setOutput(name, "—");
      });
      adjustedPanel.hidden = true;
      notice.hidden = false;
      notice.textContent = "Cannot calculate. " + result.errors.join(" ");
      resultAnnouncement.textContent = "Calculation unavailable. " + result.errors.join(" ");
      document.getElementById("ionomerTable").innerHTML = "";
      document.getElementById("extraTable").innerHTML = "";
      return;
    }

    setOutput("catalyst_mg", format(result.catalyst_mg, 2) + " mg");
    setOutput("ink_volume_ml", format(result.ink_volume_ml, 3) + " mL");
    setOutput("total_ionomer_mg", format(result.total_ionomer_mg, 2) + " mg");
    setOutput("total_dispersion_volume_ul", format(result.total_dispersion_volume_ul, 1) + " µL");

    adjustedPanel.hidden = !result.adjusted;
    if (result.adjusted) {
      setOutput("loss_factor", format(result.adjusted.loss_factor * 100, 1) + "%");
      setOutput("scale_factor", format(result.adjusted.scale_factor, 3) + "×");
      setOutput("required_catalyst_mg", format(result.adjusted.required_catalyst_mg, 2) + " mg");
      setOutput("required_ink_volume_ml", format(result.adjusted.ink_volume_ml, 3) + " mL");
      setOutput("required_ionomer_mg", format(result.adjusted.total_ionomer_mg, 2) + " mg");
      setOutput("required_dispersion_volume_ul", format(result.adjusted.total_dispersion_volume_ul, 1) + " µL");
      setOutput("required_extra_mg", format(result.adjusted.total_extra_mg, 2) + " mg");
      setOutput("required_total_solids_mg", format(result.adjusted.total_solids_mg, 2) + " mg");
    }

    notice.hidden = true;
    notice.textContent = "";
    resultAnnouncement.textContent = "Calculation updated. Catalyst " +
      format(result.catalyst_mg, 2) + " milligrams; ink volume " +
      format(result.ink_volume_ml, 3) + " milliliters.";
    renderTables(result);
  }

  function renderTables(result) {
    document.getElementById("ionomerTable").innerHTML = tableMarkup(
      "Ionomers",
      result.adjusted
        ? ["Type", "Active mg", "Dispersion mg", "Volume µL", "Required active mg", "Required dispersion mg", "Required volume µL"]
        : ["Type", "Active mg", "Dispersion mg", "Volume µL"],
      result.ionomers.map(function (row, index) {
        var values = [
          String(row.type),
          format(row.active_mg, 3),
          format(row.dispersion_mg, 3),
          format(row.volume_ul, 2)
        ];
        if (result.adjusted) {
          var adjusted = result.adjusted.ionomers[index];
          values.push(format(adjusted.active_mg, 3));
          values.push(format(adjusted.dispersion_mg, 3));
          values.push(format(adjusted.volume_ul, 2));
        }
        return values;
      })
    );

    document.getElementById("extraTable").innerHTML = tableMarkup(
      "Other solids",
      result.adjusted ? ["Material", "Mass mg", "Required mass mg"] : ["Material", "Mass mg"],
      result.extras.map(function (row, index) {
        var values = [row.name, format(row.mass_mg, 3)];
        if (result.adjusted) {
          values.push(format(result.adjusted.extras[index].mass_mg, 3));
        }
        return values;
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
      "<thead><tr>" + headers.map(function (header) { return "<th scope=\"col\">" + header + "</th>"; }).join("") + "</tr></thead>",
      "<tbody>" + rows.map(function (row) {
        return "<tr>" + row.map(function (cell, index) {
          var tag = index === 0 ? "th scope=\"row\"" : "td";
          return "<" + tag + ">" + escapeHtml(cell) + "</" + (index === 0 ? "th" : "td") + ">";
        }).join("") + "</tr>";
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
      return "—";
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

  function focusLastInput(container, selector) {
    var inputs = container.querySelectorAll(selector);
    if (inputs.length) {
      inputs[inputs.length - 1].focus();
    }
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
    focusLastInput(ionomerRows, "[data-key='ratio']");
  });

  document.getElementById("addExtra").addEventListener("click", function () {
    readStateFromDom();
    state.extras.push({ name: "Material", ratio: 0 });
    renderRows();
    renderResults(calculateInkRecipe(state));
    saveState();
    focusLastInput(extraRows, "[data-key='name']");
  });

  form.addEventListener("click", function (event) {
    var button = event.target.closest("[data-remove]");
    if (!button) {
      return;
    }
    readStateFromDom();
    var index = Number(button.dataset.index);
    var type = button.dataset.remove;
    if (type === "ionomer") {
      state.ionomers.splice(index, 1);
    } else {
      state.extras.splice(index, 1);
    }
    renderRows();
    renderResults(calculateInkRecipe(state));
    saveState();
    document.getElementById(type === "ionomer" ? "addIonomer" : "addExtra").focus();
  });

  document.getElementById("resetButton").addEventListener("click", function () {
    state = cloneValue(defaultState);
    saveState();
    refresh();
  });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("sw.js").then(function (registration) {
        registration.update();
      }).catch(function () {
        return;
      });
    });
  }

  refresh();
  window.calculateInkRecipe = calculateInkRecipe;
})();
