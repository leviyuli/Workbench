const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");


function loadCalculator() {
  const genericElement = () => ({
    addEventListener() {},
    focus() {},
    hidden: false,
    innerHTML: "",
    textContent: "",
    querySelectorAll() { return []; },
    setAttribute() {}
  });
  const elements = new Map();
  [
    "calculator", "ionomerRows", "extraRows", "extraEmpty", "notice",
    "adjustedPanel", "resultAnnouncement", "ionomerTable", "extraTable",
    "addIonomer", "addExtra", "resetButton"
  ].forEach((id) => elements.set(id, genericElement()));

  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, genericElement());
      }
      return elements.get(id);
    },
    querySelector() { return null; },
    querySelectorAll() { return []; }
  };
  const window = {
    addEventListener() {},
    localStorage: {
      getItem() { return null; },
      setItem() {}
    }
  };
  const context = vm.createContext({
    console,
    document,
    Intl,
    navigator: {},
    window
    // Deliberately omit structuredClone to exercise the compatibility fallback.
  });
  const source = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });
  return window.calculateInkRecipe;
}


const calculate = loadCalculator();
const validArea = {
  mode: "area",
  length: 5,
  width: 4,
  loading: 0.5,
  avail_cat: 12.5,
  ink_density: 2,
  ionomers: [{ ratio: 20, conc: 5, density: 0.78 }],
  extras: [{ name: "Carbon", ratio: 10 }]
};


test("invalid web recipes have no actionable numeric result", () => {
  const invalid = calculate({ ...validArea, ink_density: 0 });
  assert.equal(invalid.valid, false);
  assert.equal(invalid.catalyst_mg, null);
  assert.equal(invalid.adjusted, null);
  assert.match(invalid.errors.join(" "), /Ink concentration must be above zero/);
});


test("web validation rejects invalid component domains", () => {
  const badIonomer = calculate({
    ...validArea,
    ionomers: [{ ratio: 100, conc: 0, density: -1 }]
  });
  assert.equal(badIonomer.valid, false);
  assert.match(badIonomer.errors.join(" "), /below 100%/);

  const badExtras = calculate({
    ...validArea,
    extras: [{ name: "A", ratio: 60 }, { name: "B", ratio: 40 }]
  });
  assert.equal(badExtras.valid, false);
  assert.match(badExtras.errors.join(" "), /total less than 100%/);
});


test("area adjustment contains scaled recipe totals and components", () => {
  const result = calculate(validArea);
  assert.equal(result.valid, true);
  const scale = result.adjusted.scale_factor;
  assert.equal(result.adjusted.ink_volume_ml, result.ink_volume_ml * scale);
  assert.equal(result.adjusted.total_solids_mg, result.total_solids_mg * scale);
  assert.equal(
    result.adjusted.ionomers[0].dispersion_mg,
    result.ionomers[0].dispersion_mg * scale
  );
  assert.equal(
    result.adjusted.extras[0].mass_mg,
    result.extras[0].mass_mg * scale
  );
});


test("catalyst mode does not invent a spray-loss required catalyst", () => {
  const result = calculate({ ...validArea, mode: "catalyst", avail_cat: 8 });
  assert.equal(result.valid, true);
  assert.equal(result.catalyst_mg, 8);
  assert.equal(result.adjusted, null);
});
