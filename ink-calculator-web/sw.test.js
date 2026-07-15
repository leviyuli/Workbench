const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");


function createHarness() {
  const listeners = {};
  const entries = new Map();
  const puts = [];
  const key = (request) => typeof request === "string" ? request : request.url;
  const cache = {
    match(request) {
      const response = entries.get(key(request));
      return Promise.resolve(response ? response.clone() : undefined);
    },
    put(request, response) {
      puts.push({ url: key(request), status: response.status });
      entries.set(key(request), response.clone());
      return Promise.resolve();
    }
  };
  const caches = {
    open() { return Promise.resolve(cache); },
    keys() { return Promise.resolve([]); },
    delete() { return Promise.resolve(true); }
  };
  const self = {
    addEventListener(type, listener) { listeners[type] = listener; },
    clients: { claim() { return Promise.resolve(); } },
    location: { origin: "https://example.test" },
    registration: { scope: "https://example.test/calculator/" },
    skipWaiting() { return Promise.resolve(); }
  };
  const context = vm.createContext({
    caches,
    console,
    fetch() { throw new Error("fetch implementation not set"); },
    Promise,
    Request,
    Response,
    self,
    URL
  });
  const source = fs.readFileSync(path.join(__dirname, "sw.js"), "utf8");
  vm.runInContext(source, context, { filename: "sw.js" });

  async function fetchThroughWorker(url, fetchImpl) {
    context.fetch = fetchImpl;
    const event = {
      request: new Request(url),
      respondWith(promise) { this.responsePromise = promise; }
    };
    listeners.fetch(event);
    return event.responsePromise;
  }

  return { entries, fetchThroughWorker, puts };
}


test("network-first replaces a stale cached asset", async () => {
  const harness = createHarness();
  const url = "https://example.test/calculator/app.js";
  harness.entries.set(url, new Response("old", { status: 200 }));

  const response = await harness.fetchThroughWorker(
    url,
    async () => new Response("new", { status: 200 })
  );

  assert.equal(await response.text(), "new");
  assert.equal(await harness.entries.get(url).text(), "new");
});


test("non-OK network responses are not cached and use the last good asset", async () => {
  const harness = createHarness();
  const url = "https://example.test/calculator/app.js";
  harness.entries.set(url, new Response("last-good", { status: 200 }));

  const response = await harness.fetchThroughWorker(
    url,
    async () => new Response("deploy-error", { status: 503 })
  );

  assert.equal(await response.text(), "last-good");
  assert.equal(harness.puts.length, 0);
  assert.equal(await harness.entries.get(url).text(), "last-good");
});


test("offline requests fall back to the cache", async () => {
  const harness = createHarness();
  const url = "https://example.test/calculator/styles.css";
  harness.entries.set(url, new Response("cached", { status: 200 }));

  const response = await harness.fetchThroughWorker(
    url,
    async () => { throw new Error("offline"); }
  );

  assert.equal(await response.text(), "cached");
});
