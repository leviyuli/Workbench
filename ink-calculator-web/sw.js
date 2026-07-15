const CACHE_PREFIX = "ink-calculator-";
const CACHE_NAME = `${CACHE_PREFIX}v2`;
const APP_ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.webmanifest",
  "./assets/icon.svg",
  "./assets/icon-192.png",
  "./assets/icon-512.png"
];

function scopedUrl(path) {
  return new URL(path, self.registration.scope).toString();
}

function cacheIfUsable(cache, request, response) {
  if (response && response.ok) {
    return cache.put(request, response.clone());
  }
  return Promise.resolve();
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => Promise.all(
      APP_ASSETS.map((path) => {
        const request = new Request(scopedUrl(path), { cache: "reload" });
        return fetch(request).then((response) => {
          if (!response.ok) {
            throw new Error(`Could not precache ${path}: HTTP ${response.status}`);
          }
          return cache.put(request, response);
        });
      })
    )).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys
        .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME)
        .map((key) => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") {
    return;
  }

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return;
  }

  // Network-first keeps deployed calculations and markup current.  The cache
  // is strictly an offline fallback, and error responses are never persisted.
  event.respondWith(
    caches.open(CACHE_NAME).then((cache) => fetch(request).then((response) => {
      if (!response.ok) {
        return cache.match(request).then((cached) => cached || response);
      }
      return cacheIfUsable(cache, request, response)
        .catch(() => undefined)
        .then(() => response);
    }).catch(() => cache.match(request).then((cached) => {
      if (cached) {
        return cached;
      }
      if (request.mode === "navigate") {
        return cache.match(scopedUrl("./index.html")).then(
          (fallback) => fallback || Response.error()
        );
      }
      return Response.error();
    })))
  );
});
