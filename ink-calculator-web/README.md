# Ink Calculator Web

Static Android-friendly PWA for the AEM catalyst-ink recipe calculation.

## Run locally

```powershell
cd ink-calculator-web
python -m http.server 4173
```

Open `http://localhost:4173`.

## Deploy on Netlify

The root `netlify.toml` publishes this folder directly:

```toml
[build]
  publish = "ink-calculator-web"
```

No build command or dependencies are required.
