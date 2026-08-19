# apitofsim-resultviewer

Read-only web viewer for APi-ToF simulation results.

## Installing

Download the build for your platform from the
[latest release](../../releases/latest), unpack it, and run `apitofresview`
(`APiToF Result Viewer.app` on macOS).

The builds are not code-signed, so:

- **macOS**: Gatekeeper will refuse to open a downloaded app. Clear the
  quarantine flag once, after moving it to `/Applications`:

  ```bash
  xattr -dr com.apple.quarantine "/Applications/APiToF Result Viewer.app"
  ```

- **Windows**: SmartScreen shows "Windows protected your PC". Choose
  *More info* → *Run anyway*.

On Windows and macOS the viewer opens in its own window. On Linux it starts a
local server and opens your usual browser, because the native window there
would need a system WebKitGTK installation.

## Running

The viewer needs a path to an APi-ToF experiment database (a DuckDB file).

```sh
apitofresview --database /path/to/experiments.duckdb
```

or set the database in the environment (this also keeps the previous way of
running working):

```sh
export DATABASE=/path/to/experiments.duckdb
apitofresview
```

### Options

| Option | Effect |
| --- | --- |
| `--database PATH` | Experiment database path (overrides `$DATABASE`) |
| `--port N` | Serve on a fixed port instead of a free one |
| `--no-window` | Serve only; don't open a window or a browser |
| `--no-browser` | Don't open a browser |
| `--debug` | Show tracebacks in the browser |

## The current way of running things

The viewer's logic lives in the `apitofresview` package, but the original
entry point is preserved as a thin shim, so the ASGI way of running still
works unchanged:

```sh
DATABASE=/path/to/experiments.duckdb uvicorn main:app
```

## Developing

```bash
uv sync --group build
npm install && npm run build     # build the frontend into the package
uv run apitofresview --database /path/to/experiments.duckdb
```

`uv run uvicorn apitofresview.webapp:create_app --factory` also works if you
want a plain ASGI server.

### Frontend assets

Client-side dependencies (Alpine, htmx, Tabulator, Tailwind) are pulled from
npm and bundled into `src/apitofresview/static/index.js` /
`src/apitofresview/static/index.css` (committed, so the app runs offline):

```sh
npm install
npm run build       # one-off build
npm run watch       # rebuild on change (css + js)
```

Sources live in `src/js/index.ts` (bundle entry) and `src/css/index.css`
(Tailwind v4 entry, scanning `src/apitofresview/templates/` for utility
classes). This mirrors the setup in apitofsim-web.

### Building a release locally

```bash
uv sync --group build
uv run pyinstaller apitofresview.spec --noconfirm --clean
./dist/apitofresview/apitofresview --smoke-test
```

`--smoke-test` builds a throwaway database, starts the server, requests the
pages and static trees, and forces the lazily-imported bokeh/panel/holoviews
and matplotlib code paths. It is the check that catches PyInstaller problems,
since those are runtime import failures rather than build failures. It works
unfrozen too (`uv run apitofresview --smoke-test`), so you can compare the two
directly when something breaks only in the bundle.

CI builds all four targets on every push and attaches them to a GitHub
Release on tags matching `v*`. Note that CI clones the `apitofsim` and
`mplbed` path dependencies from git before building, so a standalone
checkout needs network access to those repositories.

## Tests

Install Chromium once, then run the pytest suite. The suite starts the viewer
against a temporary DuckDB database and drives it through a real browser.

```sh
uv run playwright install chromium
uv run pytest
```
