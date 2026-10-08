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
  _More info_ → _Run anyway_.

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

| Option            | Effect                                           |
| ----------------- | ------------------------------------------------ |
| `--database PATH` | Experiment database path (overrides `$DATABASE`) |
| `--port N`        | Serve on a fixed port instead of a free one      |
| `--no-window`     | Serve only; don't open a window or a browser     |
| `--no-browser`    | Don't open a browser                             |
| `--debug`         | Show tracebacks in the browser                   |

## Developing

```bash
uv sync --group build
npm install && npm run build     # build the frontend into the package
uv run apitofresview --database /path/to/experiments.duckdb
```

`DATABASE=/path/to/experiments.duckdb uv run uvicorn apitofresview.webapp:create_app --factory` also works if you
want a plain ASGI server.

### Using a development copy of apitofsim

Check out apitofsim as a sibling directory of apitofsim-resultviewer. From the
apitofsim-resultviewer directory, run:

```bash
./install_dev.sh
```

This creates `.venv-local` from the release lockfile, installs Meson build
tools, and installs the sibling apitofsim checkout as an editable package
explicitly into `.venv-local/bin/python`. Building apitofsim also requires a
C++ toolchain and its native dependencies, including TBB >=2023.0 and
Eigen >=3.4.

Set `UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=.venv-local` when running development
commands. Disabling sync preserves the editable install instead of restoring
the pinned release wheel.

```bash
UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=.venv-local uv run apitofresview \
  --database /path/to/experiments.duckdb
```

To check the selected interpreter and simulator installation without importing
or rebuilding apitofsim:

```bash
UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=.venv-local uv run python -c \
  'import sys, importlib.metadata as m; print(sys.executable); print(m.distribution("apitofsim").read_text("direct_url.json"))'
```

The output should show `.venv-local/bin/python`, the sibling apitofsim
checkout, and `"editable": true`. To switch back to the pinned release, drop
`UV_NO_SYNC` and `UV_PROJECT_ENVIRONMENT` (or remove `.venv-local`).

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
