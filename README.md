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

## Embedding the Explorer

Both `explorer_bokeh(db, doc, layout=...)` from `apitofresview.webapp` and
`build_document(db, doc, experiment, cluster, layout=...)` from
`apitofresview.plotting.explorer` accept three policies:

- `"viewport"` (default): fills the remaining document viewport below the Explorer,
  with independent scrolling for controls, plots, and details.
- `"contained"`: fills a definitely sized parent region, with the same independent
  scrolling. Adding content never enlarges the region.
- `"page"`: controls, plots, and details grow in document flow. Use this for a
  content-sized page or an iframe whose host resizes it to its content.

Fullscreen temporarily fills the screen with bounded panel scrolling; exiting
restores the selected policy and sidebar placement.

### Parent-sized region in the same document

Use `layout="contained"` when the Explorer follows a region inside its document.
For example, in a Bokeh server template:

```html
{% extends base %} {% block contents %}
<header>Content above the Explorer</header>
<section id="explorer-host">
  {% for root in roots %} {{ embed(root) }} {% endfor %}
</section>
<footer>Content below the Explorer</footer>
<style>
  #explorer-host {
    height: clamp(320px, 60dvh, 720px);
    min-height: 0;
    min-width: 0;
    box-sizing: border-box;
  }
  #explorer-host > div {
    height: 100%;
    min-height: 0;
    min-width: 0;
  }
</style>
{% endblock %}
```

The Explorer’s containing block must have a definite height. Any intervening
box-generating wrappers need a resolvable height (for example `height: 100%` through a chain of
explicitly sized ancestors). A constrained grid/flex region also works; allow
its items to shrink with `min-height: 0; min-width: 0`. The Explorer uses
`box-sizing: border-box`; use the same for a host with padding/borders, and avoid
external margins on the Explorer root. The height fills the host's content box.
Parent-only width and height changes remain responsive after fullscreen.

### Bounded iframe

An iframe already supplies its own viewport, so use `layout="viewport"` inside
it. The outer page determines the iframe height:

```html
<iframe
  src="https://viewer.example/explorer?experiment=1&cluster=1"
  title="Simulation Explorer"
  allow="fullscreen"
  allowfullscreen
  style="display: block; width: 100%; height: clamp(320px, 65dvh, 760px); border: 0; box-sizing: border-box;"
></iframe>
```

Serve the Explorer with viewport layout at that URL. Disable content-driven
iframe autosizing for this view (including resize-message handlers or iframe
resizer integrations): panel content scrolls inside the fixed region and must
not feed back into the iframe height. Fullscreen requires permission from the
embedding page and any enclosing frames.

Stable CSS classes identify `.explorer-bokeh-frame`, `.explorer-controls`,
`.explorer-center`, `.explorer-plots`, and `.explorer-details`. Bokeh regions may
be inside shadow roots. The embedding application owns the available region,
iframe styling, and autosizing policy.

## Developing

```bash
uv sync --group build
npm install && npm run build     # build the frontend into the package
uv run apitofresview --database /path/to/experiments.duckdb
```

`DATABASE=/path/to/experiments.duckdb uv run uvicorn apitofresview.webapp:create_app --factory` also works if you
want a plain ASGI server. Build the frontend assets first; this entry point
does not check or build them.

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
`src/apitofresview/static/index.css`. These generated files are Git-ignored and
must be built in a fresh checkout:

```sh
npm install
npm run build       # one-off build
npm run watch       # rebuild on change (css + js)
```

The desktop command checks these assets at startup and rebuilds them when
sources change. `--reload` also watches the frontend sources and templates.
PyInstaller forces a fresh build before bundling; the frozen app validates its
bundled assets without running npm. Browser tests start through the CLI, which
builds the assets before serving. A direct ASGI launch does not build them.

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

Install Chromium once, then run pytest. The suite starts the viewer through its
CLI against a temporary DuckDB database and drives it through a real browser.

```sh
uv run playwright install chromium
uv run pytest
```
