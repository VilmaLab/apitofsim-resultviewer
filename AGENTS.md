# AGENTS.md

## Project

Read-only web viewer for APi-ToF simulation results stored in DuckDB.

## Layout and tooling

- `src/apitofresview/`: Python web app, plotting code, Jinja templates, and built static assets.
- `src/css/` and `src/js/`: frontend source; `tests/`: browser-based tests.
- `uv` manages Python dependencies; `npm` builds Tailwind CSS and bundles JavaScript with esbuild. The app uses Starlette, HTMX, Alpine.js, and Tabulator.
- Rebuild frontend changes with `npm run build` and commit the generated static assets so the app works offline.
- Run tests with `uv run pytest`; install Chromium first with `uv run playwright install chromium` if needed.

## Style — DRY and reuse first

Follow the patterns already in the codebase rather than inventing new ones.
Be succinct!
Prefer Tailwind utilities over custom CSS. Prefer CSS or Alpine.js over custom JavaScript; choose between CSS and Alpine.js based on which is easier for the task. Keep layouts responsive.
Keep database access read-only.

## Formatting

`prek run --all-files` runs formatting checks through `prek.toml`.
