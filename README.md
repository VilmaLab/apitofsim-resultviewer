# apitofsim-resultviewer

Read-only web viewer for APi-ToF simulation results.

## Frontend assets

Client-side dependencies (Alpine, htmx, Tabulator, Tailwind) are pulled from
npm and bundled into `static/index.js` / `static/index.css` (committed, so the
app runs offline):

```sh
npm install
npm run build       # one-off build
npm run watch       # rebuild on change (css + js)
```

Sources live in `src/js/index.ts` (bundle entry) and `src/css/index.css`
(Tailwind v4 entry, scanning `templates/` for utility classes). This mirrors
the setup in apitofsim-web.

## Tests

Run the pytest suite with:

```sh
uv run pytest
```
