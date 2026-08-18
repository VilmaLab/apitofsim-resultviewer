"""The result viewer's Starlette application."""

import re
from importlib.resources import files
from io import StringIO
from math import ceil
from os import environ
from urllib.parse import urlencode

from apitofsim.plotting import get_report  # type: ignore[reportMissingImports]
from apitofsim.workflow.db import (
    ExperimentDatabase,  # type: ignore[reportMissingImports]
)
from bokeh.server.asgi import BokehASGI  # type: ignore[reportMissingImports]
from mplbed import (  # type: ignore[reportMissingImports]
    FigureCollector,
    mplbed_starlette,
    safe_html,
)
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates


def _resource_dir(rel):
    return str(files("apitofresview") / rel)


OVERVIEW_REPORT_TYPES = [
    "cluster-report",
    "pathway-report",
    "experiment-pathway-report",
    "experiment-cluster-report",
    "experiment-summary",
    "spectrogram",
]

EXPERIMENT_REPORT_TYPES = [
    "experiment-pathway-report",
    "experiment-cluster-report",
    "spectrogram",
]

EXPERIMENT_VIEWS = [
    ("report", "report", "Report"),
    ("survivals", "survivals", "Survivals"),
    ("cluster", "cluster", "Cluster"),
]

CLUSTER_VIEWS = [
    ("report", "report", "Report"),
    ("spectrogram", "spectrogram", "Spectrogram"),
    ("realizations", "realizations", "Realizations"),
]

REPORT_PAGE_SIZE = 100
REPORT_TYPES = set(OVERVIEW_REPORT_TYPES) | set(EXPERIMENT_REPORT_TYPES)
SORT_PARAM_RE = re.compile(r"^sort\[(\d+)\]\[(field|dir)\]$")


def pathway_type_lbl(is_single_pathway):
    return "single pathway" if is_single_pathway else "multi-pathway"


def get_experiment_choices(db):
    experiment_choices = []
    df = db.report_df("experiment_summary")
    for row in df.itertuples():
        pathway_desc = pathway_type_lbl(row.is_single_pathway)
        label = (
            f"#{row.experiment_run_id} {row.config_name} run at {row.start_time} "
            f"({pathway_desc}, success rate: {row.successes}/{row.successes + row.failures})"
        )
        value = row.experiment_run_id
        experiment_choices.append((label, value))
    return experiment_choices


def get_cluster_choices(db, experiment):
    """The clusters occurring in the given experiment run, as (label, value)."""
    if experiment is None:
        return []
    df = db.db.execute(
        """
        select distinct cluster_id, cluster_common_name, cluster_atomic_mass
        from experiment_cluster_report
        where experiment_run_id = ?
        order by cluster_atomic_mass, cluster_common_name
        """,
        (experiment,),
    ).fetchdf()
    return [
        (f"{row.cluster_common_name} ({row.cluster_atomic_mass} amu)", row.cluster_id)
        for row in df.itertuples()
    ]


def maybe_int(value):
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def url_with(request, name, **params):
    """Build a URL for a named route, dropping empty query parameters."""
    url = str(request.url_for(name))
    query = urlencode({k: v for k, v in params.items() if v not in (None, "")})
    return f"{url}?{query}" if query else url


def report_types_for(experiment):
    return EXPERIMENT_REPORT_TYPES if experiment is not None else OVERVIEW_REPORT_TYPES


def selected_report(request, experiment):
    report = request.query_params.get("report", "")
    return report if report in report_types_for(experiment) else ""


def requested_report(request):
    """Return a validated report type from an API request."""
    report_type = request.query_params.get("report", "")
    if report_type not in REPORT_TYPES:
        raise ValueError("Unknown report type")
    return report_type


def positive_int_param(request, name, default):
    value = request.query_params.get(name)
    try:
        result = default if value is None else int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def requested_sorters(request):
    """Parse Tabulator's sort[n][field/dir] query parameters."""
    sorters = {}
    for key, value in request.query_params.multi_items():
        if not key.startswith("sort["):
            continue
        match = SORT_PARAM_RE.fullmatch(key)
        if match is None:
            raise ValueError("Invalid sort parameter")
        index, part = match.groups()
        sorter = sorters.setdefault(int(index), {})
        if part in sorter:
            raise ValueError("Duplicate sort parameter")
        sorter[part] = value

    result = []
    for index in sorted(sorters):
        sorter = sorters[index]
        if set(sorter) != {"field", "dir"} or sorter["dir"] not in {"asc", "desc"}:
            raise ValueError("Invalid sorter")
        result.append((sorter["field"], sorter["dir"]))
    return result


def quote_identifier(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def paginated_report(db, report_type, page, sorters):
    """Return the total row count and one sorted page of a report."""
    offset = (page - 1) * REPORT_PAGE_SIZE
    if report_type == "spectrogram":
        df = get_report(db, report_type)
        columns = set(df.columns)
        for field, _ in sorters:
            if field not in columns:
                raise ValueError("Unknown sort field")
        if sorters:
            df = df.sort_values(
                by=[field for field, _ in sorters],
                ascending=[direction == "asc" for _, direction in sorters],
            )
        return len(df), df.iloc[offset : offset + REPORT_PAGE_SIZE]

    relation = db.db.table(report_type.replace("-", "_"))
    columns = set(relation.columns)
    for field, _ in sorters:
        if field not in columns:
            raise ValueError("Unknown sort field")
    if sorters:
        order = ", ".join(
            f"{quote_identifier(field)} {direction.upper()}"
            for field, direction in sorters
        )
        relation = relation.order(order)
    row_count = relation.count("*").fetchone()[0]
    return row_count, relation.limit(REPORT_PAGE_SIZE, offset=offset).fetchdf()


def selected_cluster(request, clusters):
    cluster = maybe_int(request.query_params.get("cluster"))
    return cluster if cluster in {value for _, value in clusters} else None


def head_context(request):
    return {
        "mpl_head": safe_html.head_content(core=True),
    }


def nav_context(request):
    """Values the navigation chrome in base.html needs on every page."""
    db = _db
    experiment = maybe_int(request.query_params.get("experiment"))
    report = selected_report(request, experiment)
    clusters = get_cluster_choices(db, experiment)
    cluster = selected_cluster(request, clusters)
    # Parameters the experiment picker carries over, minus anything we dropped.
    query_params = dict(request.query_params)
    query_params.pop("report", None)
    query_params.pop("cluster", None)
    if report:
        query_params["report"] = report
    return {
        "experiments": get_experiment_choices(db),
        "experiment": experiment,
        "clusters": clusters,
        "cluster": cluster,
        "experiment_views": EXPERIMENT_VIEWS,
        "cluster_views": CLUSTER_VIEWS,
        "report": report,
        "report_types": report_types_for(experiment),
        "query_params": query_params,
        "url_with": url_with,
    }


templates = Jinja2Templates(
    directory=_resource_dir("templates"),
    context_processors=[head_context, nav_context],
)

# Set by create_app before any route is served; referenced by the route handlers
# and the template context processors above.
_db: ExperimentDatabase = None  # type: ignore[assignment]


async def overview(request):
    return templates.TemplateResponse(
        request,
        "overview.html",
        {
            "section": "overview",
            "route": "overview",
        },
    )


async def experiment(request):
    return templates.TemplateResponse(
        request,
        "experiment.html",
        {
            "section": "experiment",
            "route": "experiment",
        },
    )


async def comparison(request):
    return templates.TemplateResponse(
        request,
        "comparison.html",
        {
            "section": "comparison",
            "route": "comparison",
        },
    )


async def report(request):
    experiment = maybe_int(request.query_params.get("experiment"))
    report_type = selected_report(request, experiment)
    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "report_data_url": url_with(request, "report_data", report=report_type),
            "report_download_url": url_with(
                request, "report_download", report=report_type
            ),
            "section": "experiment" if experiment is not None else "overview",
            "view": "report",
            "route": "report",
        },
    )


async def report_data(request):
    try:
        report_type = requested_report(request)
        page = positive_int_param(request, "page", 1)
        size = positive_int_param(request, "size", REPORT_PAGE_SIZE)
        if size != REPORT_PAGE_SIZE:
            raise ValueError(f"size must be {REPORT_PAGE_SIZE}")
        sorters = requested_sorters(request)
        row_count, df = paginated_report(_db, report_type, page, sorters)
    except ValueError as exc:
        return Response(str(exc), status_code=400, media_type="text/plain")

    last_page = max(1, ceil(row_count / REPORT_PAGE_SIZE))
    data = df.to_json(orient="records")
    content = f'{{"last_page":{last_page},"last_row":{row_count},"data":{data}}}'
    return Response(content, media_type="application/json")


async def report_download(request):
    try:
        report_type = requested_report(request)
    except ValueError as exc:
        return Response(str(exc), status_code=400, media_type="text/plain")

    csv = StringIO()
    get_report(_db, report_type).to_csv(csv, index=False)
    return Response(
        csv.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{report_type}.csv"',
        },
    )


async def survivals(request):
    return templates.TemplateResponse(
        request,
        "survivals.html",
        {
            "section": "experiment",
            "view": "survivals",
            "route": "survivals",
        },
    )


async def cluster(request):
    return templates.TemplateResponse(
        request,
        "cluster.html",
        {
            "section": "experiment",
            "view": "cluster",
            "route": "cluster",
        },
    )


def spectrogram_mpl(experiment, cluster):
    import holoviews  # type: ignore[reportMissingImports]
    from apitofsim.plotting import (  # type: ignore[reportMissingImports]
        basic_spectrogram,
        get_intensities,
    )

    df = get_intensities(
        _db,
        experiment_id=experiment,
        cluster_id=cluster,
        is_single_pathway=get_is_single_pathway(experiment, cluster),  # type: ignore[arg-type]
    )
    renderer = holoviews.renderer("matplotlib")
    collector = FigureCollector(target="inline", on_close="remove")
    with collector:
        renderer.show(basic_spectrogram(df))
    return collector.consume_one()


async def spectrogram_page(request):
    from bokeh.embed import server_document  # type: ignore[reportMissingImports]

    url = str(request.url_for("bokeh", path="/spectrogram"))
    experiment_id = request.query_params.get("experiment")
    cluster_id = request.query_params.get("cluster")
    script = server_document(
        url,
        arguments={
            "experiment": experiment_id,
            "cluster": cluster_id,
        },
    )
    spectrogram = spectrogram_mpl(maybe_int(experiment_id), maybe_int(cluster_id))

    return templates.TemplateResponse(
        request,
        "spectrogram.html",
        {
            "section": "experiment",
            "view": "spectrogram",
            "route": "spectrogram",
            "spectrogram_bokeh": script,
            "spectrogram_mpl": spectrogram,
        },
    )


async def realizations(request):
    return templates.TemplateResponse(
        request,
        "realizations.html",
        {
            "section": "experiment",
            "view": "realizations",
            "route": "realizations",
        },
    )


def get_is_single_pathway(experiment, cluster):
    if experiment is None or cluster is None:
        return None
    row = _db.db.execute(
        """
        select is_single_pathway
        from experiment_cluster_report
        where experiment_run_id = ? and cluster_id = ?
        """,
        (experiment, cluster),
    ).fetchone()
    return row[0] if row else None


def spectrogram_bokeh(doc):
    import holoviews  # type: ignore[reportMissingImports]
    from apitofsim.plotting import (  # type: ignore[reportMissingImports]
        basic_spectrogram,
        get_intensities,
    )
    from bokeh.layouts import layout  # type: ignore[reportMissingImports]
    from panel import layout  # type: ignore[reportMissingImports]

    args = doc.session_context.request.arguments

    def arg(name):
        vals = args.get(name)
        return maybe_int(vals[-1].decode()) if vals else None

    experiment = arg("experiment")
    cluster = arg("cluster")
    df = get_intensities(
        _db,
        experiment_id=experiment,
        cluster_id=cluster,
        is_single_pathway=get_is_single_pathway(experiment, cluster),  # type: ignore[arg-type]
    )
    renderer = holoviews.renderer("bokeh").instance(mode="server")
    plot = renderer.get_plot(basic_spectrogram(df), doc)
    root = layout(  # type: ignore[call-arg]
        [[plot.state]], sizing_mode="fixed"
    )
    doc.add_root(root)


def create_app(database_path=None, debug=True):
    """Build the Starlette application around the given experiment database.

    ``database_path`` defaults to the ``$DATABASE`` environment variable,
    which keeps the ``uvicorn main:app`` / ``import main`` way of running
    working unchanged.
    """
    global _db
    if database_path is None:
        database_path = environ["DATABASE"]
    _db = ExperimentDatabase(database_path, readonly=True)

    app = Starlette(
        debug=debug,
        routes=[
            Route("/", overview, name="overview"),
            Route("/experiment", experiment, name="experiment"),
            Route("/report", report, name="report"),
            Route("/report/data", report_data, name="report_data"),
            Route("/report/download", report_download, name="report_download"),
            Route("/experiment/survivals", survivals, name="survivals"),
            Route("/experiment/cluster", cluster, name="cluster"),
            Route(
                "/experiment/cluster/spectrogram", spectrogram_page, name="spectrogram"
            ),
            Route(
                "/experiment/cluster/realizations", realizations, name="realizations"
            ),
            Route("/comparison", comparison, name="comparison"),
            Mount(
                "/static", StaticFiles(directory=_resource_dir("static")), name="static"
            ),
            Mount(
                "/bokeh", BokehASGI({"/spectrogram": spectrogram_bokeh}), name="bokeh"
            ),
        ],
    )
    mplbed_starlette.setup(app)
    return app


# Default app built from $DATABASE at import time, so ``uvicorn main:app`` and
# ``import main`` keep working. When $DATABASE is absent (the packaged CLI
# passes --database instead) leave it unset rather than failing at import.
try:
    app = create_app()
except KeyError:
    app = None
