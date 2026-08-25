"""The result viewer's Starlette application."""

import json
import re
from importlib.resources import files
from io import StringIO
from math import ceil
from os import environ
from urllib.parse import urlencode
from functools import partial

import duckdb
from apitofsim.plotting.report import get_report
from apitofsim.workflow.db import (
    ExperimentDatabase,
)
from bokeh.server.asgi import BokehASGI
from mplbed import (
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
    ("overview", "experiment", "Overview"),
    ("report", "report", "Report"),
    ("survivals", "survivals", "Survivals"),
    ("cluster", "cluster", "Cluster"),
]

CLUSTER_VIEWS = [
    ("report", "report", "Report"),
    ("spectrogram", "spectrogram", "Spectrogram"),
    ("realizations", "realizations", "Realizations"),
]

DEFAULT_REPORT_PAGE_SIZE = 40
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


def get_experiment_overview(db, experiment):
    """Return display-ready run metadata and configuration for one experiment."""
    row = db.db.execute(
        """
        select
            summary.experiment_run_id,
            summary.config_name,
            summary.start_time,
            summary.successes,
            summary.failures,
            summary.is_single_pathway,
            config.config,
            run.run_config
        from experiment_summary as summary
        join experiment_run as run on run.id = summary.experiment_run_id
        join experiment_config as config on config.id = run.experiment_config_id
        where summary.experiment_run_id = ?
        """,
        (experiment,),
    ).fetchone()
    if row is None:
        return None

    start_time = row[2]
    return {
        "experiment_run_id": row[0],
        "config_name": row[1],
        "start_time": f"{start_time.day} {start_time:%b %Y, %H:%M:%S}",
        "successes": row[3],
        "failures": row[4],
        "pathway_mode": "Single pathway" if row[5] else "Multi-pathway",
        "configuration": {
            "experiment_config": json.loads(row[6]),
            "run_config": json.loads(row[7]),
        },
    }


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


def optional_positive_int_param(request, name):
    if request.query_params.get(name) is None:
        return None
    return positive_int_param(request, name, 1)


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


def paginated_report(db, report_type, page, sorters, experiment=None, cluster=None, size=DEFAULT_REPORT_PAGE_SIZE):
    """Return the total row count and one sorted page of a report."""
    offset = (page - 1) * size
    if report_type == "spectrogram":
        df = get_report(db, report_type)
        if experiment is not None:
            df = df[df["experiment_run_id"] == experiment]
        if cluster is not None:
            df = df[df["cluster_id"] == cluster]
        columns = set(df.columns)
        for field, _ in sorters:
            if field not in columns:
                raise ValueError("Unknown sort field")
        if sorters:
            df = df.sort_values(
                by=[field for field, _ in sorters],
                ascending=[direction == "asc" for _, direction in sorters],
            )
        return len(df), df.iloc[offset : offset + size]

    relation = db.db.table(report_type.replace("-", "_"))
    if experiment is not None:
        relation = relation.filter(
            duckdb.ColumnExpression("experiment_run_id")
            == duckdb.ConstantExpression(experiment)
        )
    if cluster is not None:
        relation = relation.filter(
            duckdb.ColumnExpression("cluster_id")
            == duckdb.ConstantExpression(cluster)
        )
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
    return row_count, relation.limit(size, offset=offset).fetchdf()


def selected_cluster(request, clusters):
    cluster = maybe_int(request.query_params.get("cluster"))
    return cluster if cluster in {value for _, value in clusters} else None


def head_context(request):
    return {
        "mpl_head": safe_html.head_content(core=True),
    }


def nav_context(request):
    """Values the navigation chrome in base.html needs on every page."""
    db = request.app.state.db
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
    experiment_id = maybe_int(request.query_params.get("experiment"))
    return templates.TemplateResponse(
        request,
        "experiment.html",
        {
            "section": "experiment",
            "view": "overview",
            "route": "experiment",
            "overview": (
                get_experiment_overview(request.app.state.db, experiment_id)
                if experiment_id is not None
                else None
            ),
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
            "report_data_url": url_with(
                request, "report_data", report=report_type, experiment=experiment
            ),
            "report_download_url": url_with(
                request,
                "report_download",
                report=report_type,
                experiment=experiment,
            ),
            "section": "experiment" if experiment is not None else "overview",
            "view": "report",
            "route": "report",
        },
    )


async def report_data(request):
    db = request.app.state.db
    report_type = requested_report(request)
    page = positive_int_param(request, "page", 1)
    size = positive_int_param(request, "size", DEFAULT_REPORT_PAGE_SIZE)
    sorters = requested_sorters(request)
    experiment = optional_positive_int_param(request, "experiment")
    cluster = optional_positive_int_param(request, "cluster")
    if experiment is not None and report_type not in EXPERIMENT_REPORT_TYPES:
        raise ValueError("Report cannot be filtered by experiment")
    row_count, df = paginated_report(
        db, report_type, page, sorters, experiment=experiment, cluster=cluster, size=size
    )

    last_page = max(1, ceil(row_count / size))
    data = df.to_json(orient="records")
    content = f'{{"last_page":{last_page},"last_row":{row_count},"data":{data}}}'
    return Response(content, media_type="application/json")


async def report_download(request):
    db = request.app.state.db
    try:
        report_type = requested_report(request)
        experiment = optional_positive_int_param(request, "experiment")
        if experiment is not None and report_type not in EXPERIMENT_REPORT_TYPES:
            raise ValueError("Report cannot be filtered by experiment")
    except ValueError as exc:
        return Response(str(exc), status_code=400, media_type="text/plain")

    csv = StringIO()
    df = get_report(db, report_type)
    if experiment is not None:
        df = df[df["experiment_run_id"] == experiment]
    df.to_csv(csv, index=False)
    return Response(
        csv.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{report_type}.csv"',
        },
    )


async def survivals(request):
    from apitofsim.plotting.survivals import make_survival_plot, get_joint_survivals
    from mplbed import mplbed_starlette, safe_html
    db = request.app.state.db
    experiment = maybe_int(request.query_params.get("experiment"))
    joint_survivals = get_joint_survivals(db, experiment)
    fig = make_survival_plot(joint_survivals.keys(), joint_survivals.values())
    return templates.TemplateResponse(
        request,
        "survivals.html",
        {
            "section": "experiment",
            "view": "survivals",
            "route": "survivals",
            "survivals": safe_html.figure_html(fig),
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


def spectrogram_mpl(db, experiment, cluster):
    import holoviews  # type: ignore[reportMissingImports]
    from apitofsim.plotting.spectrogram import (  # type: ignore[reportMissingImports]
        basic_spectrogram,
        get_intensities,
    )

    df = get_intensities(
        db,
        experiment_id=experiment,
        cluster_id=cluster,
        is_single_pathway=get_is_single_pathway(db, experiment, cluster),  # type: ignore[arg-type]
    )
    renderer = holoviews.renderer("matplotlib")
    collector = FigureCollector(target="inline", on_close="remove")
    with collector:
        renderer.show(basic_spectrogram(df))
    return collector.consume_one()


async def spectrogram_page(request):
    from bokeh.embed import server_document  # type: ignore[reportMissingImports]

    db = request.app.state.db
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
    spectrogram = spectrogram_mpl(db, maybe_int(experiment_id), maybe_int(cluster_id))

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
    from apitofsim.plotting.events import plot_events_cluster

    db = request.app.state.db
    experiment_id = request.query_params.get("experiment")
    cluster_id = request.query_params.get("cluster")
    plot_type = request.query_params.get("plot_type", "beeswarm")
    rescale = request.query_params.get("rescale", "none")

    is_single_pathway = get_is_single_pathway(db, experiment_id, cluster_id)
    fig = plot_events_cluster(db, experiment_id, is_single_pathway, cluster_id, rescale, plot_type)
    return templates.TemplateResponse(
        request,
        "realizations.html",
        {
            "section": "experiment",
            "view": "realizations",
            "route": "realizations",
            "plot_type": plot_type,
            "rescale": rescale,
            "fig": safe_html.figure_html(fig),
            "plot_type_opts": [
                "off-center"
                "off-center-facet",
                "beeswarm",
                "beeswarm-facet",
                "stripplot",
                "stripplot-facet",
                "violinplot",
                "violinplot-facet",
            ],
            "rescale_opts": [
                "none", "equal", "schematic"
            ]
        }
    )


def get_is_single_pathway(db, experiment, cluster):
    if experiment is None or cluster is None:
        return None
    row = db.db.execute(
        """
        select is_single_pathway
        from experiment_cluster_report
        where experiment_run_id = ? and cluster_id = ?
        """,
        (experiment, cluster),
    ).fetchone()
    return row[0] if row else None


def spectrogram_bokeh(db, doc):
    import holoviews
    from apitofsim.plotting.spectrogram import (
        basic_spectrogram,
        get_intensities,
    )
    from bokeh.layouts import layout

    args = doc.session_context.request.arguments

    def arg(name):
        vals = args.get(name)
        return maybe_int(vals[-1].decode()) if vals else None

    experiment = arg("experiment")
    cluster = arg("cluster")
    df = get_intensities(
        db,
        experiment_id=experiment,
        cluster_id=cluster,
        is_single_pathway=get_is_single_pathway(db, experiment, cluster),  # type: ignore[arg-type]
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
    if database_path is None:
        database_path = environ["DATABASE"]

    db = ExperimentDatabase(database_path, readonly=True)
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
                "/bokeh", BokehASGI({"/spectrogram": partial(spectrogram_bokeh, db)}), name="bokeh"
            ),
        ],
    )
    app.state.db = db
    mplbed_starlette.setup(app)
    return app
