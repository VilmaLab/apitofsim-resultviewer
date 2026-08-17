from os import environ
from pathlib import Path
from urllib.parse import urlencode
from threading import local

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware

from bokeh.server.asgi import BokehASGI
from apitofsim.workflow.db import ExperimentDatabase
from apitofsim.plotting import get_report

from mplbed import safe_html, mplbed_starlette, FigureCollector


database_path = environ["DATABASE"]
db = ExperimentDatabase(database_path, readonly=True)

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


def pathway_type_lbl(is_single_pathway):
    return "single pathway" if is_single_pathway else "multi-pathway"


def get_experiment_choices():
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


def get_cluster_choices(experiment):
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
    except (TypeError, ValueError):
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


def selected_cluster(request, clusters):
    cluster = maybe_int(request.query_params.get("cluster"))
    return cluster if cluster in {value for _, value in clusters} else None


def nav_context(request):
    """Values the navigation chrome in base.html needs on every page."""
    experiment = maybe_int(request.query_params.get("experiment"))
    report = selected_report(request, experiment)
    clusters = get_cluster_choices(experiment)
    cluster = selected_cluster(request, clusters)
    # Parameters the experiment picker carries over, minus anything we dropped.
    query_params = dict(request.query_params)
    query_params.pop("report", None)
    query_params.pop("cluster", None)
    if report:
        query_params["report"] = report
    return {
        "experiments": get_experiment_choices(),
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


def head_context(request):
    return {
        "mpl_head": safe_html.head_content(core=True),
    }


templates = Jinja2Templates(directory="templates", context_processors=[head_context, nav_context])


async def overview(request):
    return templates.TemplateResponse(request, "overview.html", {
        "section": "overview",
        "route": "overview",
    })


async def experiment(request):
    return templates.TemplateResponse(request, "experiment.html", {
        "section": "experiment",
        "route": "experiment",
    })


async def comparison(request):
    return templates.TemplateResponse(request, "comparison.html", {
        "section": "comparison",
        "route": "comparison",
    })


async def report(request):
    experiment = maybe_int(request.query_params.get("experiment"))
    report = selected_report(request, experiment)
    if report:
        df = get_report(db, report)
        report_data = df.to_json(orient="records")
    else:
        report_data = None
    return templates.TemplateResponse(request, "report.html", {
        "report_data": report_data,
        "section": "experiment" if experiment is not None else "overview",
        "view": "report",
        "route": "report",
    })


async def survivals(request):
    return templates.TemplateResponse(request, "survivals.html", {
        "section": "experiment",
        "view": "survivals",
        "route": "survivals",
    })


async def cluster(request):
    return templates.TemplateResponse(request, "cluster.html", {
        "section": "experiment",
        "view": "cluster",
        "route": "cluster",
    })


def spectrogram_mpl(experiment, cluster):
    from apitofsim.plotting import basic_spectrogram, get_intensities
    import holoviews
    df = get_intensities(
        db,
        experiment_id=experiment,
        cluster_id=cluster,
        is_single_pathway=get_is_single_pathway(experiment, cluster),
    )
    renderer = holoviews.renderer('matplotlib')
    collector = FigureCollector(target="inline", on_close="remove")
    with collector:
        renderer.show(basic_spectrogram(df))
    return collector.consume_one()


async def spectrogram_page(request):
    from bokeh.embed import server_document
    url = str(request.url_for("bokeh", path="/spectrogram"))
    experiment_id = request.query_params.get("experiment")
    cluster_id = request.query_params.get("cluster")
    script = server_document(url, arguments={
        "experiment": experiment_id,
        "cluster": cluster_id,
    })
    spectrogram = spectrogram_mpl(maybe_int(experiment_id), maybe_int(cluster_id))

    return templates.TemplateResponse(request, "spectrogram.html", {
        "section": "experiment",
        "view": "spectrogram",
        "route": "spectrogram",
        "spectrogram_bokeh": script,
        "spectrogram_mpl": spectrogram,
    })


async def realizations(request):
    return templates.TemplateResponse(request, "realizations.html", {
        "section": "experiment",
        "view": "realizations",
        "route": "realizations",
    })


def get_is_single_pathway(experiment, cluster):
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


def spectrogram_bokeh(doc):
    from panel import layout
    from bokeh.layouts import layout
    import holoviews
    from apitofsim.plotting import basic_spectrogram, get_intensities

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
        is_single_pathway=get_is_single_pathway(experiment, cluster),
    )
    renderer = holoviews.renderer('bokeh').instance(mode='server')
    plot = renderer.get_plot(basic_spectrogram(df), doc)
    root = layout(
        [[plot.state]], sizing_mode='fixed'
    )
    doc.add_root(root)


app = Starlette(
    debug=True,
    routes=[
        Route('/', overview, name="overview"),
        Route('/experiment', experiment, name="experiment"),
        Route('/report', report, name="report"),
        Route('/experiment/survivals', survivals, name="survivals"),
        Route('/experiment/cluster', cluster, name="cluster"),
        Route('/experiment/cluster/spectrogram', spectrogram_page, name="spectrogram"),
        Route('/experiment/cluster/realizations', realizations, name="realizations"),
        Route('/comparison', comparison, name="comparison"),
        Mount('/static', StaticFiles(directory='static'), name='static'),
        Mount('/bokeh', BokehASGI({"/spectrogram": spectrogram_bokeh}), name="bokeh"),
    ],
)
mplbed_starlette.setup(app)
