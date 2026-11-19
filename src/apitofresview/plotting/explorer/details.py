"""HTML rendering and unit conversion for realization details."""

from bisect import bisect_right
from html import escape
from math import hypot


PAGE_SIZE = 50


def history_page(events, selected_event, page=None):
    """Resolve a page without putting the two states of an event on separate pages."""
    pairs = [
        column
        for name in ("collision_id", "fragmentation_id")
        if (column := events.get_column(name, default=None)) is not None
    ]
    bounds = [0]
    while bounds[-1] < events.height:
        end = min(bounds[-1] + PAGE_SIZE, events.height)
        if end < events.height and any(column[end] is not None for column in pairs):
            end += 1
        bounds.append(end)
    if not events.height:
        return (0, 1, 0, 0)
    if page is None:
        index = (
            events["id"].index_of(selected_event)
            if selected_event is not None
            else None
        )
        page = bisect_right(bounds, index or 0) - 1
    last_page = len(bounds) - 2
    page = min(max(page, 0), last_page)
    return (page, last_page + 1, bounds[page], bounds[page + 1])


def format_value(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return f"{value:.6g}"


def _columns(events, radial, summarise_energy):
    headings = ["Type", "Time (ns)"] + (
        ["Axial dist. (mm)", "Radial dist. (mm)"]
        if radial
        else ["x (mm)", "y (mm)", "z (mm)"]
    )
    motion_columns = (
        [
            ("velocity", "z", "Axial velocity (m/s)"),
            ("velocity", "xy", "Radial speed (m/s)"),
            ("omega", "xyz", "Angular speed (rad/s)"),
        ]
        if radial
        else [
            (key, axis, f"{label} {axis} ({unit})")
            for key, label, unit in (
                ("velocity", "Velocity", "m/s"),
                ("omega", "Angular velocity", "rad/s"),
            )
            for axis in "xyz"
        ]
    )
    motion_columns = [
        column for column in motion_columns if column[0] in events.columns
    ]
    headings += [label for _, _, label in motion_columns]
    detail_columns = [
        ("rot_energy", "Rotational energy (J)"),
        ("vibrational_energy", "Vibrational energy (J)"),
        ("theta", "Collision angle (rad)"),
        ("u_norm", "Collision u_norm (m/s)"),
        ("accepted", "Collision accepted"),
    ]
    if summarise_energy and any(
        key in events.columns for key in ("rot_energy", "vibrational_energy")
    ):
        detail_columns = [
            ("internal_energy", "Internal energy (J)"),
            *detail_columns[2:],
        ]
    detail_columns = [
        (key, label)
        for key, label in detail_columns
        if key == "internal_energy" or key in events.columns
    ]
    headings += [label for _, label in detail_columns]

    return headings, motion_columns, detail_columns


def _cells(
    event,
    radial,
    motion_columns,
    detail_columns,
    clickable=False,
    *,
    rowspan=1,
    after=False,
    shared_selected=False,
):
    kind = escape(event["type"])
    if event["type"] == "fragmentation":
        pathway = escape(
            event.get("event_pathway") or "Unknown fragmentation", quote=True
        )
        daughter = escape(event.get("daughter") or "Unknown charged daughter")
        kind = f'<span title="{pathway}">→ {daughter}</span>'
    if clickable:
        kind = (
            f'<button type="button" aria-label="Inspect event #{escape(str(event["id"]))}">'
            f"{kind}</button>"
        )
    coordinates = (
        [event["z"], event["radial"]]
        if radial
        else [event[axis] for axis in ("x", "y", "z")]
    )
    values = [event["t"] * 1e9, *(value * 1000 for value in coordinates)]
    shared = f' rowspan="{rowspan}"' if rowspan > 1 else ""
    if rowspan > 1 and shared_selected:
        shared += ' style="background:#fde68a"'
    cells = (
        []
        if after
        else [
            f"<td{shared}>{kind}</td>",
            *(f"<td{shared}>{format_value(value)}</td>" for value in values),
        ]
    )
    values = []
    for key, axes, _ in motion_columns:
        vector = event[key]
        values.append(
            None
            if vector is None
            else vector[axes]
            if len(axes) == 1
            else hypot(*(vector[axis] for axis in axes))
        )
    for key, _ in detail_columns:
        if key in ("theta", "u_norm", "accepted"):
            continue
        values.append(_energy(event) if key == "internal_energy" else event[key])
    for index, value in enumerate(values):
        text = format_value(value)
        if after and clickable and index == 0:
            text = (
                f'<button type="button" aria-label="Inspect event #{escape(str(event["id"]))}">'
                f"{text}</button>"
            )
        cells.append(f"<td>{text}</td>")
    if not after:
        cells.extend(
            f"<td{shared}>{format_value(event[key])}</td>"
            for key, _ in detail_columns
            if key in ("theta", "u_norm", "accepted")
        )
    return "".join(cells)


def _energy(event):
    rot, vib = event.get("rot_energy"), event.get("vibrational_energy")
    return None if rot is None or vib is None else rot + vib


def render_details(
    realization,
    events,
    selected_event,
    radial=True,
    summarise_energy=True,
    *,
    window=None,
):
    if realization.is_empty():
        return "<p><em>Click an event to inspect its realization.</em></p>"
    member = realization.row(0, named=True)
    headings, motion_columns, detail_columns = _columns(
        events, radial, summarise_energy
    )
    rows = []

    def pair(event):
        return event.get("collision_id") or event.get("fragmentation_id")

    page, pages, start, stop = window or history_page(events, selected_event)
    records = list(events.slice(start, stop - start).iter_rows(named=True))
    paired = {pair(event) for event in records if pair(event) is not None}
    selected_pair = next(
        (
            pair(event) or selected_event
            for event in records
            if event["id"] == selected_event
        ),
        selected_event,
    )
    for event in records:
        after = pair(event) is not None
        row_class = "explorer-event explorer-after" if after else "explorer-event"
        rows.append(
            f'<tr class="{row_class}" data-event="{escape(str(event["id"]))}" id="event-{escape(str(event["id"]))}" '
            f'style="background:{"#fde68a" if event["id"] == selected_event else "white"}">'
            + _cells(
                event,
                radial,
                motion_columns,
                detail_columns,
                clickable=True,
                rowspan=2 if event["id"] in paired else 1,
                after=after,
                shared_selected=event["id"] == selected_pair,
            )
            + "</tr>"
        )
    context = (
        f"<p>Realization #{member['realization_number']}<br>{escape(member['simulated_pathway'])}</p>"
        if "simulated_pathway" in member
        else f"<p><em>{escape(member.get('fate_label', member['fate']))}</em></p>"
    )
    if "provenance" in events.columns:
        diagnostics = set(events["provenance"].drop_nulls().unique()) - {"persisted"}
        if not events["continuity_valid"].fill_null(False).all():
            diagnostics.add("unresolved continuity")
        if diagnostics:
            context += f"<p>Recording diagnostics: {escape(', '.join(sorted(diagnostics)))}</p>"
    heading = (
        "Details"
        if "simulated_pathway" in member
        else f"Realization #{member['realization_number']}"
    )
    pagination = (
        '<nav aria-label="History pages">'
        f'<button type="button" data-history-page="{page - 1}"'
        f"{' disabled' if page == 0 else ''}>Previous</button> "
        f'<label>Page <input type="number" min="1" max="{pages}" value="{page + 1}"'
        f' data-history-page-input="" style="width:5em"></label> of {pages} '
        '<button type="button" data-history-page-jump="">Go</button> '
        f'<button type="button" data-history-page="{page + 1}"'
        f"{' disabled' if page + 1 == pages else ''}>Next</button>"
        f"<div>Events {start + 1}–{stop} of {events.height}</div></nav>"
        if pages > 1
        else ""
    )
    return (
        "<button id='clear-selection'>Clear selection</button>"
        f"<h3>{heading}</h3>"
        + context
        + pagination
        + '<div style="overflow-x:auto"><table><thead><tr>'
        + "".join(f"<th>{heading}</th>" for heading in headings)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_choices(events, radial=True, summarise_energy=True):
    headings, motion_columns, detail_columns = _columns(
        events, radial, summarise_energy
    )
    rows = []
    for event in events.iter_rows(named=True):
        rows.append(
            f'<tr class="explorer-choice" data-realization="{escape(str(event["realization_id"]))}" '
            f'data-event="{escape(str(event["id"]))}">'
            f'<td><button type="button" aria-label="Inspect event #{escape(str(event["id"]))} '
            f'in realization #{event["realization_number"]}">{event["realization_number"]}</button></td>'
            + _cells(event, radial, motion_columns, detail_columns)
            + "</tr>"
        )
    return (
        "<button id='clear-selection'>Clear selection</button>"
        "<h3>Choose an event</h3>"
        '<div style="overflow-x:auto"><table><thead><tr><th>#</th>'
        + "".join(f"<th>{heading}</th>" for heading in headings)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
