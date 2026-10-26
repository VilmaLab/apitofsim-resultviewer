"""HTML rendering and unit conversion for realization details."""

from html import escape
from math import hypot


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


def _cells(event, radial, motion_columns, detail_columns, clickable=False):
    kind = escape(event["type"])
    if clickable:
        kind = (
            f'<button type="button" aria-label="Inspect event #{event["id"]}">'
            f"{kind}</button>"
        )
    coordinates = (
        [event["z"], event["radial"]]
        if radial
        else [event[axis] for axis in ("x", "y", "z")]
    )
    values = [event["t"] * 1e9, *(value * 1000 for value in coordinates)]
    for key, axes, _ in motion_columns:
        vector = event[key]
        values.append(
            None
            if vector is None
            else vector[axes]
            if len(axes) == 1
            else hypot(*(vector[axis] for axis in axes))
        )
    return (
        f"<td>{kind}</td>"
        + "".join(f"<td>{format_value(value)}</td>" for value in values)
        + "".join(
            f"<td>{format_value(_energy(event) if key == 'internal_energy' else event[key])}</td>"
            for key, _ in detail_columns
        )
    )


def _energy(event):
    rot, vib = event.get("rot_energy"), event.get("vibrational_energy")
    return None if rot is None or vib is None else rot + vib


def render_details(
    realization, events, selected_event, radial=True, summarise_energy=True
):
    if realization.is_empty():
        return "<p><em>Click an event to inspect its realization.</em></p>"
    member = realization.row(0, named=True)
    headings, motion_columns, detail_columns = _columns(
        events, radial, summarise_energy
    )
    rows = []
    for event in events.iter_rows(named=True):
        rows.append(
            f'<tr class="explorer-event" data-event="{event["id"]}" id="event-{event["id"]}" '
            f'style="background:{"#fde68a" if event["id"] == selected_event else "white"}">'
            + _cells(event, radial, motion_columns, detail_columns, clickable=True)
            + "</tr>"
        )
    return (
        "<button id='clear-selection'>Clear selection</button>"
        f"<h3>Realization #{member['realization_number']}</h3>"
        f"<p><em>{escape(member['fate'])}</em></p>"
        '<div style="overflow-x:auto"><table><thead><tr>'
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
            f'<tr class="explorer-choice" data-realization="{event["realization_id"]}" '
            f'data-event="{event["id"]}">'
            f'<td><button type="button" aria-label="Inspect event #{event["id"]} '
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
