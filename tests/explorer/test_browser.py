"""Explorer browser behavior."""

from contextlib import contextmanager

import pytest
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.resources import INLINE
from playwright.sync_api import expect

from apitofresview.plotting.explorer import build_document, session

from .helpers import cohort, event


@pytest.mark.parametrize("width,height", [(920, 995), (640, 600)])
def test_control_groups_do_not_overlap(monkeypatch, page, tmp_path, width, height):
    frames = cohort([1], event(1, "escape", 1, 1, 5))
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    html = tmp_path / "explorer.html"
    html.write_text(file_html(doc, INLINE, "Explorer"))
    page.set_viewport_size({"width": width, "height": height})
    page.goto(html.as_uri())
    groups = page.locator(".bk-GroupBox")
    expect(groups).to_have_count(8)
    # A shrinking flex item can be shorter than its fieldset, which then
    # paints over the next group even though all controls remain in the DOM.
    page.wait_for_function("""() => {
        function check(root) {
            for (const element of root.querySelectorAll('*')) {
                if (element.matches('.bk-GroupBox')) {
                    const fieldset = element.shadowRoot.querySelector('fieldset');
                    const outer = element.getBoundingClientRect();
                    const inner = fieldset.getBoundingClientRect();
                    if (!outer.height || inner.bottom > outer.bottom + 1)
                        return false;
                }
                if (element.shadowRoot && !check(element.shadowRoot)) return false;
            }
            return true;
        }
        return check(document);
    }""")
    for label in ("Place initial/escape in their own zones", "Spread initial/escape X"):
        control = page.get_by_text(label, exact=True)
        expect(control).to_have_count(1)
        assert control.evaluate("el => el.getBoundingClientRect().width") < 205
    page.get_by_text("Spread initial/escape X", exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(tmp_path / f"controls-{width}.png"))


@contextmanager
def serve_explorer(monkeypatch, frames, db=None):
    """Serve a synthetic cohort with real Bokeh session callbacks."""
    from queue import Queue
    from threading import Thread

    from bokeh.server.server import Server
    from tornado.ioloop import IOLoop

    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    ready = Queue()

    def serve():
        loop = IOLoop()
        try:
            server = Server(
                {"/": lambda doc: build_document(db, doc, 1, 1)},
                io_loop=loop,
                port=0,
                address="127.0.0.1",
            )
            server.start()
            ready.put((loop, server))
            loop.start()
        finally:
            loop.close(all_fds=True)

    thread = Thread(target=serve, daemon=True)
    thread.start()
    loop, server = ready.get(timeout=10)
    try:
        yield f"http://localhost:{server.port}/"
    finally:

        async def stop():
            await server.stop_async()
            loop.stop()

        loop.add_callback(stop)
        thread.join(timeout=10)


@pytest.fixture
def explorer_server(monkeypatch):
    frames = cohort(
        [1, 2],
        event(1, "init", 1, 0, 0),
        event(2, "escape", 1, 1, 5),
        event(3, "init", 2, 0, 0),
        event(4, "escape", 2, 1, 5),
    )
    with serve_explorer(monkeypatch, frames) as url:
        yield url


@pytest.fixture
def overlap_server(monkeypatch):
    frames = cohort(
        [100, 101],
        event(1, "collision", 100, 2e-9, 0.003),
        event(2, "collision", 101, 2e-9, 0.003),
        event(3, "escape", 101, 2e-9, 0.003),
    )
    with serve_explorer(monkeypatch, frames) as url:
        yield url


@pytest.mark.parametrize(
    "details_layout", ["Right sidebar", "Right (half width)", "Bottom"]
)
def test_live_inspection_preserves_plot_and_zoom(page, explorer_server, details_layout):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(explorer_server)
    page.wait_for_function("""() => {
        const plot = window.Bokeh?.documents[0]?.get_model_by_name('events');
        return plot?.inner_width > 0 && plot?.inner_height > 0;
    }""")
    initial = page.evaluate("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        plot.x_range.start = -0.5;
        plot.x_range.end = 9.5;
        const source = plot.renderers.find(r => r.data_source?.data.id?.includes(1)).data_source;
        const view = Bokeh.index.find_one(plot);
        const bounds = view.canvas_view.events_el.getBoundingClientRect();
        const i = source.data.id.indexOf(1);
        return {plot: plot.id, range: plot.x_range.id,
            x: bounds.left + view.frame.x_scale.compute(source.data.x[i]),
            y: bounds.top + view.frame.y_scale.compute(source.data.y[i])};
    }""")
    page.mouse.click(initial.pop("x"), initial.pop("y"))
    expect(
        page.get_by_role("button", name="Inspect event #2", exact=True)
    ).to_be_visible()
    for layout in (details_layout, "Right sidebar", "Bottom", "Right (half width)"):
        page.get_by_role("button", name=layout, exact=True).click()
        for event_id in (2, 1):
            if event_id == 2:
                page.get_by_role("button", name="Inspect event #2", exact=True).click()
            else:
                # Clicking an ordinary cell must select its row too.
                page.locator('.explorer-event[data-event="1"] td').nth(1).click()
            page.wait_for_function(
                """eventId => {
                    const plot = Bokeh.documents[0].get_model_by_name('events');
                    const clicked = plot.renderers.find(r => r.glyph?.size?.value === 20).data_source.data;
                    const source = plot.renderers.find(r => r.data_source?.data.id?.includes(eventId)).data_source.data;
                    const index = source.id.indexOf(eventId);
                    return clicked.position.length === 1 && clicked.position[0] === source.x[index] &&
                        clicked.plot_y[0] === source.y[index];
                }""",
                arg=event_id,
            )
    assert page.evaluate("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        return {plot: plot.id, range: plot.x_range.id, start: plot.x_range.start, end: plot.x_range.end};
    }""") == {**initial, "start": -0.5, "end": 9.5}
    page.get_by_role("button", name="Clear selection", exact=True).click()
    page.wait_for_function("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        return plot.renderers.find(r => r.glyph?.size?.value === 20).data_source.data.position.length === 0;
    }""")


@pytest.mark.parametrize("width", [640, 1280])
def test_live_rate_selection_and_layout(monkeypatch, page, width, tmp_path):
    from .test_rates import RateDatabase, rate_frames

    with serve_explorer(monkeypatch, rate_frames(), RateDatabase()) as url:
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(url)
        page.wait_for_function(
            "() => window.Bokeh?.documents[0]?.get_model_by_name('events')?.inner_width > 0"
        )
        initial = page.evaluate("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('events');
            plot.x_range.start = 1; plot.x_range.end = 2;
            const source = plot.renderers.find(r => r.data_source?.data.id?.includes(1)).data_source;
            source.selected.indices = [source.data.id.indexOf(1)];
            return plot.id;
        }""")
        expect(page.get_by_text("Internal energy (J)", exact=True)).to_be_visible()
        separator = page.get_by_role("separator")
        expect(separator).to_be_visible()
        assert separator.evaluate("el => el.getBoundingClientRect().width") > 100
        page.get_by_text("Show rate constants", exact=True).click()
        page.wait_for_function("""() => {
            const doc = Bokeh.documents[0];
            return doc.get_model_by_name('details_layout_right').active &&
                doc.get_model_by_name('rate_plots').children[0]?.inner_width > 0;
        }""")
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('rate_plots').children[0];
            const label = plot.center.find(m => m.name === 'fragmentation_readout');
            return label?.visible && Math.abs(label.y - 10) < 1e-10 && label.text === '10 s⁻¹';
        }""")
        for show in (False, True):
            page.evaluate(
                "show => Bokeh.documents[0].get_model_by_name('show_readoffs').active = show ? [0] : []",
                show,
            )
            page.wait_for_function(
                """show => {
                const plot = Bokeh.documents[0].get_model_by_name('rate_plots').children[0];
                return plot.center.find(m => m.name === 'fragmentation_readout').visible === show &&
                    plot.center.find(m => m.type === 'Span').visible;
            }""",
                arg=show,
            )
        page.get_by_text("Show all pathways", exact=True).click()
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('rate_plots').children[0].renderers.length === 2"
        )
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('rate_plots').children[0];
            const legend = plot.below.find(m => m.type === 'Legend');
            const view = Bokeh.index.find_one(plot);
            const legendView = view?.renderer_views.get(legend);
            return legendView?.bbox.top >= view.frame.bbox.bottom;
        }""")
        for viewport in (width, 1920 if width == 640 else 640, width):
            page.set_viewport_size({"width": viewport, "height": 900})
            page.wait_for_function(
                """columns => {
                const plot = Bokeh.documents[0].get_model_by_name('rate_plots').children[0];
                return plot.below.find(m => m.type === 'Legend').ncols === columns;
            }""",
                arg=1 if viewport == 640 else 2,
            )
        page.locator('.explorer-event[data-event="2"] td').first.click()
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('rate_plots').children[0];
            return plot.center.some(m => m.type === 'Span' && Math.abs(m.location - 5e-20) < 1e-30);
        }""")
        page.get_by_text("Share plot", exact=True).click()
        page.wait_for_function("""() => {
            const plots = Bokeh.documents[0].get_model_by_name('rate_plots').children.filter(m => m.name === 'pathway_rates');
            return plots.length === 2 && plots.every(p => p.inner_width > 0) && plots[0].x_range === plots[1].x_range;
        }""")
        assert page.evaluate("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('events');
            return [plot.id, plot.x_range.start, plot.x_range.end];
        }""") == [initial, 1, 2]
        page.screenshot(path=str(tmp_path / f"rates-{width}.png"), full_page=True)
        page.get_by_role("button", name="Right sidebar", exact=True).click()
        expect(page.get_by_text("Share plot", exact=True)).not_to_be_visible()
        page.get_by_role("button", name="Bottom", exact=True).click()
        page.get_by_text("Show rate constants", exact=True).click()
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('details_layout_bottom').active"
        )


@pytest.mark.parametrize("width", [640, 1280])
@pytest.mark.parametrize("logarithmic", [True, False])
def test_gas_rate_alignment_and_repeat_selection(
    monkeypatch, page, width, logarithmic, tmp_path
):
    from types import SimpleNamespace

    import numpy as np
    from apitofsim.api import Q_

    from apitofresview.plotting.explorer import rates

    from .test_rates import RateDatabase, rate_frames

    monkeypatch.setattr(rates, "skimmer", lambda config: None)
    monkeypatch.setattr(
        rates,
        "collision_profile",
        lambda *args, **kwargs: SimpleNamespace(
            frequency=lambda z, velocity, **kwargs: Q_(
                np.where(z < 2, 1e8, 1e4) * (1 + np.linalg.norm(velocity)), "1/s"
            )
        ),
    )
    with serve_explorer(monkeypatch, rate_frames(), RateDatabase()) as url:
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(url)
        page.get_by_text("Gas collision rate", exact=True).click()
        if not logarithmic:
            page.get_by_text("Logarithmic rates", exact=True).filter(
                visible=True
            ).click()
        page.wait_for_function("""() => {
            const panels = Bokeh.documents[0].get_model_by_name('explorer_plots').children;
            const plots = panels.filter(p => ['collision_rates', 'schematic', 'events'].includes(p.name));
            if (plots.length !== 3) return false;
            const boxes = plots.map(p => {
                const view = Bokeh.index.find_one(p);
                if (!view?.frame?.bbox || !p.inner_width) return null;
                const rect = view.canvas_view.events_el.getBoundingClientRect();
                return [rect.left + view.frame.bbox.left, rect.left + view.frame.bbox.right];
            });
            return boxes.every(b => b && Math.abs(b[0] - boxes[0][0]) < 1 && Math.abs(b[1] - boxes[0][1]) < 1);
        }""")
        assert page.evaluate("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('schematic');
            return plot.renderers[0].data_source.data.right.at(-1) === plot.x_range.end;
        }""")
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates').center.find(m => m.name === 'collision_velocity').text === 'v = 0 m/s'"
        )
        page.wait_for_function("""() => {
            const panels = Bokeh.documents[0].get_model_by_name('explorer_plots').children;
            const gas = Bokeh.index.find_one(panels.find(p => p.name === 'collision_rates'));
            const events = Bokeh.index.find_one(panels.find(p => p.name === 'events'));
            const bottom = gas.canvas_view.events_el.getBoundingClientRect().top + gas.frame.bbox.bottom;
            const top = events.canvas_view.events_el.getBoundingClientRect().top + events.frame.bbox.top;
            return Math.abs(top - bottom) < 1;
        }""")
        page.evaluate("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('events');
            const source = plot.renderers.find(r => r.data_source?.data.id?.includes(1)).data_source;
            source.selected.indices = [source.data.id.indexOf(1)];
        }""")
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates').center.find(m => m.name === 'collision_velocity').text === 'v = 7.07107 m/s'"
        )
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates');
            const marker = plot.center.find(m => m.name === 'collision_event_marker');
            const label = plot.center.find(m => m.name === 'collision_event_readout');
            return marker.visible && marker.location === 0 && label.visible && label.x === 0 && label.y > 1e8;
        }""")
        for show in (False, True):
            page.evaluate(
                "show => Bokeh.documents[0].get_model_by_name('show_readoffs').active = show ? [0] : []",
                show,
            )
            page.wait_for_function(
                """show => {
                const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates');
                return plot.center.find(m => m.name === 'collision_event_readout').visible === show &&
                    plot.center.find(m => m.name === 'collision_event_marker').visible;
            }""",
                arg=show,
            )

        def bounds():
            return page.evaluate("""() => {
                const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates');
                return [plot.y_range.start, plot.y_range.end];
            }""")

        fitted = bounds()
        page.evaluate("""() => {
            const range = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates').y_range;
            range.start -= 100; range.end *= 10;
        }""")
        for _ in range(3):
            page.get_by_role("button", name="Inspect event #1", exact=True).click()
            page.wait_for_function(
                """expected => {
                const range = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'collision_rates').y_range;
                return range.start === expected[0] && range.end === expected[1];
            }""",
                arg=fitted,
            )
        assert bounds() == fitted
        page.evaluate("""() => {
            const widgets = [...Bokeh.documents[0]._all_models.values()];
            widgets.find(m => m.type === 'CheckboxGroup' && m.labels[0] === 'Facet pathways').active = [0];
        }""")
        page.wait_for_function("""() => {
            const panels = Bokeh.documents[0].get_model_by_name('explorer_plots').children;
            const gas = Bokeh.index.find_one(panels.find(p => p.name === 'collision_rates'));
            const events = Bokeh.index.find_one(panels.find(p => p.name === 'events'));
            const bottom = gas.canvas_view.events_el.getBoundingClientRect().top + gas.frame.bbox.bottom;
            const top = events.canvas_view.events_el.getBoundingClientRect().top + events.frame.bbox.top;
            return top - bottom > 10;
        }""")
        page.screenshot(
            path=str(tmp_path / f"gas-rate-{width}-{logarithmic}.png"), full_page=True
        )


def test_schematic_edges_drag_and_update_events(page, explorer_server):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(explorer_server)
    page.wait_for_function(
        """() => [...Bokeh.documents[0]?._all_models.values() ?? []].some(p => p.name === 'schematic' && p.inner_width > 0 && Bokeh.index.find_one(p)?.el?.isConnected)"""
    )

    def widths():
        return page.evaluate("""() => {
            const plot = [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1);
            const data = plot.renderers[0].data_source.data;
            return data.right.map((right, i) => right - data.left[i]);
        }""")

    def x_range():
        return page.evaluate("""() => {
            const plot = [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1);
            return [plot.x_range.start, plot.x_range.end];
        }""")

    def drag(boundary, pixels):
        point = page.evaluate(
            """boundary => {
            const plot = [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1);
            const view = Bokeh.index.find_one(plot);
            const bounds = view.canvas_view.events_el.getBoundingClientRect();
            const x = plot.renderers[0].data_source.data.right[boundary];
            return {x: bounds.left + view.frame.x_scale.compute(x),
                y: bounds.top + view.frame.y_scale.compute(0.5)};
        }""",
            boundary,
        )
        page.mouse.move(point["x"], point["y"])
        page.mouse.down()
        page.mouse.move(point["x"] + pixels, point["y"], steps=5)
        page.mouse.up()
        page.wait_for_function(
            """() => [...Bokeh.documents[0]._all_models.values()].some(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected && p.renderers[0].data_source.data.right.length === 7)"""
        )

    before = widths()
    original_range = x_range()
    center = page.evaluate("""() => {
        const plot = [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1);
        const view = Bokeh.index.find_one(plot);
        const bounds = view.canvas_view.events_el.getBoundingClientRect();
        const data = plot.renderers[0].data_source.data;
        return {x: bounds.left + view.frame.x_scale.compute((data.left[1] + data.right[1]) / 2),
            y: bounds.top + view.frame.y_scale.compute(0.5)};
    }""")
    page.mouse.move(center["x"], center["y"])
    page.mouse.down()
    page.mouse.move(center["x"] + 40, center["y"], steps=5)
    page.mouse.up()
    assert x_range() == pytest.approx(original_range)
    assert widths() == pytest.approx(before)

    drag(0, 25)
    page.wait_for_function(
        """() => [...Bokeh.documents[0]._all_models.values()].some(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected && p.renderers[0].data_source.data.right[0] > 0)"""
    )
    after = widths()
    assert after[0] > before[0]
    assert all(after[i] < before[i] for i in range(1, 6))
    assert after[6] == pytest.approx(before[6])
    assert x_range() == pytest.approx(original_range)
    expect(page.get_by_label("Sizing")).to_have_value("Custom")

    for boundary, pixels, grow, shrink, fixed in (
        (0, -20, (1, 2, 3, 4, 5), (0,), (6,)),
        (1, 20, (1,), (2, 3, 4, 5), (0, 6)),
        (2, 20, (1, 2), (3, 4, 5), (0, 6)),
        (2, -20, (3, 4, 5), (1, 2), (0, 6)),
        (3, 20, (3,), (4,), (0, 1, 2, 5, 6)),
        (4, 20, (4,), (5,), (0, 1, 2, 3, 6)),
        (5, 20, (1, 2, 3, 4, 5), (6,), (0,)),
        (5, -20, (6,), (1, 2, 3, 4, 5), (0,)),
    ):
        old_id = page.evaluate(
            """() => [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1)?.id"""
        )
        page.get_by_label("Sizing").select_option(label="Equal chambers")
        expect(page.get_by_label("Sizing")).to_have_value("Equal chambers")
        page.wait_for_function(
            """oldId => [...Bokeh.documents[0]._all_models.values()].some(p => p.name === 'schematic' && p.id !== oldId && Bokeh.index.find_one(p)?.el?.isConnected)""",
            arg=old_id,
        )
        before = widths()
        drag(boundary, pixels)
        expect(page.get_by_label("Sizing")).to_have_value("Custom")
        after = widths()
        assert sum(after) == pytest.approx(sum(before))
        for index in grow:
            assert after[index] > before[index]
        for index in shrink:
            assert after[index] < before[index]
        for index in fixed:
            assert after[index] == pytest.approx(before[index])

    old_id = page.evaluate(
        """() => [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1)?.id"""
    )
    page.get_by_label("Sizing").select_option(label="Equal chambers")
    page.wait_for_function(
        """oldId => [...Bokeh.documents[0]._all_models.values()].some(p => p.name === 'schematic' && p.id !== oldId && Bokeh.index.find_one(p)?.el?.isConnected)""",
        arg=old_id,
    )
    drag(3, 20)
    expect(page.get_by_label("Sizing")).to_have_value("Custom")
    before = widths()
    old_id = page.evaluate(
        """() => [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1)?.id"""
    )
    drag(0, 20)
    page.wait_for_function(
        """oldId => [...Bokeh.documents[0]._all_models.values()].some(p => p.name === 'schematic' && p.id !== oldId && Bokeh.index.find_one(p)?.el?.isConnected)""",
        arg=old_id,
    )
    after = widths()
    assert after[3] / before[3] == pytest.approx(after[4] / before[4])
    assert after[4] / before[4] == pytest.approx(after[5] / before[5])

    # A large drag clamps the terminal zone to exactly five screen pixels.
    drag(5, 2000)
    expect(page.get_by_label("Sizing")).to_have_value("Custom")
    pixels = page.evaluate("""() => {
        const plot = [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1);
        const data = plot.renderers[0].data_source.data;
        return (data.right[6] - data.left[6]) * plot.inner_width / (plot.x_range.end - plot.x_range.start);
    }""")
    assert pixels == pytest.approx(5, abs=0.1)
    label = page.evaluate("""() => {
        const plot = [...Bokeh.documents[0]._all_models.values()].filter(p => p.name === 'schematic' && Bokeh.index.find_one(p)?.el?.isConnected).at(-1);
        return plot.center.filter(item => typeof item.text === 'string').at(-1).text;
    }""")
    assert label == ""
    page.get_by_label("X coordinate").select_option(label="Axial distance (mm)")
    expect(page.get_by_label("Sizing")).to_be_hidden()
    page.get_by_label("X coordinate").select_option(label="Schematic")
    expect(page.get_by_label("Sizing")).to_be_visible()
    expect(page.get_by_label("Sizing")).to_have_value("Custom")


def test_overlapping_hover_choice_and_beeswarm_direct_selection(page, overlap_server):
    page.set_viewport_size({"width": 640, "height": 700})
    page.goto(overlap_server)
    page.get_by_label("X coordinate").select_option(label="Axial distance (mm)")
    page.get_by_label("Y coordinate").select_option(label="Radial distance (mm)")

    def point_for(event_id, beeswarm=False):
        page.wait_for_function(
            """beeswarm => {
            try {
                const center = Bokeh.documents[0].roots()[0].children[0].children[1].children[0];
                const plot = center.children[2].children.find(p => p.name === 'events');
                const source = plot?.renderers.find(r => r.data_source?.data.id?.includes(1))?.data_source;
                return plot?.inner_width > 0 && plot?.inner_height > 0 && source &&
                    source.data.x.every(x => x === 3) &&
                    (beeswarm ? source.data.y.some(y => y !== 5000)
                              : source.data.y.every(y => y === 5000));
            } catch { return false; }
        }""",
            arg=beeswarm,
        )
        return page.evaluate(
            """event_id => {
            const center = Bokeh.documents[0].roots()[0].children[0].children[1].children[0];
            const plot = center.children[2].children.find(p => p.name === 'events');
            const source = plot.renderers.find(r => r.data_source?.data.id?.includes(event_id)).data_source;
            const view = Bokeh.index.find_one(plot);
            const bounds = view.canvas_view.events_el.getBoundingClientRect();
            const i = source.data.id.indexOf(event_id);
            return {x: bounds.left + view.frame.x_scale.compute(source.data.x[i]),
                y: bounds.top + view.frame.y_scale.compute(source.data.y[i])};
        }""",
            event_id,
        )

    point = point_for(1)
    page.mouse.move(point["x"], point["y"])
    expect(page.locator(".explorer-hover-row")).to_have_count(3)
    assert page.locator(".explorer-hover-row").evaluate_all(
        "rows => rows.map(row => row.querySelector('span').textContent)"
    ) == ["1", "2", "2"]
    expect(page.locator(".explorer-hover-head:visible")).to_have_count(1)
    assert page.locator(".explorer-hover-row").first.evaluate("""row => {
        const bounds = row.getRootNode().host.getBoundingClientRect();
        return bounds.left >= 0 && bounds.right <= window.innerWidth;
    }""")
    page.mouse.click(point["x"], point["y"])
    expect(page.get_by_text("Choose an event", exact=True)).to_be_visible()
    expect(page.locator(".explorer-choice")).to_have_count(3)
    assert page.locator(".explorer-choice").first.evaluate("""row => {
        const wrapper = row.closest('table').parentElement;
        return wrapper.getBoundingClientRect().right <= window.innerWidth + 1;
    }""")
    page.get_by_role("button", name="Bottom", exact=True).click()
    page.get_by_role("button", name="Inspect event #3 in realization #2").click()
    expect(page.get_by_text("Realization #2", exact=True)).to_be_visible()
    expect(
        page.get_by_role("button", name="Inspect event #3", exact=True)
    ).to_be_visible()

    page.get_by_role("button", name="Clear selection", exact=True).click()
    page.evaluate("""() => {
        const control = [...Bokeh.documents[0].all_models]
            .find(model => model.type === 'Select' && model.title === 'Y coordinate');
        control.value = 'beeswarm';
    }""")
    point = point_for(1, beeswarm=True)
    page.mouse.click(point["x"], point["y"])
    expect(page.get_by_text("Realization #1", exact=True)).to_be_visible()
    expect(page.get_by_text("Choose an event", exact=True)).to_have_count(0)


@pytest.mark.parametrize("width", [640, 1280])
def test_details_layout_geometry(page, explorer_server, width, tmp_path):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(explorer_server)
    page.get_by_role("button", name="☷", exact=True).click()
    page.evaluate("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        const source = plot.renderers.find(r => r.data_source?.data.id?.includes(1)).data_source;
        source.selected.indices = [source.data.id.indexOf(1)];
    }""")
    expect(page.get_by_text("Realization #1", exact=True)).to_be_visible()
    page.get_by_role("button", name="Right (half width)", exact=True).click()

    def bounds():
        return page.evaluate("""() => {
            const body = Bokeh.documents[0].roots()[0].children[0];
            const [center, right] = body.children[1].children;
            const rect = model => {
                const r = Bokeh.index.find_one(model).el.getBoundingClientRect();
                return {x: r.x, y: r.y, width: r.width, height: r.height};
            };
            return {center: rect(center), right: rect(right),
                    controls: body.children[0].children[1].visible};
        }""")

    page.wait_for_function("""() => {
        const [center, right] = Bokeh.documents[0].roots()[0].children[0].children[1].children;
        return right.width == null && Bokeh.index.find_one(right).el.getBoundingClientRect().width > 100;
    }""")
    geometry = bounds()
    assert not geometry["controls"]
    assert geometry["center"]["width"] == pytest.approx(
        geometry["right"]["width"], abs=2
    )
    assert (
        geometry["right"]["x"] >= geometry["center"]["x"] + geometry["center"]["width"]
    )
    page.screenshot(path=str(tmp_path / f"details-right-{width}.png"))
    page.get_by_role("button", name="Bottom", exact=True).click()
    page.wait_for_function("""() => {
        const workspace = Bokeh.documents[0].roots()[0].children[0].children[1];
        if (workspace.type !== 'Column') return false;
        const [center, right] = workspace.children.map(m => Bokeh.index.find_one(m)?.el?.getBoundingClientRect());
        return center && right && right.top >= center.bottom;
    }""")
    geometry = bounds()
    assert geometry["center"]["width"] == pytest.approx(
        geometry["right"]["width"], abs=2
    )
    assert geometry["right"]["height"] < geometry["center"]["height"]
    expect(page.get_by_text("Realization #1", exact=True)).to_be_visible()
    page.screenshot(path=str(tmp_path / f"details-bottom-{width}.png"), full_page=True)
    page.get_by_role("button", name="Hide details ↓", exact=True).click()
    page.get_by_role("button", name="☷", exact=True).click()
    expect(page.get_by_role("button", name="Right sidebar", exact=True)).to_have_class(
        "bk-btn bk-btn-default bk-active"
    )
    page.get_by_role("button", name="Right (half width)", exact=True).click()
    page.get_by_role("button", name="»", exact=True).click()
    expect(page.get_by_label("X coordinate")).to_be_visible()
    if width >= 900:
        expect(page.get_by_role("button", name="Bottom", exact=True)).to_be_visible()
    else:
        expect(page.get_by_role("button", name="☷", exact=True)).to_be_visible()
    assert not errors
