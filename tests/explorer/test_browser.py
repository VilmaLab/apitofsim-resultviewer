"""Explorer browser behavior."""

from contextlib import contextmanager

import pytest
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.resources import INLINE
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, expect

from apitofresview.plotting.explorer import build_document, session

from .helpers import cohort, event


@pytest.mark.parametrize("width,height", [(920, 995), (640, 600), (955, 300)])
def test_control_groups_do_not_overlap(monkeypatch, page, tmp_path, width, height):
    frames = cohort([1], event(1, "escape", 1, 1, 5))
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    html = tmp_path / "explorer.html"
    html.write_text(file_html(doc, INLINE, "Explorer"))
    page.set_viewport_size({"width": width, "height": height})
    page.goto(html.as_uri())
    groups = page.locator(".bk-GroupBox").filter(visible=True)
    expect(groups).to_have_count(8)
    # A shrinking flex item can be shorter than its fieldset, which then
    # paints over the next group even though all controls remain in the DOM.
    page.wait_for_function("""() => {
        function check(root) {
            for (const element of root.querySelectorAll('*')) {
                if (element.matches('.bk-GroupBox') && element.checkVisibility()) {
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
def serve_explorer(monkeypatch, frames, db=None, *, layout="viewport"):
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
                {"/": lambda doc: build_document(db, doc, 1, 1, layout=layout)},
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
        // Zooming asynchronously repacks terminal markers on the server.
        // Select directly so this layout test does not race their coordinates.
        source.selected.indices = [source.data.id.indexOf(1)];
        return {plot: plot.id, range: plot.x_range.id};
    }""")
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
        expect(separator).to_be_hidden()
        page.get_by_text("Show rate constants", exact=True).click()
        expect(separator).to_be_visible()
        assert separator.evaluate("el => el.getBoundingClientRect().width") > 100
        assert (
            separator.bounding_box()["y"]
            > page.get_by_text("Show rate constants", exact=True).bounding_box()["y"]
        )
        page.wait_for_function("""() => {
            const doc = Bokeh.documents[0];
            const rect = name => Bokeh.index.find_one(doc.get_model_by_name(name)).el.getBoundingClientRect();
            const rates = rect('detail_rates'), pathways = rect('detail_pathways');
            return rates.width > 0 && pathways.left >= rates.right && Math.abs(rates.top - pathways.top) < 1;
        }""")
        for expected in ([], [0]):
            page.get_by_text("Show readoff values", exact=True).filter(
                visible=True
            ).click()
            page.wait_for_function(
                """expected => {
                const doc = Bokeh.documents[0];
                return JSON.stringify(doc.get_model_by_name('show_readoffs').active) === JSON.stringify(expected);
            }""",
                arg=expected,
            )
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
            assert_horizontal_fit(page)
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
    page.wait_for_function("""() => {
        const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
        return getComputedStyle(Bokeh.index.find_one(right).el).borderTopWidth === '1px';
    }""")
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
                    border: getComputedStyle(Bokeh.index.find_one(right).el).borderTopWidth,
                    controls: body.children[0].children[1].visible};
        }""")

    page.wait_for_function("""() => {
        const [center, right] = Bokeh.documents[0].roots()[0].children[0].children[1].children;
        return right.width == null && Bokeh.index.find_one(right).el.getBoundingClientRect().width > 100;
    }""")
    geometry = bounds()
    assert geometry["border"] == "0px"
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
    assert geometry["border"] == "1px"
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


def test_paired_collision_rows_and_split_controls(monkeypatch, page):
    from .helpers import collision_cohort

    with serve_explorer(monkeypatch, collision_cohort()) as url:
        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(url)
        page.get_by_text("Separate rejected collisions", exact=True).click()
        expect(page.get_by_text("Accepted collisions", exact=True)).to_be_visible()
        expect(page.get_by_text("Rejected collisions", exact=True)).to_be_visible()
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
            const source = plot?.renderers.find(r => r.data_source?.data.id?.includes(1))?.data_source;
            if (!source || !plot.inner_width) return false;
            source.selected.indices = [source.data.id.indexOf(1)];
            return true;
        }""")
        before = page.locator("#event-1")
        after = page.locator("#event-2")
        expect(before.locator('td[rowspan="2"]')).to_have_count(7)
        # Event names must stay on one line even when the numeric table scrolls.
        assert (
            before.locator("td").first.evaluate("""cell => {
            const range = document.createRange();
            range.selectNodeContents(cell.querySelector('button'));
            return range.getClientRects().length;
        }""")
            == 1
        )
        expect(
            after.get_by_role("button", name="Inspect event #2", exact=True)
        ).to_be_visible()
        highlight = page.evaluate("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
            return plot.renderers.find(r => r.glyph?.size?.value === 20).data_source.data;
        }""")
        # Exercise the actual delegated click handler on an ordinary after-state cell.
        after.locator("td").last.click()
        expect(after).to_have_attribute("style", "background:#fde68a")
        expect(before).to_have_attribute("style", "background:white")
        for cell in before.locator('td[rowspan="2"]').all():
            expect(cell).to_have_css("background-color", "rgb(253, 230, 138)")
        assert (
            page.evaluate("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
            return plot.renderers.find(r => r.glyph?.size?.value === 20).data_source.data;
        }""")
            == highlight
        )
        page.get_by_role("button", name="Inspect event #1", exact=True).click()
        expect(before).to_have_attribute("style", "background:#fde68a")
        page.get_by_role("button", name="Inspect event #3", exact=True).click()
        expect(before).to_have_attribute("style", "background:white")
        expect(before.locator('td[rowspan="2"][style]')).to_have_count(0)
        page.get_by_text("Rejected collisions", exact=True).click()
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
            return !plot.renderers.some(r => r.data_source?.data.id?.includes(3));
        }""")
        expect(page.locator("#event-3")).to_have_count(1)


@pytest.mark.parametrize(
    "width,fullscreen", [(640, False), (1280, False), (1280, True)]
)
def test_explorer_panels_scroll_independently(page, explorer_server, width, fullscreen):
    page.set_viewport_size({"width": width, "height": 700})
    page.goto(explorer_server)
    if fullscreen:
        page.get_by_role("button", name="Fullscreen", exact=True).click()
        page.get_by_role("button", name="»", exact=True).click()
    else:
        # Account for the application's navigation above the embedded explorer.
        page.evaluate("""document.body.style.cssText =
            'margin: 0; padding: 120px 6px 6px; box-sizing: border-box;'""")
    page.get_by_text("Cumulative", exact=True).click()
    page.get_by_text("Bar chart", exact=True).click()
    page.get_by_role("button", name="☷", exact=True).click()

    geometry = """() => {
        const frame = Bokeh.documents[0].roots()[0];
        const [left, workspace] = frame.children[0].children;
        const [center, right] = workspace.children;
        const el = model => Bokeh.index.find_one(model).el;
        const rect = model => el(model).getBoundingClientRect();
        return {frameBottom: rect(frame).bottom, workspaceHeight: rect(workspace).height,
            workspaceBottom: rect(workspace).bottom,
            bottom: rect(right).bottom, height: rect(right).height,
            centerBottom: rect(center).bottom, centerHeight: el(center).clientHeight,
            centerContent: el(center).scrollHeight, plotScroll: el(center).scrollTop,
            detailHeight: el(right.children[1]).clientHeight,
            detailContent: el(right.children[1]).scrollHeight,
            detailScroll: el(right.children[1]).scrollTop,
            leftTop: rect(left).top, pageScroll: window.scrollY};
    }"""

    def set_details(html):
        page.evaluate(
            """html => {
            const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
            right.children[1].children[0].text = html;
        }""",
            html,
        )

    set_details('<div style="height: 1600px; width: 1200px">Long details</div>')
    for layout in ("Right sidebar", "Right (half width)", "Bottom"):
        page.get_by_role("button", name=layout, exact=True).click()
        page.wait_for_function(
            """layout => {
                const workspace = Bokeh.documents[0].roots()[0].children[0].children[1];
                const [center, right] = workspace.children;
                const el = model => Bokeh.index.find_one(model)?.el;
                const bottom = layout === 'Bottom';
                if ((workspace.type === 'Column') !== bottom) return false;
                if ((right.width != null) !== (layout === 'Right sidebar')) return false;
                if (!el(center) || !el(right)) return false;
                return el(center).scrollHeight > el(center).clientHeight &&
                    (!bottom || Math.abs(el(right).getBoundingClientRect().bottom -
                        el(workspace).getBoundingClientRect().bottom) <= 1);
            }""",
            arg=layout,
        )
        page.wait_for_function("""() => {
            const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
            const el = Bokeh.index.find_one(right.children[1]).el;
            return el.scrollHeight > el.clientHeight;
        }""")
        before = page.evaluate(geometry)
        assert before["frameBottom"] <= 700
        assert before["centerBottom"] <= before["frameBottom"] + 1
        page.evaluate("""() => {
            const center = Bokeh.documents[0].roots()[0].children[0].children[1].children[0];
            Bokeh.index.find_one(center).el.scrollTop = 20;
        }""")
        after = page.evaluate(geometry)
        assert after["plotScroll"] == 20
        for key in ("bottom", "leftTop", "detailScroll", "pageScroll"):
            assert after[key] == before[key]

        page.evaluate("""() => {
            const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
            Bokeh.index.find_one(right.children[1]).el.scrollTop = 40;
        }""")
        details_scrolled = page.evaluate(geometry)
        assert details_scrolled["detailScroll"] == 40
        assert details_scrolled["plotScroll"] == after["plotScroll"]
        assert details_scrolled["pageScroll"] == after["pageScroll"]
        assert page.evaluate("""() => {
            const root = document.scrollingElement;
            return root.scrollHeight <= innerHeight && root.scrollWidth <= innerWidth;
        }""")

    set_details("Short details")
    page.wait_for_function("""() => {
        const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
        return Bokeh.index.find_one(right).el.getBoundingClientRect().height < 150;
    }""")
    short = page.evaluate(geometry)
    assert short["bottom"] == pytest.approx(short["workspaceBottom"], abs=1)
    set_details('<div style="height: 1600px">Long details</div>')
    page.wait_for_function("""() => {
        const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
        const el = Bokeh.index.find_one(right.children[1]).el;
        return el.scrollHeight > el.clientHeight;
    }""")
    long = page.evaluate(geometry)
    assert long["height"] == pytest.approx(long["workspaceHeight"] / 2, abs=1)
    assert long["bottom"] == pytest.approx(short["bottom"], abs=1)
    page.evaluate("""() => {
        const right = Bokeh.documents[0].roots()[0].children[0].children[1].children[1];
        Bokeh.index.find_one(right.children[1]).el.scrollTop = 20;
    }""")
    scrolled = page.evaluate(geometry)
    assert scrolled["detailScroll"] == 20
    assert scrolled["plotScroll"] == long["plotScroll"]
    assert scrolled["pageScroll"] == long["pageScroll"]
    if fullscreen:
        page.evaluate("document.exitFullscreen()")
        page.wait_for_function("document.fullscreenElement === null")
        page.get_by_role("button", name="☷", exact=True).click()
        page.get_by_role("button", name="Bottom", exact=True).click()
    page.set_viewport_size({"width": width, "height": 600})
    page.wait_for_function("""() => {
        const frame = Bokeh.documents[0].roots()[0];
        const right = frame.children[0].children[1].children[1];
        const rect = model => Bokeh.index.find_one(model).el.getBoundingClientRect();
        return rect(frame).bottom <= 600 && Math.abs(rect(right).bottom - rect(frame.children[0].children[1]).bottom) <= 1;
    }""")


def assert_horizontal_fit(page):
    script = """() => {
        const overflow = [];
        function check(root) {
            for (const el of root.querySelectorAll('*')) {
                if (!el.checkVisibility()) continue;
                // Canvas annotations and slider handles extend beyond internal anchors.
                // Details tables intentionally scroll within their own wrapper.
                if (!el.matches('.bk-CartesianFrame, .noUi-target, .noUi-base, .noUi-origin, [style="overflow-x:auto"]') &&
                    el.clientWidth && el.scrollWidth > el.clientWidth + 1)
                    overflow.push([el.tagName, el.className, el.clientWidth, el.scrollWidth]);
                if (el.shadowRoot) check(el.shadowRoot);
            }
        }
        check(document);
        return overflow;
    }"""
    try:
        page.wait_for_function(f"() => ({script})().length === 0", timeout=5000)
    except PlaywrightTimeoutError:
        pytest.fail(str(page.evaluate(script)))
    page.wait_for_function("""() => {
        const plots = Bokeh.documents[0].get_model_by_name('explorer_plots').children;
        return plots.filter(p => p.name === 'events').every(p => p.inner_width >= 180);
    }""")


@pytest.mark.parametrize("width", [1440, 920, 704])
def test_horizontal_fit(page, explorer_server, width, tmp_path):
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(explorer_server)
    page.get_by_label("X coordinate").wait_for()
    page.evaluate(
        """width => {
        const frame = Bokeh.index.find_one(Bokeh.documents[0].roots()[0]).el;
        frame.parentElement.style.display = "block";
        frame.parentElement.style.width = `${width - 32}px`;
        frame.parentElement.style.margin = '0 16px';
    }""",
        width,
    )

    page.wait_for_function(
        """width =>
        Bokeh.documents[0].get_model_by_name('explorer_viewport').text.split(',')[0] === String(width - 32)
    """,
        arg=width,
    )
    assert_horizontal_fit(page)
    if width == 1440:
        page.get_by_role("button", name="Fullscreen", exact=True).click()
        page.wait_for_function("() => document.fullscreenElement != null")
        assert_horizontal_fit(page)
        page.get_by_role("button", name="☷", exact=True).click()
        page.get_by_role("button", name="Right (half width)", exact=True).click()
        assert_horizontal_fit(page)
        page.evaluate("() => document.exitFullscreen()")
        page.wait_for_function("() => document.fullscreenElement == null")
        expect(page.get_by_label("X coordinate")).to_be_visible()
        assert_horizontal_fit(page)
    page.get_by_text("Realizations pager", exact=True).click()
    expect(
        page.get_by_text("2 selected / 2 total realizations", exact=True)
    ).to_be_visible()
    assert_horizontal_fit(page)
    page.get_by_text("Facet pathways", exact=True).click()
    page.wait_for_function("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
        return plot?.title?.text.includes('(N=') && plot.inner_width > 0;
    }""")
    assert_horizontal_fit(page)
    page.get_by_role("button", name="☷", exact=True).click()
    page.evaluate("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
        const source = plot.renderers.find(r => r.data_source?.data.id?.includes(1)).data_source;
        source.selected.indices = [source.data.id.indexOf(1)];
    }""")
    expect(page.get_by_text("Realization #1", exact=True)).to_be_visible()
    assert_horizontal_fit(page)
    for layout in ("Right (half width)", "Bottom", "Right sidebar"):
        page.get_by_role("button", name=layout, exact=True).click()
        assert_horizontal_fit(page)
    # Resize only the embedding container: the browser viewport stays wide.
    for resized in (672, 1408, width - 32):
        page.evaluate(
            """width => {
            const frame = Bokeh.index.find_one(Bokeh.documents[0].roots()[0]).el;
            frame.parentElement.style.width = `${width}px`;
        }""",
            resized,
        )
        page.wait_for_function(
            """width =>
            Bokeh.documents[0].get_model_by_name('explorer_viewport').text.split(',')[0] === String(width)
        """,
            arg=resized,
        )
        assert_horizontal_fit(page)
    page.get_by_role("button", name="»", exact=True).click()
    expect(page.get_by_label("X coordinate")).to_be_visible()
    assert_horizontal_fit(page)
    page.get_by_role("button", name="« Controls", exact=True).click()
    assert_horizontal_fit(page)
    page.screenshot(path=str(tmp_path / f"responsive-{width}.png"), full_page=True)


@pytest.mark.parametrize("width", [704, 1280])
def test_bottom_details_preserve_controls(page, explorer_server, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(explorer_server)
    page.get_by_role("button", name="☷", exact=True).click()
    page.get_by_role("button", name="Bottom", exact=True).click()
    if width < 900:
        page.get_by_role("button", name="»", exact=True).click()
    expect(page.get_by_label("X coordinate")).to_be_visible()
    expect(
        page.get_by_role("button", name="Hide details ↓", exact=True)
    ).to_be_visible()
    page.wait_for_function("""() => {
        const doc = Bokeh.documents[0];
        const workspace = doc.roots()[0].children[0].children[1];
        return workspace.type === 'Column' &&
            ['sidebar', 'bottom', 'right'].every(mode =>
                doc.get_model_by_name(`details_layout_${mode}`).active === (mode === 'bottom'));
    }""")
    assert_horizontal_fit(page)
    page.get_by_role("button", name="Right sidebar", exact=True).click()
    page.wait_for_function("""() => {
        const doc = Bokeh.documents[0];
        const workspace = doc.roots()[0].children[0].children[1];
        return workspace.type === 'Row' &&
            ['sidebar', 'bottom', 'right'].every(mode =>
                doc.get_model_by_name(`details_layout_${mode}`).active === (mode === 'sidebar'));
    }""")
    if width < 900:
        expect(page.get_by_label("X coordinate")).to_be_hidden()
    else:
        expect(page.get_by_label("X coordinate")).to_be_visible()
    assert_horizontal_fit(page)


@pytest.mark.parametrize("width", [640, 1280])
def test_page_layout_flows_and_restores_after_fullscreen(monkeypatch, page, width):
    frames = cohort([1], event(1, "escape", 1, 1, 5))
    with serve_explorer(monkeypatch, frames, layout="page") as url:
        page.set_viewport_size({"width": width, "height": 700})
        page.goto(url)
        frame = page.locator(".explorer-bokeh-frame")
        controls = page.locator(".explorer-controls")
        details = page.locator(".explorer-details")
        expect(controls).to_be_visible()
        expect(frame).to_have_css("overflow", "visible")
        page.wait_for_function(
            "document.querySelector('.explorer-bokeh-frame').getBoundingClientRect().height > 800"
        )
        initial_height = frame.bounding_box()["height"]
        page.get_by_role("button", name="« Controls", exact=True).click()
        page.wait_for_function(
            """height =>
            document.querySelector('.explorer-bokeh-frame').getBoundingClientRect().height < height - 100
        """,
            arg=initial_height,
        )
        page.get_by_role("button", name="»", exact=True).click()
        expect(controls).to_be_visible()
        page.get_by_role("button", name="☷", exact=True).click()

        def set_details(height):
            page.evaluate(
                """height => {
                const models = Bokeh.documents[0].all_models;
                const details = [...models].find(m => m.css_classes?.includes('explorer-details'));
                details.children[0].text = `<div style="height: ${height}px">Details</div>`;
            }""",
                height,
            )

        for placement in ("Right sidebar", "Bottom", "Right (half width)"):
            set_details_layout(page, placement)
            set_details(1600)
            page.wait_for_function("""() => {
                function find(root, selector) {
                    const match = root.querySelector(selector);
                    if (match) return match;
                    for (const el of root.querySelectorAll('*')) {
                        if (el.shadowRoot) {
                            const result = find(el.shadowRoot, selector);
                            if (result) return result;
                        }
                    }
                }
                const details = find(document, '.explorer-details');
                return details?.getBoundingClientRect().height >= 1600;
            }""")
            for region in (controls, details, page.locator(".explorer-center")):
                if region.is_visible():
                    assert region.evaluate("""el => {
                        const css = getComputedStyle(el);
                        return css.overflowX === 'visible' && css.overflowY === 'visible';
                    }""")
            tall = frame.bounding_box()["height"]
            # A host resizing its iframe changes the viewport height. Intrinsic
            # content height must stay fixed, including surrounding page padding.
            page.evaluate("document.body.style.padding = '80px 0'")
            page.set_viewport_size({"width": width, "height": round(tall + 160)})
            page.wait_for_function(
                """height => Bokeh.documents[0].get_model_by_name('explorer_viewport')
                    .text.split(',')[1] === String(height)""",
                arg=round(tall + 160),
            )
            assert frame.bounding_box()["height"] == pytest.approx(tall, abs=1)
            set_details(20)
            expect(details).to_have_css("overflow", "visible")
            page.wait_for_function(
                """height =>
                document.querySelector('.explorer-bokeh-frame').getBoundingClientRect().height < height - 100
            """,
                arg=tall,
            )

        page.get_by_role("button", name="Hide details »", exact=True).click()
        page.get_by_role("button", name="»", exact=True).click()
        page.set_viewport_size({"width": width, "height": 700})
        page.get_by_role("button", name="Fullscreen", exact=True).click()
        page.wait_for_function("document.fullscreenElement !== null")
        expect(frame).to_have_css("overflow", "hidden")
        page.get_by_role("button", name="»", exact=True).first.click()
        expect(controls).to_have_css("overflow", "auto")
        page.evaluate("document.exitFullscreen()")
        page.wait_for_function("document.fullscreenElement === null")
        expect(controls).to_be_visible()
        expect(controls).to_have_css("overflow", "visible")
        expect(frame).to_have_css("overflow", "visible")


def set_details_layout(target, placement):
    target.get_by_role("button", name=placement, exact=True).click()
    # A click updates its own toggle immediately; the server then updates the
    # other toggles and rebuilds the workspace. Wait for that whole transition.
    target.wait_for_function(
        """placement => {
        const doc = Bokeh.documents[0];
        const modes = {'Right sidebar': 'sidebar', 'Right (half width)': 'right', 'Bottom': 'bottom'};
        const mode = modes[placement];
        const body = doc.roots()[0].children[0];
        const workspace = body.children[1];
        const left = body.children[0].children[1];
        return doc.is_idle && Object.values(modes).every(value =>
            doc.get_model_by_name(`details_layout_${value}`).active === (value === mode)) &&
            (workspace.type === 'Column') === (mode === 'bottom') &&
            Bokeh.index.find_one(left).el.checkVisibility() === left.visible;
    }""",
        arg=placement,
    )
    target.evaluate(
        "async () => await Bokeh.index.find_one(Bokeh.documents[0].roots()[0]).ready"
    )


def set_long_details(target, height=1600):
    # Sidebar clicks round-trip through the server. Wait for the panel to open
    # before measuring scrolling, otherwise its arrival can resize its siblings.
    expect(target.locator(".explorer-details")).to_be_visible()
    target.evaluate(
        """height => {
        const details = [...Bokeh.documents[0].all_models]
            .find(m => m.css_classes?.includes('explorer-details'));
        details.children[0].text = `<div style="height:${height}px; width:1200px">Details</div>`;
    }""",
        height,
    )


def assert_region_scrolling(target):
    """Each visible panel scrolls without shifting its siblings or the page."""
    selectors = (".explorer-controls", ".explorer-center", ".explorer-details")
    geometry = """() => {
        const classes = ['explorer-controls', 'explorer-center', 'explorer-details'];
        return classes.map(cls => {
            const model = [...Bokeh.documents[0].all_models]
                .find(m => m.css_classes?.includes(cls));
            const el = Bokeh.index.find_one(model).el;
            const rect = el.getBoundingClientRect();
            return [rect.x, rect.y, rect.width, rect.height, el.scrollTop];
        }).concat([[window.scrollX, window.scrollY]]);
    }"""
    for index, selector in enumerate(selectors):
        region = target.locator(selector)
        if not region.is_visible():
            continue
        expect(region).to_have_css("overflow", "auto")
        try:
            target.wait_for_function(
                """selector => {
                const model = [...Bokeh.documents[0].all_models]
                    .find(m => m.css_classes?.includes(selector.slice(1)));
                const el = Bokeh.index.find_one(model).el;
                return el.checkVisibility() && el.scrollHeight > el.clientHeight + 25;
            }""",
                arg=selector,
                timeout=5000,
            )
        except PlaywrightTimeoutError:
            pytest.fail(f"{selector} does not overflow: {target.evaluate(geometry)}")
        before = target.evaluate(geometry)
        region.evaluate("el => el.scrollTop = 25")
        assert region.evaluate("el => el.scrollTop") == 25
        before[index][-1] = 25
        assert target.evaluate(geometry) == before
        # Reaching an arbitrary scroll offset does not prove that the bottom is
        # usable: the scrollport itself may extend past the clipped frame.
        region.evaluate("el => el.scrollTop = el.scrollHeight")
        assert region.evaluate("""el => {
            const rect = el.getBoundingClientRect();
            const frame = document.querySelector('.explorer-bokeh-frame').getBoundingClientRect();
            const children = [...el.shadowRoot.children].filter(child =>
                child.checkVisibility() && child.getBoundingClientRect().height);
            const last = children.at(-1).getBoundingClientRect();
            return rect.bottom <= frame.bottom + 1 && last.bottom <= rect.bottom + 1;
        }""")
        region.evaluate("el => el.scrollTop = 0")


@pytest.mark.parametrize("layout", ["viewport", "contained"])
@pytest.mark.parametrize("width,height", [(640, 300), (955, 300), (1280, 700)])
def test_sidebar_bottom_controls_are_reachable(
    monkeypatch, page, layout, width, height
):
    frames = cohort([1], event(1, "escape", 1, 1, 5))
    with serve_explorer(monkeypatch, frames, layout=layout) as url:
        page.set_viewport_size({"width": width, "height": height})
        page.goto(url)
        expect(page.get_by_label("X coordinate")).to_be_visible()
        if layout == "contained":
            page.locator(".explorer-bokeh-frame").evaluate(
                "el => el.parentElement.style.height = '240px'"
            )

        controls = page.locator(".explorer-controls")
        controls.evaluate("el => el.scrollTop = el.scrollHeight")
        bottom_control = controls.get_by_text("Show readoff values", exact=True)
        expect(bottom_control).to_be_in_viewport()
        controls.get_by_role("checkbox", name="Show readoff values").click()
        expect(
            controls.get_by_role("checkbox", name="Show readoff values")
        ).not_to_be_checked()
        page.get_by_role("button", name="☷", exact=True).click()

        for placement in ("Right sidebar", "Right (half width)", "Bottom"):
            set_details_layout(page, placement)
            set_long_details(page)
            page.evaluate("""() => {
                const details = [...Bokeh.documents[0].all_models]
                    .find(m => m.css_classes?.includes('explorer-details'));
                details.children[1].visible = details.children[2].visible = true;
            }""")
            details = page.locator(".explorer-details")
            energy = details.get_by_role("checkbox", name="Summarise energy")
            expect(energy).to_be_visible()
            page.wait_for_function("""() => {
                const model = [...Bokeh.documents[0].all_models]
                    .find(m => m.css_classes?.includes('explorer-details'));
                const rects = model.children.filter(child => child.visible)
                    .map(child => Bokeh.index.find_one(child).el.getBoundingClientRect());
                return rects.every((rect, i) => !i || rect.top >= rects[i - 1].bottom);
            }""")
            details.evaluate("el => el.scrollTop = el.scrollHeight")
            expect(energy).to_be_in_viewport()
            checked = energy.is_checked()
            energy.click()
            expect(energy).to_be_checked(checked=not checked)


@pytest.mark.parametrize("width", [640, 1100])
def test_contained_layout_follows_host(monkeypatch, page, width):
    frames = cohort([1], event(1, "escape", 1, 1, 5))
    with serve_explorer(monkeypatch, frames, layout="contained") as url:
        page.set_viewport_size({"width": 1280, "height": 700})
        page.goto(url)
        expect(page.get_by_label("X coordinate")).to_be_visible()
        page.evaluate(
            """width => {
            document.body.style.cssText = 'margin:0; padding:20px;';
            const frame = document.querySelector('.explorer-bokeh-frame');
            const mount = frame.parentElement;
            const host = document.createElement('section');
            host.id = 'explorer-host';
            host.style.cssText = `width:${width}px; height:420px; padding:10px;
                border:2px solid black; box-sizing:border-box;`;
            mount.before(host);
            host.append(mount);
            mount.style.cssText = 'height:100%; min-height:0; min-width:0;';
            for (const [position, id] of [['beforebegin', 'above'], ['afterend', 'below']]) {
                host.insertAdjacentHTML(position, `<div id="${id}" style="height:60px">${id}</div>`);
            }
        }""",
            width,
        )

        def assert_host_fit():
            page.wait_for_function("""() => {
                const host = document.querySelector('#explorer-host').getBoundingClientRect();
                const frame = document.querySelector('.explorer-bokeh-frame').getBoundingClientRect();
                const below = document.querySelector('#below').getBoundingClientRect();
                return Math.abs(frame.height - (host.height - 24)) <= 1 &&
                    Math.abs(frame.width - (host.width - 24)) <= 1 &&
                    Math.abs(frame.top - host.top - 12) <= 1 &&
                    Math.abs(below.top - host.bottom) <= 1 && window.scrollY === 0;
            }""")

        assert_host_fit()
        page.get_by_text("Cumulative", exact=True).click()
        page.get_by_text("Bar chart", exact=True).click()
        page.get_by_role("button", name="☷", exact=True).click()
        for placement in ("Right sidebar", "Right (half width)", "Bottom") * 2:
            set_details_layout(page, placement)
            if (
                placement == "Bottom"
                and not page.locator(".explorer-controls").is_visible()
            ):
                page.get_by_role("button", name="»", exact=True).click()
            set_long_details(page)
            assert_host_fit()
            assert_region_scrolling(page)
            set_long_details(page, 20)
            assert_host_fit()
        set_long_details(page)
        for size in ((580, 340), (1000, 500)):
            page.evaluate(
                """([width, height]) => {
                document.querySelector('#explorer-host').style.width = `${width}px`;
                document.querySelector('#explorer-host').style.height = `${height}px`;
            }""",
                size,
            )
            assert_host_fit()
            assert_region_scrolling(page)
        page.get_by_role("button", name="« Controls", exact=True).click()
        assert_host_fit()
        page.get_by_role("button", name="»", exact=True).click()
        page.get_by_role("button", name="Fullscreen", exact=True).click()
        page.wait_for_function("document.fullscreenElement !== null")
        page.wait_for_function("""() => {
            const rect = document.fullscreenElement.getBoundingClientRect();
            return rect.width === innerWidth && rect.height === innerHeight;
        }""")
        expect(page.locator(".explorer-center")).to_have_css("overflow", "auto")
        page.get_by_role("button", name="☷", exact=True).click()
        set_long_details(page)
        assert_region_scrolling(page)
        # Change the host while fullscreen; exit must use its current dimensions.
        page.evaluate("""document.querySelector('#explorer-host').style.cssText +=
            ';width:620px;height:360px'""")
        page.evaluate("document.exitFullscreen()")
        page.wait_for_function("document.fullscreenElement === null")
        assert_host_fit()
        page.evaluate("document.querySelector('#explorer-host').style.height = '460px'")
        assert_host_fit()
        set_long_details(page)
        assert_region_scrolling(page)


def test_viewport_in_bounded_iframe(page, explorer_server):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(explorer_server)
    page.set_content(f"""
        <style>body {{margin:0; padding:20px}} iframe {{display:block; width:100%;
            height:clamp(320px, 50vh, 600px); border:0; box-sizing:border-box}}</style>
        <header style="height:60px">Above</header>
        <iframe name="explorer" src="{explorer_server}" allow="fullscreen" allowfullscreen></iframe>
        <footer style="height:60px">Below</footer>
    """)
    target = page.frame(name="explorer")
    expect(target.get_by_label("X coordinate")).to_be_visible()
    target.evaluate(
        "document.body.style.cssText = 'margin:0; padding:50px 10px 6px; box-sizing:border-box'"
    )
    iframe = page.locator("iframe")
    bounds = iframe.bounding_box()
    assert bounds["height"] == 450
    target.get_by_text("Cumulative", exact=True).click()
    target.get_by_text("Bar chart", exact=True).click()
    target.get_by_role("button", name="☷", exact=True).click()
    target.get_by_role("button", name="Bottom", exact=True).click()
    if not target.locator(".explorer-controls").is_visible():
        target.get_by_role("button", name="»", exact=True).click()
    set_long_details(target)
    assert_region_scrolling(target)
    target.wait_for_function("""() => {
        const rect = document.querySelector('.explorer-bokeh-frame').getBoundingClientRect();
        return Math.abs(rect.top - 50) <= 1 && Math.abs(rect.bottom - (innerHeight - 6)) <= 1 &&
            document.scrollingElement.scrollHeight <= innerHeight &&
            document.scrollingElement.scrollWidth <= innerWidth;
    }""")
    assert iframe.bounding_box() == bounds
    assert page.evaluate("window.scrollY") == 0
    target.get_by_role("button", name="Fullscreen", exact=True).click()
    target.wait_for_function("document.fullscreenElement !== null")
    expect(target.locator(".explorer-center")).to_have_css("overflow", "auto")
    target.evaluate("document.exitFullscreen()")
    target.wait_for_function("document.fullscreenElement === null")
    assert iframe.bounding_box() == bounds
    page.evaluate("document.querySelector('iframe').style.height = '360px'")
    target.wait_for_function("""() => {
        const rect = document.querySelector('.explorer-bokeh-frame').getBoundingClientRect();
        return innerHeight === 360 && Math.abs(rect.bottom - 354) <= 1;
    }""")
    set_long_details(target, 20)
    assert iframe.bounding_box()["height"] == 360
