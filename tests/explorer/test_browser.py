"""Explorer browser behavior."""

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
    expect(groups).to_have_count(7)
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


@pytest.fixture
def explorer_server(monkeypatch):
    """Serve the synthetic cohort with real Bokeh session callbacks."""
    from queue import Queue
    from threading import Thread

    from bokeh.server.server import Server
    from tornado.ioloop import IOLoop

    frames = cohort(
        [1, 2],
        event(1, "init", 1, 0, 0),
        event(2, "escape", 1, 1, 5),
        event(3, "init", 2, 0, 0),
        event(4, "escape", 2, 1, 5),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    ready = Queue()

    def serve():
        loop = IOLoop()
        try:
            server = Server(
                {"/": lambda doc: build_document(None, doc, 1, 1)},
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


def test_live_inspection_preserves_plot_and_zoom(page, explorer_server):
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
    page.get_by_role("button", name="Inspect event #2", exact=True).click()
    page.wait_for_function("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        const clicked = plot.renderers.find(r => r.glyph?.size?.value === 20).data_source.data;
        const source = plot.renderers.find(r => r.data_source?.data.id?.includes(2)).data_source.data;
        return clicked.position.length === 1 && clicked.position[0] === source.x[source.id.indexOf(2)];
    }""")
    assert page.evaluate("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        return {plot: plot.id, range: plot.x_range.id, start: plot.x_range.start, end: plot.x_range.end};
    }""") == {**initial, "start": -0.5, "end": 9.5}
    page.get_by_role("button", name="Clear selection", exact=True).click()
    page.wait_for_function("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('events');
        return plot.renderers.find(r => r.glyph?.size?.value === 20).data_source.data.position.length === 0;
    }""")
