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
