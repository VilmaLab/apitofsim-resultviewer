import AlpineModule from "alpinejs";
import persist from "@alpinejs/persist";
import HtmxModule from "htmx.org";
import "htmx-ext-sse";
import { TabulatorFull as Tabulator } from "tabulator-tables";

declare global {
    var Alpine: typeof AlpineModule;
    var htmx: typeof HtmxModule;
    var Tabulator: typeof import("tabulator-tables").TabulatorFull;
}

window.htmx = HtmxModule;
window.Tabulator = Tabulator;

AlpineModule.plugin(persist);
window.Alpine = AlpineModule;
AlpineModule.start();

// The report page embeds its data as an HTML-escaped JSON string on the
// `#report-table` element (set by the template); build the table once the DOM
// is ready.
window.addEventListener("DOMContentLoaded", () => {
    const el = document.querySelector<HTMLElement>("#report-table");
    if (el && el.dataset.report) {
        let data;
        try {
            data = JSON.parse(el.dataset.report);
        } catch (err) {
            console.error("Failed to parse report data:", err);
            return;
        }
        new Tabulator(el, {
            data,
            autoColumns: true,
        });
    }
});
