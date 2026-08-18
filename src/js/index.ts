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

// Report rows are loaded a page at a time; the server applies sorting before
// returning each page so column sorting always covers the complete report.
window.addEventListener("DOMContentLoaded", () => {
    const el = document.querySelector<HTMLElement>("#report-table");
    if (el && el.dataset.url) {
        new Tabulator(el, {
            ajaxURL: el.dataset.url,
            autoColumns: true,
            pagination: true,
            paginationMode: "remote",
            paginationSize: 100,
            paginationCounter: "rows",
            sortMode: "remote",
        });
    }
});
