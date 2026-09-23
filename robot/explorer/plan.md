# Interactive realization explorer

## Scope and existing code

Implement the Explorer tab in `apitofresview.plotting.explorer`, scoped to the
selected experiment and cluster. Keep the existing Realizations tab available.
Use Bokeh server models and callbacks, following the viewer's existing
`bokeh_document` / `BokehASGI` integration. Prefer built-in Bokeh functionality;
set figures to `output_backend="webgl"`. Initial views include all realizations;
performance optimization, sampling, and cross-experiment comparisons are out of
scope.

Investigation references:

- `src/apitofresview/webapp.py`: Realizations route and Bokeh embedding pattern.
  The locally edited Explorer route, template, and callback are scaffolding:
  the template says `Test`, the callback repeats the spectrogram code, and only
  the spectrogram is mounted. Preserve and integrate that work during implementation.
- `apitofsim/python/apitofsim/plotting/events.py`: geometry extraction, region
  boundaries, event queries, and existing radial-distance/strip/beeswarm plots.
  Schematic scaling is unimplemented; review equal-region mapping rather than
  copying it blindly.
- `apitofsim/python/apitofsim/workflow/sql/{realizations,event_report}.sql`:
  realization and event storage. The report omits event IDs and fragmentation
  pathway IDs, so it is insufficient for stable selection and fate grouping.
- `/home/frankier/rse/hipercog/ctap-dashboard/src/venn_ts/plot.py`: VennDiff's
  fullscreen frame, collapsible controls, separate normal/fullscreen sidebar
  states, and 40 px toolbar buttons. Adapt these patterns without introducing
  a dependency on the dashboard.

## Frame and controls

The fullscreen target is the entire explorer: toolbar, both sidebars, plots,
axes, schematic, and realization-range selector. Use large buttons, responsive
plot sizing, and independently scrollable sidebars. Retain separate sidebar
visibility states for normal and fullscreen views; Escape exits fullscreen.

- Left, hideable: fate selection/faceting; event selection/faceting; X coordinate
  and scaling; event layout; violin envelope; schematic checkbox; independent
  visibility toggles for realizations, CDF, and regional bars.
- Center: aligned facet panels, optional schematic, linked plot stack, and an
  index-range selector below the bottom X axis. Keep the selector available in
  aggregate-only views.
- Right, hideable: selected realization metadata and its complete event list.

Defaults: all fates and all three event types (collision, fragmentation, escape)
included; event-type faceting on and fate faceting off; radial distance;
schematic X proportions; schematic visible; realizations visible, CDF and bars
off; full realization range; left sidebar open and right sidebar closed until
selection. Include convenient collision-only, fragmentation-only, and combined
choices through the event-type control.

## Data and filtering

Load a realization cohort independently of the visible events. Preserve raw
coordinates and use stable realization/event IDs throughout. Each event record
needs its type, time, x/y/z position, realization ID, and fragmentation pathway
where available; derive radial distance as `sqrt(x*x + y*y)`.

Resolve the selected experiment run to its experiment results explicitly. The
existing event query compares its run argument with `experiment_result_id`;
verify joins and support both single- and multi-pathway results rather than
inheriting the existing single-pathway assertion.

Assign each realization a pathway fate: its fragmentation pathway, or Escaped.
Check simulation termination semantics before implementing this classification.
Do not silently classify missing/incomplete event histories as escaped or count
multiple fragmentation records as multiple realizations. Surface incomplete or
ambiguous histories explicitly; the escaped interpretation below requires
complete, mutually exclusive terminal outcomes.

Filtering order:

1. Experiment/cluster scope defines a stable list ordered by realization ID.
2. An inclusive index-range selector chooses a contiguous slice of that list.
   This is an arbitrary realization ordering, not event time or a spatial zoom.
3. Fate selections filter whole realizations within that slice.
4. Event-type selections filter markers only. They do not remove realizations
   from aggregate denominators or hide their fragmentation outcomes from CDFs.

The index range filters everything, including CDFs and bars. Show selected and
total realization counts. Avoid renumbering the underlying list when other
controls change. No implicit sampling or initial truncation.

Fate and event type can both be faceted simultaneously. Use fate rows and event
type columns when both are enabled; all events of a realization belong to its
fate row. Without faceting, overlay the selected categories with clear legends.
Place one aggregate stack per fate row, spanning its event-type columns, rather
than duplicating the same aggregate for every event type. Without fate faceting,
aggregate the selected fates together. Label each aggregate's cohort size.

## Coordinates, regions, and schematic

Use one coordinate mapping and linked Bokeh X ranges for events, CDFs, bars,
region annotations, and schematic. Preserve physical coordinates for details.
Connecting a selected realization follows event-time order, even if its z
coordinate reverses direction.

| X mode | Mapping | Machine depiction | Regional bars | Escaped treatment |
| --- | --- | --- | --- | --- |
| Schematic (default) | Piecewise linear within fixed readable region widths | Aligned block diagram | Enabled | Shared terminal slot and CDF brace |
| Equal regions | Piecewise linear, one unit per region | Aligned block diagram | Enabled | Shared terminal slot and CDF brace |
| Physical distance | Raw axial z with physical units | Labeled boundary ticks; no block diagram | Disabled | Raw escape position; brace at right edge |
| Elapsed time | Recorded event time relative to realization start | Disabled | Disabled | Raw escape time; no CDF escaped indicator |

Choose explicit schematic width weights in one constant (initially
`[2, 1, 1, 3, 2]` for the five existing geometry intervals). Obtain region names
and boundaries from the machine configuration/model and confirm their meaning;
do not infer labels solely from array indices. The schematic is a simplified
block diagram whose edges exactly match the transformed region boundaries.
In physical mode, the schematic control governs labeled boundary guides instead.

For schematic/equal modes, add an Escaped slot beyond the final machine boundary.
Place escape markers there, alongside the regional escaped bar and the CDF
brace. Escape remains an outcome category, not another machine region. Details
always show actual recorded escape coordinates/time. Share the slot across all
facets and retain it even when escape markers are hidden.

When changing to an incompatible mode, disable bars/schematic as applicable and
explain why beside the control. Retain preferences for restoration when returning.
If bars were the only visible view, show the CDF so the explorer remains usable.
Confirm the time origin from simulator semantics; do not subtract the first
recorded collision time as a proxy for realization start.

## Realization plot and selection

Event glyphs are X markers. Support radial distance (default), deterministic
jittered strips, and beeswarm layouts. Provide an optional violin envelope for
strip/beeswarm modes; compute it from visible event X positions within each
facet. It represents event density, not fragmentation probability. Empty,
singleton, and constant-position groups must work without density estimation.
Keep categorical escaped markers separate from continuous density estimation.

Use Bokeh scatter, line, patch, hover, and selection models. A small data-layout
helper may be needed for beeswarm packing and violin density; use existing
dependencies where practical, without adding a custom renderer initially.

Only one realization is selected at a time:

- Clicking an event highlights that realization's visible markers, draws a line
  through its visible events in time order within each facet, and emphasizes the
  clicked event. Do not connect unrelated facet coordinate systems.
- Open the right sidebar, show all stored events in time order (including hidden
  types), highlight the clicked row, and scroll it into view.
- Clicking a sidebar event emphasizes its marker. If its type is hidden, show a
  temporary selection indicator without changing the user's event filters.
- Display every stored field, pathway details where available, units, and derived
  radial distance. Do not promise energies/velocities absent from the database.
- Keep selection stable across scaling and layout changes. Clear it when the
  selected realization leaves the range/fate cohort. Closing the sidebar does
  not clear selection; provide an explicit clear-selection action.

## CDF and bars

For each aggregate cohort of N realizations, show a raw, unsmoothed step function:

`F(x) = count(realizations with fragmentation coordinate <= x) / N`.

Use one terminal fragmentation outcome per realization, not collision counts
or a CDF normalized over fragmented realizations alone. Tied coordinates form a
single jump. Spatial modes describe fragmentation locations; time mode describes
elapsed fragmentation times. Transform spatial step locations with the same
mapping as the event plot. Keep the CDF Y range fixed to `[0, 1]`.

For complete histories, the plateau is the fragmented fraction. A right-side
brace spans that plateau to 1 and labels the escaped proportion. In schematic
and equal modes it occupies the Escaped slot above the escaped bar; in physical
mode it sits at the plot's right edge. Hide the indicator in time mode. Handle
all-fragmented and all-escaped cohorts without invalid geometry. If histories
are incomplete, do not label the entire complement as escaped; show an explicit
data-quality state instead.

Each regional bar is `count(fragmented within region) / N`, not cumulative.
The final bar is `count(escaped) / N`, in the shared Escaped slot. For complete
histories all bars sum to 1. Bars occupy their corresponding transformed region
intervals; adopt one consistent boundary convention, with the final endpoint
included. Use a Y range from zero to the largest bar (with minimal visual
headroom), not a fixed maximum of 1. Use a sensible empty/all-zero fallback.

The three view toggles are independent: any single view, any pair, or all three.
Stack realizations above CDF above bars, omitting hidden views and maintaining
linked X axes. Prevent all views being switched off. Empty cohorts show an
explicit message rather than a division by zero or a misleading zero curve.

## Implementation sequence

1. Add the plotting package and Explorer entry point, integrate the existing local
   scaffolding, mount its Bokeh application, and embed it in the Explorer template.
2. Build the scoped data adapter with stable IDs, complete event details, fate
   classification, geometry, and shared cohort/filter state. Reuse verified
   geometry helpers and database patterns.
3. Implement coordinate mappings, the fullscreen frame, collapsible sidebars,
   large controls, faceting, range selector, and radial event rendering.
4. Add synchronized selection, chronological details, strip/beeswarm layouts,
   optional violin envelopes, and aligned machine schematic.
5. Add raw CDFs, regional/escaped bars, linked ranges, view toggles, incompatible
   mode handling, and empty/incomplete-data states.
6. Verify numerical behavior and browser interactions; document any necessary
   deviations from this plan before broadening scope.

## Verification

- Small known cohorts: correct experiment/result joins, single/multi-pathway
  support, stable IDs, fate membership, time ordering, and complete sidebar fields.
- Numerical fixtures: coordinate boundaries (including z=0 and final endpoint),
  CDF ties/denominators, regional counts, escaped complement, range filtering,
  empty/all-escaped/all-fragmented cohorts, and incomplete histories.
- Browser checks: both facet dimensions together; all three event types;
  selection in both directions and automatic sidebar scrolling; selection across
  layouts; independent views; linked axes; mode restrictions; aligned Escaped
  slot; fullscreen including controls and axes; sidebar restoration on exit.
- Confirm WebGL is requested, allowing Bokeh's normal fallback for unsupported
  glyphs. Do not add performance targets or a separate optimization phase yet.

This change is a plan only; implementation follows in a separate task.
