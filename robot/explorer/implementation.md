# Explorer implementation notes

- The cohort consists of stored `realization` rows for results belonging to the
  chosen run and cluster. The simulator's `EventRecorder` creates a realization
  row when it receives that realization's first event. A simulation with no
  recorded event has no stable realization ID to display; aggregate result
  counters cannot reconstruct its event history.
- A fate requires exactly one final fragmentation or escape event. Missing,
  repeated, conflicting, or out-of-region terminal records are shown as
  incomplete or ambiguous. They remain in CDF and bar denominators but are not
  counted as escaped.
- The time view uses persisted `postime.t` directly. The storage schema has no
  separate realization start timestamp. This avoids treating the first collision
  as time zero; the installed simulator package does not document the time origin
  further.
- The later layout request replaces event-type columns with one overlaid event
  plot per fate row. Event-type filtering and fate faceting remain available; the
  schematic, event plots, CDFs, and bars share one X range and only the bottom
  plot displays its X axis. The frame uses Bokeh GroupBox controls and a Bokeh
  toolbar fullscreen action with the built-in fullscreen icon, keeping the
  full explorer (including controls and range selector) as the fullscreen target.
- Browser checks were attempted through the project's Playwright suite. Its
  Chromium binary exits during sandbox initialization with `Operation not
  permitted`, so interactive behavior remains unverified in this environment.
