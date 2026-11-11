"""Persisted execution trees, outcome leaves, and subtree controls."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from html import escape

import polars as pl
from bokeh.events import DocumentReady
from bokeh.models import CustomJS, Div, InlineStyleSheet, Select


@dataclass
class ExecutionTree:
    nodes: dict
    transitions: dict
    species: dict
    pathways: dict
    incomplete: set = field(default_factory=set)
    daughters: dict = field(default_factory=dict)

    @classmethod
    def load(cls, db, run, cluster, pathways):
        args = [run, cluster]
        scope = "experiment_result_id in (select id from cluster_tree_experiment_result where experiment_run_id = ? and cluster_id = ?)"
        nodes = db.db.execute(
            f"select experiment_result_id, node_index, cluster_id from cluster_tree_node where {scope} order by 1, 2",
            args,
        ).fetchall()
        transitions = db.db.execute(
            f"select experiment_result_id, transition_index, source_node_index, pathway_id, destination_node_index from cluster_tree_transition where {scope} order by 1, 2",
            args,
        ).fetchall()
        return cls(
            {(r, n): c for r, n, c in nodes},
            {(r, t): (n, p, dest) for r, t, n, p, dest in transitions},
            dict(db.db.execute("select id, common_name from cluster").fetchall()),
            dict(pathways.select("pathway_id", "pathway").iter_rows()),
            daughters=dict(pathways.select("pathway_id", "daughter").iter_rows()),
        )

    @cached_property
    def outgoing(self):
        outgoing = {}
        for (result, transition), (
            source,
            pathway,
            destination,
        ) in self.transitions.items():
            outgoing.setdefault((result, source), []).append(
                (transition, pathway, destination)
            )
        return outgoing

    @cached_property
    def incoming(self):
        return {
            (result, destination): pathway
            for (result, _), (_, pathway, destination) in self.transitions.items()
            if destination is not None
        }

    @staticmethod
    def node_key(result, node):
        return f"{result}:{node}"

    def label(self, scope):
        if scope == "root":
            return "Initial cluster"
        result, node = map(int, scope.split(":"))
        return self.species[self.nodes[result, node]]

    def classify(self, members, events, *, validate=True):
        histories = events.group_by("realization_id").agg(
            pl.col("id").last().alias("terminal_event_id"),
            pl.col("type").last(),
            pl.col("particle_index").last(),
            pl.col("transition_index").last(),
            pl.col("next_particle_index").last(),
            (
                (
                    (pl.col("provenance") == "persisted")
                    & pl.col("continuity_valid").fill_null(False)
                )
                if validate
                else pl.lit(True)
            )
            .all()
            .alias("valid"),
        )
        rows = []
        for member in members.join(
            histories,
            left_on="id",
            right_on="realization_id",
            how="left",
            maintain_order="left",
        ).iter_rows(named=True):
            result, node = member["experiment_result_id"], member["particle_index"]
            if (
                member["valid"]
                and member["type"] == "fragmentation"
                and member["next_particle_index"] is not None
                and member["next_particle_index"] >= 0
            ):
                node = member["next_particle_index"]
            fate = (
                f"{result}:incomplete:{node}"
                if (result, node) in self.nodes
                else "Incomplete"
            )
            terminal = None
            if member["valid"]:
                if member["type"] == "escape":
                    fate = f"{result}:escape:{node}"
                    terminal = member["terminal_event_id"]
                elif (
                    member["type"] == "fragmentation"
                    and member["next_particle_index"] == -1
                ):
                    fate = f"{result}:stop:{member['transition_index']}"
                    terminal = member["terminal_event_id"]
            rows.append((member["id"], result, fate, terminal))
        return pl.DataFrame(
            rows,
            schema={
                "id": members.schema["id"],
                "experiment_result_id": members.schema["experiment_result_id"],
                "fate": pl.String,
                "terminal_event_id": events.schema["id"],
            },
            orient="row",
        )

    def branches(self):
        """Return branch labels and leaf keys; repeated species stay separate."""

        def node(result, index, label, route):
            key = self.node_key(result, index)
            species = self.species[self.nodes[result, index]]
            children = [
                (
                    f"{result}:escape:{index}",
                    f"{route} / {species} → {species}".strip(" /"),
                    [],
                )
            ]
            for transition, pathway, dest in self.outgoing.get((result, index), ()):
                name = self.pathways[pathway]
                path = f"{route} / {name}" if route else name
                children.append(
                    (f"{result}:stop:{transition}", f"{path} · Stop", [])
                    if dest is None
                    else node(result, dest, name, path)
                )
            if f"{result}:incomplete:{index}" in self.incomplete:
                children.append(
                    (f"{result}:incomplete:{index}", f"{route} · Incomplete", [])
                )
            return key, label, children

        roots = [
            node(r, n, self.species[c], "")
            for (r, n), c in self.nodes.items()
            if n == 0
        ]
        if len(roots) == 1:
            _, label, children = roots[0]
            return ("root", label, children)
        return "root", "Initial cluster", roots

    def leaf_labels(self):
        labels = {"Incomplete": "Incomplete"}

        def visit(branch):
            key, label, children = branch
            if not children:
                labels[key] = label.strip(" /·")
            for child in children:
                visit(child)

        visit(self.branches())
        return labels

    def short_label(self, key, label):
        if key in ("root", "Incomplete"):
            return label
        result, kind, index = (
            key.split(":") if key.count(":") == 2 else (None, None, None)
        )
        if kind == "escape":
            return f"→ {self.species[self.nodes[int(result), int(index)]]}"
        if kind == "stop":
            pathway = self.transitions[int(result), int(index)][1]
            return f"→ {self.daughters[pathway]}"
        if kind == "incomplete":
            return "Incomplete"
        result, node = map(int, key.split(":"))
        return f"→ {self.species[self.nodes[result, node]]}" if node else label

    def short_labels(self):
        labels = {"Incomplete": "Incomplete"}

        def visit(branch, route):
            key, label, children = branch
            text = self.short_label(key, label)
            path = f"{route} {text}" if route else text
            if not children:
                labels[key] = path
            for child in children:
                visit(child, path)

        visit(self.branches(), "")
        return labels

    def cumulative_members(self, members, events, scope):
        """One immediate split/escape per history at the chosen execution node."""
        condition = pl.col("particle_index") == 0
        if scope != "root":
            result, node = map(int, scope.split(":"))
            condition = (pl.col("experiment_result_id") == result) & (
                pl.col("particle_index") == node
            )
        outcomes = events.filter(
            condition
            & pl.col("type").is_in(["fragmentation", "escape"])
            & (pl.col("provenance") == "persisted")
            & pl.col("continuity_valid").fill_null(False)
        ).unique("realization_id", keep="first", maintain_order=True)
        outcomes = outcomes.with_columns(
            pl.when(pl.col("type") == "escape")
            .then(
                pl.col("event_escape")
                if "event_escape" in events.columns
                else pl.concat_str(
                    pl.col("current_cluster_name"),
                    pl.lit(" → "),
                    pl.col("current_cluster_name"),
                )
            )
            .otherwise(pl.col("event_pathway"))
            .alias("fate")
        )
        return members.drop("fate", "terminal_event_id").join(
            outcomes.select(
                pl.col("realization_id").alias("id"),
                pl.col("id").alias("terminal_event_id"),
                "fate",
            ),
            on="id",
            how="left",
        )


def install_tree_controls(doc, ui, tree, names, counts, colors):
    """Native folding preserves expansion while Bokeh owns leaf selection."""
    from .controls import checkbox

    indices = {name: i for i, name in enumerate(names)}
    labels = tree.leaf_labels()
    scopes = [("root", "Initial cluster")]
    icon = (
        (Path(__file__).parents[2] / "static/img/curve.svg")
        .read_text()
        .replace(
            'width="160" height="160"', 'width="16" height="16" aria-hidden="true"'
        )
    )

    def render(branch, full=False):
        key, label, children = branch
        if children:
            rendered = [render(child, full) for child in children]
            leaves = [i for _, child_leaves in rendered for i in child_leaves]
            content = "".join(html for html, _ in rendered)
        else:
            leaves = [indices[key]]
        selected = sum(counts.get(names[i], 0) for i in leaves)
        if not full and not selected and key != "root":
            return "", leaves
        if children and key != "root":
            scopes.append((key, label))
        value = ",".join(map(str, leaves))
        color = colors.get(key)
        if children and key != "root":
            result, node = map(int, key.split(":"))
            pathway = tree.incoming.get((result, node))
            color = colors.get(tree.pathways.get(pathway))
        if not selected:
            color = None
        swatch = (
            f'<span style="color:{color}" aria-hidden="true">■</span> ' if color else ""
        )
        text = tree.short_label(key, label)
        control = f'<label title="{escape(label, quote=True)}"><input type="checkbox" data-leaves="{value}" checked> {swatch}{escape(text)} <i>({selected})</i></label>'
        observed = str(selected > 0 or key == "root").lower()
        hidden = "" if selected or key == "root" else " hidden"
        if not children:
            return (
                f'<div class="tree-leaf" data-observed="{observed}"{hidden}>{control}</div>',
                leaves,
            )
        button = f'<button type="button" data-scope="{key}" aria-label="Cumulative from {escape(label, quote=True)}" title="Cumulative from {escape(label, quote=True)}" aria-pressed="{str(key == "root").lower()}">{icon}</button>'
        return (
            f'<details open data-branch="{key}" data-observed="{observed}"{hidden}><summary>{control}{button}</summary><div class="tree-children">{content}</div></details>',
            leaves,
        )

    branches = tree.branches()

    def make_html(full=False):
        scopes[:] = [("root", "Initial cluster")]
        html, _ = render(branches, full)
        if "Incomplete" in indices:
            html += render(("Incomplete", "Incomplete", []), full)[0]
        return html

    html = make_html()
    view = Div(
        text=html,
        name="fragmentation_tree",
        sizing_mode="stretch_width",
        stylesheets=[
            InlineStyleSheet(
                css="""
        [hidden] { display: none !important; }
        summary { display: flex; align-items: start; gap: 3px; cursor: pointer; }
        summary::before { content: '▸'; flex-shrink: 0; }
        details[open] > summary::before { content: '▾'; }
        label { flex: 1; min-width: 0; overflow-wrap: anywhere; cursor: pointer; }
        input { margin: 2px; }
        i { color: #64748b; }
        button { border: 1px solid #cbd5e1; border-radius: 3px; background: white; cursor: pointer; padding: 2px; }
        button[aria-pressed=true] { background: #dbeafe; border-color: #2563eb; }
        .tree-children { margin-left: 10px; border-left: 1px solid #cbd5e1; padding-left: 3px; }
        .tree-leaf { display: flex; padding: 3px 0; }
    """
            )
        ],
    )
    scope = Select(value="root", options=scopes, visible=False, name="cumulative_scope")
    full_tree = checkbox("Show empty pathways")
    full_tree.name = "show_full_pathway_tree"
    ui.fate.visible = False
    container = ui.left_controls.children[1].child
    container.children = [
        view,
        full_tree,
        ui.distinguish_escapees,
        ui.fate_facets,
        scope,
    ]
    ui.fate_facets.labels = ["Facet histories"]

    def show_full(attr, old, new):
        view.text = make_html(bool(new))
        scope.options = list(scopes)
        if scope.value not in {key for key, _ in scopes}:
            scope.value = "root"

    full_tree.on_change("active", show_full)
    sync = CustomJS(
        args=dict(
            view=view, fate=ui.fate, scope=scope, full_tree=full_tree, cdf=ui.show_cdf
        ),
        code="""
        const el = Bokeh.index.find_one(view);
        const root = el?.shadow_el ?? el?.el?.shadowRoot ?? el?.el;
        if (!root) return;
        const sync = () => {
            for (const branch of root.querySelectorAll('[data-observed="false"]'))
                branch.hidden = !full_tree.active.length;
            for (const branch of root.querySelectorAll('details[data-branch]')) {
                const open = root._treeOpen?.get(branch.dataset.branch);
                if (open != null) branch.open = open;
            }
            const active = new Set(fate.active);
            for (const input of root.querySelectorAll('input[data-leaves]')) {
                const leaves = input.dataset.leaves.split(',').filter(Boolean).map(Number);
                const n = leaves.filter(i => active.has(i)).length;
                input.checked = n === leaves.length;
                input.indeterminate = n > 0 && n < leaves.length;
            }
            for (const button of root.querySelectorAll('button[data-scope]'))
                button.setAttribute('aria-pressed', String(button.dataset.scope === scope.value));
        };
        if (!root._treeInstalled) {
            root._treeInstalled = true;
            root._treeOpen = new Map();
            root.addEventListener('toggle', event => {
                const branch = event.target;
                if (branch.matches('details[data-branch]')) root._treeOpen.set(branch.dataset.branch, branch.open);
            }, true);
            root.addEventListener('change', event => {
                const input = event.target.closest('input[data-leaves]');
                if (!input) return;
                const active = new Set(fate.active);
                for (const i of input.dataset.leaves.split(',').filter(Boolean).map(Number)) {
                    if (input.checked) active.add(i); else active.delete(i);
                }
                fate.active = [...active].sort((a, b) => a - b);
                sync();
            });
            root.addEventListener('click', event => {
                const button = event.target.closest('button[data-scope]');
                if (button) {
                    event.preventDefault();
                    if (scope.value === button.dataset.scope) cdf.active = [0];
                    else scope.value = button.dataset.scope;
                }
            });
        }
        sync();
        requestAnimationFrame(sync);
    """,
    )
    doc.js_on_event(DocumentReady, sync)
    ui.fate.js_on_change("active", sync)
    scope.js_on_change("value", sync)
    full_tree.js_on_change("active", sync)
    view.js_on_change("text", sync)
    return scope, labels, full_tree, lambda: show_full(None, None, full_tree.active)
