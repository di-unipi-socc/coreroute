import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 18, "axes.labelpad": 16})
sns.set_palette("colorblind")   # unica palette dei plot (default di hue/colori)

# Font condivisi fra exp1 e exp2 (i valori sugli assi, tick, restano invariati)
LABEL_FS        = 15   # etichette assi / etichette di riga
TITLE_FS        = 15   # titoli dei pannelli
LEGEND_FS       = 13   # testo legenda
LEGEND_TITLE_FS = 14   # titolo legenda

df = pd.read_csv("results.csv")

TOPOLOGIES      = ["iaag", "er", "ba"]
PCT_MODS        = [0.1, 0.2, 0.3, 0.5]
FLOW_FACTORS    = [1.0, 0.75, 0.5, 0.25]
NODES           = [250, 500, 750, 1000]
MODE_COLORS     = {"CR": sns.color_palette("colorblind")[0], "FULL": sns.color_palette("colorblind")[3]}
N_CONFIRMATORIA = 1000
EPOCHS_SHOWN    = [0, 10, 19]
PCT_MODS_SHOWN  = [0.1, 0.3, 0.5]

os.makedirs("plots", exist_ok=True)


def speedup_table(pct_mod: float, topology: str) -> pd.DataFrame:
    """sum(T | mode=FULL) / sum(T | mode=CR), grouped by (flow_factor, n)."""
    sub = df[(df["pct_mod"] == pct_mod) & (df["topology"] == topology)]
    t_full = sub[sub["mode"] == "FULL"].groupby(["flow_factor", "n"])["T"].sum()
    t_cr   = sub[sub["mode"] == "CR"].groupby(["flow_factor", "n"])["T"].sum()
    return (t_full / t_cr).unstack().reindex(index=FLOW_FACTORS, columns=NODES)


def speedup_heatmap():
    tables = {(p, t): speedup_table(p, t) for p in PCT_MODS for t in TOPOLOGIES}
    vmin = min(t.values.min() for t in tables.values())
    vmax = max(t.values.max() for t in tables.values())

    fig, axes = plt.subplots(len(PCT_MODS), len(TOPOLOGIES), figsize=(16, 18))
    for i, pct_mod in enumerate(PCT_MODS):
        for j, topology in enumerate(TOPOLOGIES):
            ax = axes[i, j]
            sns.heatmap(tables[(pct_mod, topology)], annot=True, fmt=".1f", cmap="Blues",
                        vmin=vmin, vmax=vmax, cbar=False, ax=ax,
                        annot_kws={"fontsize": 19}, linewidths=0.5, linecolor="white")
            ax.tick_params(labelsize=16)
            if i == 0:
                ax.set_title(topology.upper(), fontsize=TITLE_FS, fontweight="bold")
            show_xlabel = i == len(PCT_MODS) - 1 and j == len(TOPOLOGIES) // 2
            ax.set_xlabel("Nodes" if show_xlabel else "", fontsize=LABEL_FS)
            ax.set_ylabel(f"$\\mathbf{{Perturbation\\ =\\ {int(pct_mod * 100)}\\%}}$\nFlow factor" if j == 0 else "", fontsize=LABEL_FS)

    plt.tight_layout()
    out_path = "plots/speedup_full_cr.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def speedup_heatmap_pct50():
    """Same as speedup_heatmap, but only the pct_mod=50% row -- the
    representative slice for the thesis body, one 1x3 figure instead of
    the full 4x3 grid (which stays available as an appendix figure)."""
    pct_mod = 0.5
    tables = {t: speedup_table(pct_mod, t) for t in TOPOLOGIES}
    vmin = min(t.values.min() for t in tables.values())
    vmax = max(t.values.max() for t in tables.values())

    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(16, 5))
    for j, topology in enumerate(TOPOLOGIES):
        ax = axes[j]
        sns.heatmap(tables[topology], annot=True, fmt=".1f", cmap="Blues",
                    vmin=vmin, vmax=vmax, cbar=False, ax=ax,
                    annot_kws={"fontsize": 19}, linewidths=0.5, linecolor="white")
        ax.tick_params(labelsize=16)
        ax.set_title(topology.upper(), fontsize=TITLE_FS, fontweight="bold")
        ax.set_xlabel("Nodes" if j == len(TOPOLOGIES) // 2 else "", fontsize=LABEL_FS)
        ax.set_ylabel("Flow factor" if j == 0 else "", fontsize=LABEL_FS)

    plt.tight_layout()
    out_path = "plots/speedup_full_cr_pct50.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def diff_simm_vs_latency():
    """Same shape as symm_dist_delay() in hreuseDelay/build_graphs.py: one
    confirmatory cell (n=1000, flow_factor=1.0, pct_mod=0.5), a few
    representative epochs, faceted by topology (rows) x epoch (cols)."""
    sub = df[(df["n"] == N_CONFIRMATORIA) & (df["flow_factor"] == 1.0)
             & (df["pct_mod"] == 0.5) & (df["epoch"].isin(EPOCHS_SHOWN))].copy()
    sub["avg_latency_ms"] = sub["avg_latency"] * 1000

    g = sns.relplot(
        data=sub,
        x="avg_latency_ms",
        y="diff_simm_tot",
        hue="mode",
        hue_order=list(MODE_COLORS),
        palette=MODE_COLORS,
        row="topology", row_order=TOPOLOGIES,
        col="epoch", col_order=EPOCHS_SHOWN,
        kind="scatter", height=3.5, aspect=1.2, legend=False,
        s=60,
    )
    g.set_axis_labels("Path delay (ms)", "Symm. distance", fontsize=LABEL_FS)
    g.set_titles("")
    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True, labelsize=15)
        ax.set_xlabel("")
        ax.xaxis.label.set_visible(True)

    legend_handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=m)
                       for m, c in MODE_COLORS.items()]
    last_topology = TOPOLOGIES[-1]
    for topology in TOPOLOGIES:
        for epoch in EPOCHS_SHOWN:
            ax = g.axes_dict[(topology, epoch)]
            if epoch == EPOCHS_SHOWN[0]:
                ax.set_ylabel(f"$\\mathbf{{{topology.upper()}}}$\nSymm. distance", fontsize=LABEL_FS)
            if topology == TOPOLOGIES[0]:
                ax.set_title(f"Epoch {epoch}", fontsize=TITLE_FS, fontweight="bold")
            # one xlabel for the whole grid, bottom row only
            if topology == last_topology and epoch == EPOCHS_SHOWN[len(EPOCHS_SHOWN) // 2]:
                ax.set_xlabel("Path delay (ms)", fontsize=LABEL_FS)

    # one legend for the whole grid, not one per row
    g.axes_dict[(TOPOLOGIES[0], EPOCHS_SHOWN[0])].legend(
        handles=legend_handles, title="Mode", loc="best",
        fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, frameon=False)

    g.tight_layout()
    out_path = "plots/diff_simm_vs_latency.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def diff_simm_vs_latency_iaag():
    """Same confirmatory cell and epochs as diff_simm_vs_latency, but only
    the reference topology (iaag), col=epoch only, no row facet. x-axis
    limits match the full grid's shared axis (all topologies), so this
    panel is directly comparable to its counterpart there."""
    cell = df[(df["n"] == N_CONFIRMATORIA) & (df["flow_factor"] == 1.0)
              & (df["pct_mod"] == 0.5) & (df["epoch"].isin(EPOCHS_SHOWN))].copy()
    cell["avg_latency_ms"] = cell["avg_latency"] * 1000
    xlim = (cell["avg_latency_ms"].min(), cell["avg_latency_ms"].max())

    sub = cell[cell["topology"] == "iaag"]

    g = sns.relplot(
        data=sub,
        x="avg_latency_ms",
        y="diff_simm_tot",
        hue="mode",
        hue_order=list(MODE_COLORS),
        palette=MODE_COLORS,
        col="epoch", col_order=EPOCHS_SHOWN,
        kind="scatter", height=4, aspect=1.2,
        s=60,
    )
    g.set(xlim=xlim)
    g.set_ylabels("Symm. distance", fontsize=LABEL_FS)
    g.set_xlabels("")
    middle_epoch = EPOCHS_SHOWN[len(EPOCHS_SHOWN) // 2]
    for epoch, ax in g.axes_dict.items():
        ax.set_title(f"Epoch {epoch}", fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(labelsize=15)
        if epoch == middle_epoch:
            ax.set_xlabel("Path delay (ms)", fontsize=LABEL_FS)
    g._legend.set_title("Mode", prop={"size": LEGEND_TITLE_FS})
    plt.setp(g._legend.get_texts(), fontsize=LEGEND_FS)

    g.tight_layout()
    out_path = "plots/diff_simm_vs_latency_iaag.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def repair_workload():
    """Share of flows requiring repair (P_KO) per epoch, by topology, for CR
    at the confirmatory cell -- evidence for the discussion's claim that
    IAAG concentrates more traffic on shared links, invalidating more flows
    per perturbation."""
    sub = df[(df["n"] == N_CONFIRMATORIA) & (df["flow_factor"] == 1.0)
             & (df["pct_mod"] == 0.5) & (df["mode"] == "CR")].copy()
    sub["P_KO_pct"] = sub["P_KO"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=sub, x="epoch", y="P_KO_pct", hue="topology",
                 hue_order=TOPOLOGIES, marker="o", errorbar="sd", ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Flows KO", fontsize=LABEL_FS)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[t.upper() for t in TOPOLOGIES],
              title="Topology", prop={"size": LEGEND_FS, "weight": "bold"}, title_fontsize=LEGEND_TITLE_FS, frameon=False)

    plt.tight_layout()
    out_path = "plots/repair_workload.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def flows_changed_vs_nodes():
    """Mean share of flows changed vs network size, by mode -- averaged
    across epoch and seed (not faceted by epoch): CR and FR each drift
    through their own independent perturbation history here, so a given
    epoch index is not the same network state for both and can't be
    compared side by side like in the old paired-column design."""
    sub = df[df["pct_mod"] == 0.5].copy()
    sub["pct_flows_changed"] = 100 * sub["flows_changed"] / sub["num_flows"]
    agg = sub.groupby(["topology", "n", "flow_factor", "mode"])["pct_flows_changed"].mean().reset_index()

    g = sns.relplot(
        data=agg,
        x="n",
        y="pct_flows_changed",
        hue="mode",
        hue_order=list(MODE_COLORS),
        palette=MODE_COLORS,
        col="topology", col_order=TOPOLOGIES,
        row="flow_factor", row_order=FLOW_FACTORS,
        kind="line", markers=True,
        height=3, aspect=1.2,
    )
    g.set_axis_labels("Nodes", "Flows changed (%)", fontsize=LABEL_FS)
    g.set(xticks=NODES)
    g.tight_layout()
    out_path = "plots/flows_changed_vs_nodes.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def flows_changed_vs_epoch():
    """Share of flows changed per epoch, by mode, at the confirmatory cell --
    each mode's own trajectory over its independent perturbation history,
    not a per-epoch CR vs FR comparison of the same network state."""
    sub = df[(df["n"] == N_CONFIRMATORIA) & (df["flow_factor"] == 1.0)
             & (df["pct_mod"] == 0.5)].copy()
    sub["pct_flows_changed"] = 100 * sub["flows_changed"] / sub["num_flows"]

    g = sns.relplot(
        data=sub,
        x="epoch",
        y="pct_flows_changed",
        hue="mode",
        hue_order=list(MODE_COLORS),
        palette=MODE_COLORS,
        col="topology", col_order=TOPOLOGIES,
        kind="line", marker="o", errorbar="sd",
        height=4, aspect=1.2,
    )
    g.set_axis_labels("Epoch", "Flows changed (%)", fontsize=LABEL_FS)
    for topology, ax in g.axes_dict.items():
        ax.set_title(topology.upper(), fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(labelsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    g._legend.set_title("Mode", prop={"size": LEGEND_TITLE_FS})
    plt.setp(g._legend.get_texts(), fontsize=LEGEND_FS)

    g.tight_layout()
    out_path = "plots/flows_changed_vs_epoch.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def _speedup_workload_data() -> pd.DataFrame:
    """Shared aggregation for the speedup-vs-workload variants: one row per
    (topology, pct_mod, flow_factor, n), with speedup = T_FR / T_CR and
    pct_ko = mean P_KO (CR) as a percentage."""
    cr = df[df["mode"] == "CR"]
    full = df[df["mode"] == "FULL"]
    group_cols = ["topology", "pct_mod", "flow_factor", "n"]

    t_cr = cr.groupby(group_cols)["T"].sum()
    t_full = full.groupby(group_cols)["T"].sum()
    speedup = (t_full / t_cr).rename("speedup")
    pct_ko = (cr.groupby(group_cols)["P_KO"].mean() * 100).rename("pct_ko")

    return pd.concat([speedup, pct_ko], axis=1).reset_index()


def _style_workload_grid(g: sns.FacetGrid) -> None:
    """Shared row/col/legend styling for the grid variants (rows=pct_mod,
    cols=flow_factor)."""
    g.set_titles("")
    for (pct_mod, flow_factor), ax in g.axes_dict.items():
        ax.tick_params(labelsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if pct_mod == PCT_MODS[0]:
            ax.set_title(f"Flow factor = {flow_factor}", fontsize=TITLE_FS)
        if flow_factor == FLOW_FACTORS[0]:
            ax.set_ylabel(f"$\\mathbf{{Perturbation\\ =\\ {int(pct_mod * 100)}\\%}}$\nSpeedup", fontsize=LABEL_FS)
        if pct_mod == PCT_MODS[-1] and flow_factor == FLOW_FACTORS[len(FLOW_FACTORS) // 2]:
            ax.set_xlabel("Flows requiring repair (%)", fontsize=LABEL_FS)

    g._legend.set_title("Topology", prop={"size": LEGEND_TITLE_FS})
    plt.setp(g._legend.get_texts(), fontsize=LEGEND_FS)
    g.tight_layout()


def speedup_vs_workload_scatter():
    """Variant 1: grid scatter, rows=pct_mod, cols=flow_factor, hue=topology.
    12 points per panel (3 topologies x 4 n)."""
    data = _speedup_workload_data()
    g = sns.relplot(
        data=data,
        x="pct_ko",
        y="speedup",
        hue="topology", hue_order=TOPOLOGIES,
        row="pct_mod", row_order=PCT_MODS,
        col="flow_factor", col_order=FLOW_FACTORS,
        kind="scatter", height=3, aspect=1.2, s=60,
    )
    _style_workload_grid(g)
    out_path = "plots/speedup_vs_workload_scatter.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def speedup_vs_workload_line():
    """Variant 2: same grid and points as the scatter variant, but connected
    in pct_ko order (kind="line") so each topology reads as a trend curve
    instead of raw dots."""
    data = _speedup_workload_data()
    g = sns.relplot(
        data=data,
        x="pct_ko",
        y="speedup",
        hue="topology", hue_order=TOPOLOGIES,
        row="pct_mod", row_order=PCT_MODS,
        col="flow_factor", col_order=FLOW_FACTORS,
        kind="line", marker="o", height=3, aspect=1.2,
    )
    _style_workload_grid(g)
    out_path = "plots/speedup_vs_workload_line.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def speedup_vs_workload_errorbar():
    """Variant 3: same grid, but collapsed to one point per topology per
    panel (mean across n), with std-across-n error bars on both axes --
    the fewest marks per panel of the three variants."""
    data = _speedup_workload_data()
    agg = data.groupby(["topology", "pct_mod", "flow_factor"]).agg(
        pct_ko_mean=("pct_ko", "mean"), pct_ko_std=("pct_ko", "std"),
        speedup_mean=("speedup", "mean"), speedup_std=("speedup", "std"),
    ).reset_index()
    colors = dict(zip(TOPOLOGIES, sns.color_palette(n_colors=len(TOPOLOGIES))))

    fig, axes = plt.subplots(len(PCT_MODS), len(FLOW_FACTORS), figsize=(16, 16),
                              sharex=True, sharey=True)
    for i, pct_mod in enumerate(PCT_MODS):
        for j, flow_factor in enumerate(FLOW_FACTORS):
            ax = axes[i, j]
            cell = agg[(agg["pct_mod"] == pct_mod) & (agg["flow_factor"] == flow_factor)]
            for topology in TOPOLOGIES:
                row = cell[cell["topology"] == topology]
                ax.errorbar(row["pct_ko_mean"], row["speedup_mean"],
                            xerr=row["pct_ko_std"], yerr=row["speedup_std"],
                            fmt="o", color=colors[topology], capsize=3)
            ax.tick_params(labelsize=14)
            if i == 0:
                ax.set_title(f"Flow factor = {flow_factor}", fontsize=TITLE_FS)
            if j == 0:
                ax.set_ylabel(f"$\\mathbf{{Perturbation\\ =\\ {int(pct_mod * 100)}\\%}}$\nSpeedup", fontsize=LABEL_FS)
            if i == len(PCT_MODS) - 1 and j == len(FLOW_FACTORS) // 2:
                ax.set_xlabel("Flows requiring repair (%)", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=t.upper())
               for t, c in colors.items()]
    axes[0, 0].legend(handles=handles, title="Topology", prop={"size": LEGEND_FS, "weight": "bold"}, title_fontsize=LEGEND_TITLE_FS, frameon=False)

    plt.tight_layout()
    out_path = "plots/speedup_vs_workload_errorbar.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def speedup_vs_workload_scatter_flowfactor():
    """Same facet layout as speedup_vs_workload_errorbar_flowfactor (one
    panel per flow_factor, pct_mod and n not faceted), but a raw scatter of
    every (topology, pct_mod, n) point instead of collapsing to a single
    mean+std marker."""
    data = _speedup_workload_data()
    colors = dict(zip(TOPOLOGIES, sns.color_palette(n_colors=len(TOPOLOGIES))))

    fig, axes = plt.subplots(1, len(FLOW_FACTORS), figsize=(16, 5), sharex=True, sharey=True)
    for j, flow_factor in enumerate(FLOW_FACTORS):
        ax = axes[j]
        cell = data[data["flow_factor"] == flow_factor]
        for topology in TOPOLOGIES:
            row = cell[cell["topology"] == topology]
            ax.scatter(row["pct_ko"], row["speedup"], color=colors[topology], s=40)
        ax.tick_params(labelsize=14)
        ax.set_title(f"Flow factor = {flow_factor}", fontsize=TITLE_FS)
        ax.set_ylabel("Speedup" if j == 0 else "", fontsize=LABEL_FS)
    axes[len(FLOW_FACTORS) // 2].set_xlabel("Flows requiring repair (%)", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=t.upper())
               for t, c in colors.items()]
    axes[0].legend(handles=handles, title="Topology", prop={"size": LEGEND_FS, "weight": "bold"}, title_fontsize=LEGEND_TITLE_FS, frameon=False)

    plt.tight_layout()
    out_path = "plots/speedup_vs_workload_scatter_flowfactor.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def speedup_vs_workload_scatter_pctmod():
    """Same facet layout as speedup_vs_workload_errorbar_pctmod (one panel
    per pct_mod, flow_factor and n not faceted), but a raw scatter of every
    (topology, flow_factor, n) point instead of collapsing to a single
    mean+std marker."""
    data = _speedup_workload_data()
    colors = dict(zip(TOPOLOGIES, sns.color_palette(n_colors=len(TOPOLOGIES))))

    fig, axes = plt.subplots(1, len(PCT_MODS), figsize=(16, 5), sharex=True, sharey=True)
    for j, pct_mod in enumerate(PCT_MODS):
        ax = axes[j]
        cell = data[data["pct_mod"] == pct_mod]
        for topology in TOPOLOGIES:
            row = cell[cell["topology"] == topology]
            ax.scatter(row["pct_ko"], row["speedup"], color=colors[topology], s=40)
        ax.tick_params(labelsize=14)
        ax.set_title(f"Perturbation = {int(pct_mod * 100)}%", fontsize=TITLE_FS, fontweight="bold")
        ax.set_ylabel("Speedup" if j == 0 else "", fontsize=LABEL_FS)
    axes[len(PCT_MODS) // 2].set_xlabel("Flows requiring repair (%)", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=t.upper())
               for t, c in colors.items()]
    axes[0].legend(handles=handles, title="Topology", prop={"size": LEGEND_FS, "weight": "bold"}, title_fontsize=LEGEND_TITLE_FS, frameon=False)

    plt.tight_layout()
    out_path = "plots/speedup_vs_workload_scatter_pctmod.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def speedup_vs_workload_scatter_n_pctmod():
    """Grid scatter, rows=n, cols=pct_mod; each panel pools topology and
    flow_factor (12 points: 3 topologies x 4 flow_factor), hue=topology."""
    data = _speedup_workload_data()
    colors = dict(zip(TOPOLOGIES, sns.color_palette(n_colors=len(TOPOLOGIES))))

    fig, axes = plt.subplots(len(NODES), len(PCT_MODS), figsize=(16, 16), sharex=True, sharey=True)
    for i, n in enumerate(NODES):
        for j, pct_mod in enumerate(PCT_MODS):
            ax = axes[i, j]
            cell = data[(data["n"] == n) & (data["pct_mod"] == pct_mod)]
            for topology in TOPOLOGIES:
                row = cell[cell["topology"] == topology]
                ax.scatter(row["pct_ko"], row["speedup"], color=colors[topology], s=40)
            ax.tick_params(labelsize=14)
            if i == 0:
                ax.set_title(f"Perturbation = {int(pct_mod * 100)}%", fontsize=TITLE_FS, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"n = {n}\nSpeedup", fontsize=LABEL_FS)
            if i == len(NODES) - 1 and j == len(PCT_MODS) // 2:
                ax.set_xlabel("Flows requiring repair (%)", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=t.upper())
               for t, c in colors.items()]
    axes[0, 0].legend(handles=handles, title="Topology", prop={"size": LEGEND_FS, "weight": "bold"}, title_fontsize=LEGEND_TITLE_FS, frameon=False)

    plt.tight_layout()
    out_path = "plots/speedup_vs_workload_scatter_n_pctmod.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def speedup_vs_workload_scatter_topology_pctmod():
    """Grid scatter, rows=topology, cols=pct_mod (10/30/50% only); n pooled
    within each panel, flow_factor as hue."""
    data = _speedup_workload_data()
    colors = dict(zip(FLOW_FACTORS, sns.color_palette(n_colors=len(FLOW_FACTORS))))

    fig, axes = plt.subplots(len(TOPOLOGIES), len(PCT_MODS_SHOWN), figsize=(12, 12), sharex=True, sharey=True)
    for i, topology in enumerate(TOPOLOGIES):
        for j, pct_mod in enumerate(PCT_MODS_SHOWN):
            ax = axes[i, j]
            cell = data[(data["topology"] == topology) & (data["pct_mod"] == pct_mod)]
            for flow_factor in FLOW_FACTORS:
                row = cell[cell["flow_factor"] == flow_factor]
                ax.scatter(row["pct_ko"], row["speedup"], color=colors[flow_factor], s=40)
            ax.tick_params(labelsize=14)
            if i == 0:
                ax.set_title(f"Perturbation = {int(pct_mod * 100)}%", fontsize=TITLE_FS, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"$\\mathbf{{{topology.upper()}}}$\nSpeedup", fontsize=LABEL_FS)
            if i == len(TOPOLOGIES) - 1 and j == len(PCT_MODS_SHOWN) // 2:
                ax.set_xlabel("Flows KO (%)", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], linestyle="", marker="", label="Flow factor:")] + [
        Line2D([0], [0], marker="o", linestyle="", color=c, markersize=14, label=f"{ff}")
        for ff, c in colors.items()
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=len(handles),
               fontsize=LEGEND_FS, frameon=False, handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout()
    out_path = "plots/speedup_vs_workload_scatter_topology_pctmod.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def speedup_vs_workload_scatter_iaag():
    """Same as speedup_vs_workload_scatter_topology_pctmod, but only the
    reference topology (iaag), one row instead of three."""
    data = _speedup_workload_data()
    colors = dict(zip(FLOW_FACTORS, sns.color_palette(n_colors=len(FLOW_FACTORS))))

    fig, axes = plt.subplots(1, len(PCT_MODS_SHOWN), figsize=(12, 4.5), sharex=True, sharey=True)
    for j, pct_mod in enumerate(PCT_MODS_SHOWN):
        ax = axes[j]
        cell = data[(data["topology"] == "iaag") & (data["pct_mod"] == pct_mod)]
        for flow_factor in FLOW_FACTORS:
            row = cell[cell["flow_factor"] == flow_factor]
            ax.scatter(row["pct_ko"], row["speedup"], color=colors[flow_factor], s=40)
        ax.tick_params(labelsize=14)
        ax.set_title(f"Perturbation = {int(pct_mod * 100)}%", fontsize=TITLE_FS, fontweight="bold")
        if j == 0:
            ax.set_ylabel("$\\mathbf{IAAG}$\nSpeedup", fontsize=LABEL_FS)
        if j == len(PCT_MODS_SHOWN) // 2:
            ax.set_xlabel("Flows KO (%)", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], linestyle="", marker="", label="Flow factor:")] + [
        Line2D([0], [0], marker="o", linestyle="", color=c, markersize=14, label=f"{ff}")
        for ff, c in colors.items()
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.1), ncol=len(handles),
               fontsize=LEGEND_FS, frameon=False, handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout()
    out_path = "plots/speedup_vs_workload_scatter_iaag.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def failed_flows():
    """Share of flows still KO after repair (N_KO - N_R), CR vs FULL: grouped
    bars, one panel per topology, perturbation severity on x. Pooled over n,
    flow_factor, epoch and seed."""
    sub = df.copy()
    sub["pct_failed"] = 100 * (sub["N_KO"] - sub["N_R"]) / sub["num_flows"]

    g = sns.catplot(
        data=sub, kind="bar",
        x="pct_mod", y="pct_failed", hue="mode", hue_order=list(MODE_COLORS),
        palette=MODE_COLORS, col="topology", col_order=TOPOLOGIES,
        errorbar=None, height=4, aspect=1, sharey=True, legend=False,
    )
    g.set_titles("")
    g.set_xticklabels([f"{int(p * 100)}%" for p in PCT_MODS])
    for topology, ax in g.axes_dict.items():
        ax.tick_params(labelsize=14)
        ax.set_title(topology.upper(), fontsize=TITLE_FS, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Flows failed (%)" if topology == TOPOLOGIES[0] else "", fontsize=LABEL_FS)
    g.axes_dict[TOPOLOGIES[len(TOPOLOGIES) // 2]].set_xlabel("Perturbation", fontsize=LABEL_FS)

    handles = [Line2D([0], [0], marker="s", linestyle="", color=c, markersize=14, label=m)
               for m, c in MODE_COLORS.items()]
    g.fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                 ncol=len(handles), fontsize=LEGEND_FS, frameon=False)

    g.tight_layout()
    out_path = "plots/failed_flows.pdf"
    g.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    diff_simm_vs_latency_iaag()
    diff_simm_vs_latency()
    speedup_heatmap()
    speedup_vs_workload_scatter_topology_pctmod()
    failed_flows()
