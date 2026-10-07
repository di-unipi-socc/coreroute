"""Genera tutti i grafici/tabelle della tesi per exp1 (euristica
biased_k_shortest_path_latency vs esaustiva) a partire da results.csv --
l'unica fonte dati per questo esperimento. Riusa grafici.py (stessa cartella)
cosi' com'e', solo sovrascrivendo a runtime etichette/colori/percorsi.

Raggruppa in un solo file gli script sviluppati separatamente durante
l'analisi (accuratezza, tabelle divario, tempi di esecuzione nelle loro
varie viste, percentuale di timeout). Ogni plot e' una funzione indipendente,
eseguita in sequenza da main().
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import grafici as g

HERE = Path(__file__).parent
g.CSV_PATH = HERE / "results.csv"
g.OUT_DIR = HERE / "plots"
g.OUT_DIR.mkdir(exist_ok=True)

# Setup comune: una sola euristica (RuD) in questo esperimento, etichette e
# colori colorblind-safe condivisi da tutti i grafici.
g.EURISTICHE = ["biased_k_shortest_path_latency"]
g.STRATEGIE = g.EURISTICHE + [g.OTTIMO]
g.NOME_STRATEGIA["biased_k_shortest_path_latency"] = "heuristic"
g.NOME_STRATEGIA["exhaustive_optimal"] = "exhaustive"
CB = sns.color_palette("colorblind").as_hex()   # unica palette dei plot
g.STRATEGY_STYLES["biased_k_shortest_path_latency"]["color"] = CB[0]
g.STRATEGY_STYLES["exhaustive_optimal"]["color"] = CB[1]
g.COLORI["biased_k_shortest_path_latency"] = CB[0]
g.ETICHETTA_ALGO["biased_k_shortest_path_latency"] = "heuristic"
g.ETICHETTA_METRICA["diff_flussi_persi_pct"] = "FF"
g.ETICHETTA_METRICA["pct_diff_costo"] = "DF"
g.ETICHETTA_METRICA["pct_diff_lat"] = "DE"
LEGENDA_METRICHE = "FF = Flows Failed      DF = Symmetric Distance      DE = Delay"

# Font condivisi fra exp1 e exp2 (i valori sugli assi, tick, restano invariati)
LABEL_FS        = 15   # etichette assi / etichette di riga
TITLE_FS        = 15   # titoli dei pannelli
LEGEND_FS       = 13   # testo legenda
LEGEND_TITLE_FS = 14   # titolo legenda

NODI = [20, 25, 30, 35, 40]
NUM_FLOWS = [3, 4, 5, 6]


# =================================================================== accuratezza

def plot_accuratezza(df: pd.DataFrame) -> None:
    """Quota di celle in cui l'euristica coincide esattamente con l'ottimo."""
    dati = g.prepara_dati_accuratezza(df)
    GRUPPI = [*g.TOPOLOGIE_ORDINATE, "tutte"]
    ETICHETTE = {**g.NOME_TOPO, "tutte": "ALL"}
    dati = dati.assign(pct=dati["coincide"] * 100)

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.patch.set_facecolor("white")
    sns.barplot(data=dati, x="topology", y="pct", hue="strategy", order=GRUPPI,
                hue_order=g.EURISTICHE, palette=g.COLORI, errorbar=("ci", 95), width=0.4,
                capsize=0.1, err_kws=dict(color=g.INK, linewidth=1.2), legend=False, ax=ax)

    for cont, strat in zip(ax.containers, g.EURISTICHE):
        etichette = ["100%" if v.get_height() >= 99.95 else f"{v.get_height():.0f}%" for v in cont]
        ax.bar_label(cont, labels=etichette, padding=-36, fontsize=16,
                     fontweight="bold", color=g._colore_testo(g.COLORI[strat]))

    for x, gruppo in zip(ax.get_xticks(), GRUPPI):
        ax.text(x, -6, ETICHETTE[gruppo], ha="center", va="top", fontsize=LABEL_FS, fontweight="bold", color=g.INK)

    ax.set_ylim(0, 108)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("Share of repairs matching optimal (%)", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=16)
    g._pulisci_assi(ax, spine_visibili=("left",))

    legenda = [Patch(facecolor=g.COLORI[s], label=g.NOME_STRATEGIA[s]) for s in g.EURISTICHE]
    ax.legend(handles=legenda, frameon=False, fontsize=LEGEND_FS, loc="upper center",
              bbox_to_anchor=(0.5, 1.08), ncol=3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    g._salva(fig, "accuratezza.pdf")


# =================================================================== tabelle divario

def _tabella_divario_semplice(dati: pd.DataFrame, nome_file: str) -> None:
    """Come tabella_divario ma senza colonna/colore per l'algoritmo (inutile
    con una sola euristica), senza sfondo colorato nelle celle dati,
    grassetto solo sui valori a 0.0%, e meno padding attorno alla figura."""
    lungo = dati.melt(id_vars=["n", "num_flows", "strategy"],
                       value_vars=list(g.ETICHETTA_METRICA), var_name="colonna", value_name="valore")
    lungo["metrica"] = pd.Categorical(lungo["colonna"].map(g.ETICHETTA_METRICA),
                                       categories=list(g.ETICHETTA_METRICA.values()), ordered=True)

    tabella = lungo.pivot_table(index="num_flows", columns=["n", "metrica"], values="valore")
    num_flussi = tabella.index.tolist()
    nodi = tabella.columns.get_level_values("n").unique().tolist()
    metriche = tabella.columns.get_level_values("metrica").unique().tolist()
    n_metrica = len(metriche)

    LARGH_ETICH, LARGH_DATO, ALT_HEADER, ALT_DATO = 1.15, 1.0, 0.85, 0.62
    x_bordi = [0, LARGH_ETICH] + [LARGH_ETICH + k * LARGH_DATO for k in range(1, len(nodi) * n_metrica + 1)]
    y_bordi = [0, ALT_HEADER, 2 * ALT_HEADER] + [2 * ALT_HEADER + k * ALT_DATO for k in range(1, len(num_flussi) + 1)]
    largh_tot, alt_tot = x_bordi[-1], y_bordi[-1]

    ALT_LEGENDA = 0.35
    fig, ax = plt.subplots(figsize=(largh_tot * 0.9 + 0.5, (alt_tot + ALT_LEGENDA) * 0.9 + 0.5))
    ax.set_xlim(0, largh_tot)
    ax.set_ylim(alt_tot, -ALT_LEGENDA)
    ax.axis("off")
    ax.text((x_bordi[0] + x_bordi[-1]) / 2, -ALT_LEGENDA / 2, LEGENDA_METRICHE,
            ha="center", va="center", fontsize=8.5, color=g.INK)

    def cella(x0, y0, x1, y1, testo="", grassetto=False, sfondo="white", colore_testo=g.INK, fontsize=9):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=sfondo, edgecolor="none", zorder=1))
        if testo:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, testo, ha="center", va="center", fontsize=fontsize,
                     fontweight="bold" if grassetto else "normal", color=colore_testo, zorder=3)

    HEADER = "white"
    cella(x_bordi[0], y_bordi[0], x_bordi[1], y_bordi[2], "Flussi /\nNodi", grassetto=True, sfondo=HEADER, fontsize=8)

    for i, n in enumerate(nodi):
        cella(x_bordi[1 + i * n_metrica], y_bordi[0], x_bordi[1 + (i + 1) * n_metrica], y_bordi[1],
              f"n = {n}", grassetto=True, sfondo=HEADER)
        for j, m in enumerate(metriche):
            cella(x_bordi[1 + i * n_metrica + j], y_bordi[1], x_bordi[1 + i * n_metrica + j + 1], y_bordi[2],
                  m, sfondo=HEADER)

    for i, nf in enumerate(num_flussi):
        cella(x_bordi[0], y_bordi[2 + i], x_bordi[1], y_bordi[2 + i + 1],
              f"flussi = {nf}", grassetto=True, sfondo=HEADER)
        for k, n in enumerate(nodi):
            for j2, m in enumerate(metriche):
                v = tabella.loc[nf, (n, m)]
                cella(x_bordi[1 + k * n_metrica + j2], y_bordi[2 + i], x_bordi[1 + k * n_metrica + j2 + 1], y_bordi[2 + i + 1],
                      f"{v:+.1f}%" if pd.notna(v) else "--", grassetto=(pd.notna(v) and round(v, 1) == 0.0))

    y_piene = {y_bordi[0], y_bordi[2], y_bordi[-1]} | set(y_bordi[2:])
    x_piene = {x_bordi[0], x_bordi[1], x_bordi[-1]} | {x_bordi[1 + i * n_metrica] for i in range(len(nodi) + 1)}
    L = dict(color=g.INK, solid_capstyle="butt", clip_on=False, zorder=2)
    for y in y_bordi:
        if y in y_piene:
            ax.plot([x_bordi[0], x_bordi[-1]], [y, y], lw=1.8, **L)
        elif y == y_bordi[1]:
            ax.plot([x_bordi[1], x_bordi[-1]], [y, y], lw=0.4, **L)
    for x in x_bordi:
        if x in x_piene:
            ax.plot([x, x], [y_bordi[0], y_bordi[-1]], lw=1.8, **L)
        else:
            ax.plot([x, x], [y_bordi[1], y_bordi[-1]], lw=0.4, **L)

    percorso = g.OUT_DIR / nome_file
    fig.savefig(percorso, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Salvato: {percorso}")


def plot_tabelle_divario(df: pd.DataFrame, solo_generale: bool = False) -> None:
    dati_divario = g.get_data_divario(df)
    for topo in ([] if solo_generale else g.TOPOLOGIE_ORDINATE):
        _tabella_divario_semplice(dati_divario[dati_divario["topology"] == topo], f"tabella_divario_{topo}.pdf")
    dati_generale = dati_divario.groupby(["strategy", "n", "num_flows"])[list(g.ETICHETTA_METRICA)].mean().reset_index()
    _tabella_divario_semplice(dati_generale, "tabella_divario_generale.pdf")


# =================================================================== percentuale di timeout

def plot_pct_timeout(df: pd.DataFrame) -> None:
    """Percentuale di epoche in timeout per n, una linea per topologia
    (esaustiva) -- usa 'ok' (mai NaN, a differenza di N_KO che manca sulle
    epoche in timeout) quindi non soffre del bias da sopravvivenza."""
    exh = df[df["strategy"] == g.OTTIMO]
    dati = (exh.groupby(["topology", "n"])["ok"].apply(lambda s: 100 * (~s).mean())).rename("pct_timeout").reset_index()
    COLORE_TOPO = {"iaag": CB[0], "er": CB[1], "ba": CB[4]}

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    for topo in g.TOPOLOGIE_ORDINATE:
        sotto = dati[dati["topology"] == topo].set_index("n").reindex(NODI)
        ax.plot(NODI, sotto["pct_timeout"], marker="o", lw=1.8, ms=5, color=COLORE_TOPO[topo], label=g.NOME_TOPO[topo])

    ax.set_xticks(NODI)
    ax.set_xlabel("Nodes", fontsize=LABEL_FS)
    ax.set_ylabel("Epoche in timeout (%)", fontsize=LABEL_FS)
    ax.set_title("Percentuale di timeout per topologia (esaustiva)", fontsize=TITLE_FS, color=g.INK)
    g._pulisci_assi(ax, alpha=0.2)
    ax.legend(frameon=False, prop={"size": LEGEND_FS, "weight": "bold"})
    fig.tight_layout()
    g._salva(fig, "pct_timeout_per_topologia.pdf")


# =================================================================== tempo di esecuzione (righe=flussi, x=n)

def _disegna_riga_tempo(ax_riga, dati, massimo, num_flows, etichetta_flussi=True, titoli=True):
    for colonna, topo in enumerate(g.TOPOLOGIE_ORDINATE):
        ax = ax_riga[colonna]
        for strat in g.STRATEGIE:
            stile = g.STRATEGY_STYLES[strat]
            sotto = (dati[(dati["strategy"] == strat) & (dati["topology"] == topo) & (dati["num_flows"] == num_flows)]
                     .set_index("n").reindex(NODI))
            ax.plot(NODI, sotto["tempo_medio"], linestyle=stile["linestyle"], marker=stile["marker"],
                    color=stile["color"], lw=1.8, ms=4.5, label=g.NOME_STRATEGIA[strat])
            if strat == g.OTTIMO:
                valori = sotto["tempo_medio"].tolist()
                for i, (x, y, pct) in enumerate(zip(NODI, valori, sotto["pct_timeout"])):
                    if pd.notna(pct) and pct > 0:
                        vicini = [v for v in valori[max(0, i - 1):i + 2] if pd.notna(v)]
                        y_sicuro = max(vicini) + 0.05 * massimo
                        ha = "left" if i == 0 else "center"
                        ax.text(x, y_sicuro, f"{pct:.0f}%", ha=ha, va="bottom",
                                fontsize=13, fontweight="bold", color=stile["color"])
        ax.set_xlim(NODI[0] - 2, NODI[-1] + 2)
        ax.set_xticks(NODI)
        if titoli:
            ax.set_title(g.NOME_TOPO[topo], fontsize=TITLE_FS, fontweight="bold", color=g.INK)
        g._pulisci_assi(ax, alpha=0.2)
        ax.tick_params(axis="both", labelsize=16)
        if colonna == 0:
            ax.set_ylabel("Execution Time [s]", fontsize=LABEL_FS)

    if etichetta_flussi:
        ax_riga[-1].text(1.03, 0.5, f"flows = {num_flows}", transform=ax_riga[-1].transAxes, rotation=270,
                         ha="left", va="center", fontsize=LABEL_FS, fontweight="bold", color=g.INK)
    ax_riga[0].set_ylim(-0.04 * massimo, massimo * 1.4)
    handles, labels = ax_riga[0].get_legend_handles_labels()
    ax_riga[0].legend(handles, labels, frameon=False, fontsize=LEGEND_FS, loc="upper left")


def plot_tempo_griglia_topologia_flussi(df: pd.DataFrame) -> None:
    """Tempo medio reale (sulle epoche completate, non imputato a 1800s) vs n,
    righe = num_flows, colonne = topologia. % di timeout annotata su ogni
    punto dell'esaustiva. Genera sia la griglia completa sia la vista separata
    per flussi=6 (stessa scala y)."""
    tempo = (df[df["ok"]].groupby(["strategy", "topology", "num_flows", "n"])["T_CR"]
             .mean().rename("tempo_medio").reset_index())
    timeout_pct = (df.groupby(["strategy", "topology", "num_flows", "n"])["ok"]
                   .apply(lambda s: 100 * (~s).mean()).rename("pct_timeout").reset_index())
    dati = tempo.merge(timeout_pct, on=["strategy", "topology", "num_flows", "n"])
    massimo = dati["tempo_medio"].max()

    fig, assi = plt.subplots(len(NUM_FLOWS), len(g.TOPOLOGIE_ORDINATE), figsize=(12, 14), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for riga, num_flows in enumerate(NUM_FLOWS):
        _disegna_riga_tempo(assi[riga], dati, massimo, num_flows, titoli=(riga == 0))
        if riga == len(NUM_FLOWS) - 1:
            assi[riga][len(g.TOPOLOGIE_ORDINATE) // 2].set_xlabel("Nodes", fontsize=LABEL_FS)
    fig.tight_layout()
    g._salva(fig, "tempo_griglia_topologia_flussi.pdf")

    fig6, assi6 = plt.subplots(1, len(g.TOPOLOGIE_ORDINATE), figsize=(12, 4.2), sharex=True, sharey=True)
    fig6.patch.set_facecolor("white")
    _disegna_riga_tempo(assi6, dati, massimo, 6, etichetta_flussi=False)
    assi6[len(g.TOPOLOGIE_ORDINATE) // 2].set_xlabel("Nodes", fontsize=LABEL_FS)
    fig6.tight_layout()
    g._salva(fig6, "tempo_griglia_topologia_flussi6.pdf")


# =================================================================== tempo di esecuzione (righe=n, x=flussi)

def plot_tempo_griglia_flussi_topologia(df: pd.DataFrame) -> None:
    """Come plot_tempo_griglia_topologia_flussi ma con righe = n e asse X =
    num_flows, invece del contrario."""
    tempo = (df[df["ok"]].groupby(["strategy", "topology", "num_flows", "n"])["T_CR"]
             .mean().rename("tempo_medio").reset_index())
    timeout_pct = (df.groupby(["strategy", "topology", "num_flows", "n"])["ok"]
                   .apply(lambda s: 100 * (~s).mean()).rename("pct_timeout").reset_index())
    dati = tempo.merge(timeout_pct, on=["strategy", "topology", "num_flows", "n"])
    massimo = dati["tempo_medio"].max()

    fig, assi = plt.subplots(len(NODI), len(g.TOPOLOGIE_ORDINATE), figsize=(12, 17), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for riga, n in enumerate(NODI):
        for colonna, topo in enumerate(g.TOPOLOGIE_ORDINATE):
            ax = assi[riga][colonna]
            for strat in g.STRATEGIE:
                stile = g.STRATEGY_STYLES[strat]
                sotto = (dati[(dati["strategy"] == strat) & (dati["topology"] == topo) & (dati["n"] == n)]
                         .set_index("num_flows").reindex(NUM_FLOWS))
                ax.plot(NUM_FLOWS, sotto["tempo_medio"], linestyle=stile["linestyle"], marker=stile["marker"],
                        color=stile["color"], lw=1.8, ms=4.5, label=g.NOME_STRATEGIA[strat])
                if strat == g.OTTIMO:
                    valori = sotto["tempo_medio"].tolist()
                    for i, (x, y, pct) in enumerate(zip(NUM_FLOWS, valori, sotto["pct_timeout"])):
                        if pd.notna(pct) and pct > 0:
                            vicini = [v for v in valori[max(0, i - 1):i + 2] if pd.notna(v)]
                            y_sicuro = max(vicini) + 0.05 * massimo
                            ha = "left" if i == 0 else "center"
                            ax.text(x, y_sicuro, f"{pct:.0f}%", ha=ha, va="bottom",
                                    fontsize=13, fontweight="bold", color=stile["color"])
            ax.set_xlim(NUM_FLOWS[0] - 0.4, NUM_FLOWS[-1] + 0.4)
            ax.set_xticks(NUM_FLOWS)
            if riga == 0:
                ax.set_title(g.NOME_TOPO[topo], fontsize=TITLE_FS, fontweight="bold", color=g.INK)
            g._pulisci_assi(ax, alpha=0.2)
            ax.tick_params(axis="both", labelsize=16)
            if riga == len(NODI) - 1 and colonna == len(g.TOPOLOGIE_ORDINATE) // 2:
                ax.set_xlabel("Repaired Flows", fontsize=LABEL_FS)
            if colonna == 0:
                ax.set_ylabel("Execution Time [s]", fontsize=LABEL_FS)
            if colonna == len(g.TOPOLOGIE_ORDINATE) - 1:
                ax.text(1.03, 0.5, f"nodes = {n}", transform=ax.transAxes, rotation=270,
                        ha="left", va="center", fontsize=LABEL_FS, fontweight="bold", color=g.INK)

        assi[riga][0].set_ylim(-0.04 * massimo, massimo * 1.4)
        handles, labels = assi[riga][0].get_legend_handles_labels()
        assi[riga][0].legend(handles, labels, frameon=False, fontsize=LEGEND_FS, loc="upper left")

    fig.tight_layout()
    g._salva(fig, "tempo_griglia_flussi_topologia.pdf")


# =================================================================== tempo di esecuzione (righe=N_KO, x=n)

def plot_tempo_per_nko(df: pd.DataFrame) -> None:
    """Tempo vs n a parita' di N_KO (invece che di num_flows) -- isola lo
    scaling dell'esaustiva dal cambio di composizione del campione. Punti
    con meno di 10 campioni vengono omessi."""
    SOGLIA_CAMPIONI = 10
    NKO_RIGHE = [0, 1, 2, 3]

    ok = df[df["ok"]]
    agg = ok.groupby(["strategy", "topology", "N_KO", "n"])["T_CR"].agg(["mean", "count"]).reset_index()
    agg.loc[agg["count"] < SOGLIA_CAMPIONI, "mean"] = float("nan")

    fig, assi = plt.subplots(len(NKO_RIGHE), len(g.TOPOLOGIE_ORDINATE), figsize=(12, 14), sharex=True, sharey="row")
    fig.patch.set_facecolor("white")

    for riga, n_ko in enumerate(NKO_RIGHE):
        for colonna, topo in enumerate(g.TOPOLOGIE_ORDINATE):
            ax = assi[riga][colonna]
            for strat in g.STRATEGIE:
                stile = g.STRATEGY_STYLES[strat]
                sotto = (agg[(agg["strategy"] == strat) & (agg["topology"] == topo) & (agg["N_KO"] == n_ko)]
                         .set_index("n").reindex(NODI))
                ax.plot(NODI, sotto["mean"], linestyle=stile["linestyle"], marker=stile["marker"],
                        color=stile["color"], lw=1.8, ms=4.5, label=g.NOME_STRATEGIA[strat])
            ax.set_xticks(NODI)
            if riga == 0:
                ax.set_title(g.NOME_TOPO[topo], fontsize=TITLE_FS, fontweight="bold", color=g.INK)
            g._pulisci_assi(ax, alpha=0.2)
            ax.tick_params(axis="both", labelsize=16)
            if colonna == 0:
                ax.set_ylabel("Execution Time [s]", fontsize=LABEL_FS)
            if colonna == len(g.TOPOLOGIE_ORDINATE) - 1:
                ax.text(1.03, 0.5, f"N_KO = {n_ko}", transform=ax.transAxes, rotation=270,
                        ha="left", va="center", fontsize=LABEL_FS, fontweight="bold", color=g.INK)
            if riga == len(NKO_RIGHE) - 1 and colonna == len(g.TOPOLOGIE_ORDINATE) // 2:
                ax.set_xlabel("Nodes", fontsize=LABEL_FS)

        massimo_riga = agg[agg["N_KO"] == n_ko]["mean"].max()
        assi[riga][0].set_ylim(-0.04 * massimo_riga, massimo_riga * 1.1)
        handles, labels = assi[riga][0].get_legend_handles_labels()
        assi[riga][0].legend(handles, labels, frameon=False, fontsize=LEGEND_FS, loc="upper left")

    fig.tight_layout()
    g._salva(fig, "tempo_griglia_per_nko.pdf")


# =================================================================== tempo di esecuzione (solo epoche comuni a tutti gli n)

def plot_tempo_epoche_comuni(df: pd.DataFrame) -> None:
    """Tempo vs n solo sulle epoche "appaiate": stessa (topologia, num_flows,
    seed, pct_mod, epoch) presente E riuscita (esaustiva) per TUTTI i 5 valori
    di n -- il confronto fra n e' sullo stesso insieme di prove, non su un
    campione che cambia composizione da un n all'altro. Punti con meno di 10
    prove comuni vengono omessi."""
    SOGLIA_CAMPIONI = 10
    CHIAVI = ["topology", "num_flows", "seed", "pct_mod", "epoch"]

    esaustiva = df[df["strategy"] == g.OTTIMO]
    piv_ok = esaustiva.pivot_table(index=CHIAVI, columns="n", values="ok", aggfunc="first").dropna()
    chiavi_comuni = piv_ok[(piv_ok == True).all(axis=1)].index

    comune = df.set_index(CHIAVI)
    comune = comune[comune.index.isin(chiavi_comuni)].reset_index()
    agg = comune.groupby(["strategy", "topology", "num_flows", "n"])["T_CR"].agg(["mean", "count"]).reset_index()
    agg.loc[agg["count"] < SOGLIA_CAMPIONI, "mean"] = float("nan")

    fig, assi = plt.subplots(len(NUM_FLOWS), len(g.TOPOLOGIE_ORDINATE), figsize=(12, 14), sharex=True, sharey="row")
    fig.patch.set_facecolor("white")

    for riga, num_flows in enumerate(NUM_FLOWS):
        for colonna, topo in enumerate(g.TOPOLOGIE_ORDINATE):
            ax = assi[riga][colonna]
            for strat in g.STRATEGIE:
                stile = g.STRATEGY_STYLES[strat]
                sotto = (agg[(agg["strategy"] == strat) & (agg["topology"] == topo) & (agg["num_flows"] == num_flows)]
                         .set_index("n").reindex(NODI))
                ax.plot(NODI, sotto["mean"], linestyle=stile["linestyle"], marker=stile["marker"],
                        color=stile["color"], lw=1.8, ms=4.5, label=g.NOME_STRATEGIA[strat])
            ax.set_xticks(NODI)
            if riga == 0:
                ax.set_title(g.NOME_TOPO[topo], fontsize=TITLE_FS, fontweight="bold", color=g.INK)
            g._pulisci_assi(ax, alpha=0.2)
            ax.tick_params(axis="both", labelsize=16)
            if colonna == 0:
                ax.set_ylabel("Execution Time [s]", fontsize=LABEL_FS)
            if riga == len(NUM_FLOWS) - 1 and colonna == len(g.TOPOLOGIE_ORDINATE) // 2:
                ax.set_xlabel("Nodes", fontsize=LABEL_FS)

        assi[riga][-1].text(1.03, 0.5, f"flows = {num_flows}", transform=assi[riga][-1].transAxes, rotation=270,
                            ha="left", va="center", fontsize=LABEL_FS, fontweight="bold", color=g.INK)
        massimo_riga = agg[agg["num_flows"] == num_flows]["mean"].max()
        assi[riga][0].set_ylim(-0.04 * massimo_riga, massimo_riga * 1.1)
        handles, labels = assi[riga][0].get_legend_handles_labels()
        assi[riga][0].legend(handles, labels, frameon=False, fontsize=LEGEND_FS, loc="upper left")

    fig.tight_layout()
    g._salva(fig, "tempo_griglia_epoche_comuni.pdf")


# =================================================================== tabelle divario per epoca

def get_data_divario_per_epoca(df: pd.DataFrame, soglia: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divario euristica vs esaustiva sull'intera traiettoria (tutte le epoche
    delle sequenze), con le due strategie accoppiate per EPOCA (stessa
    topologia, n, flussi, seed, pct_mod, epoca, entrambe ok) invece che per
    sequenza completa: un timeout scarta solo quell'epoca.

    - FF: sum(N_KO - N_RR) / sum(num_flows) -- denominatore uguale per le due
      strategie: dopo l'epoca 0 i KO differiscono, e normalizzare sui propri
      KO premierebbe chi ne genera di piu' (e piu' facili). Divario in punti %.
    - DF: sum(diff_simm_tot) / sum(N_RR), solo sulle epoche in cui entrambe
      hanno N_KO > 0 e N_RR == N_KO -- diff_simm_tot conta anche i flussi
      falliti (tornano su p_init, percorso vuoto), qui non ce ne sono.
    - DE: avg_latency media, stesso sottoinsieme di DF -- li' tutti i flussi
      sono instradati in entrambe, quindi la media e' sullo stesso insieme.
    DF e DE in differenza relativa %. Celle con meno di `soglia` epoche -> NaN.

    Returns:
        (divario, conteggi): divario nel formato atteso da
        _tabella_divario_semplice, conteggi = epoche usate per metrica.
    """
    euristica = g.EURISTICHE[0]
    chiavi = ["topology", "n", "num_flows", "seed", "pct_mod", "epoch"]
    colonne = ["N_KO", "N_RR", "diff_simm_tot", "avg_latency"]

    ok = df[df["ok"] & df["strategy"].isin([euristica, g.OTTIMO])]
    piv = ok.pivot_table(index=chiavi, columns="strategy", values=colonne).dropna()
    e = piv.xs(euristica, axis=1, level="strategy").reset_index()
    o = piv.xs(g.OTTIMO, axis=1, level="strategy").reset_index()
    ep = e.merge(o, on=chiavi, suffixes=("_h", "_o"))
    ep["flussi_tot"] = ep["num_flows"]   # copia: num_flows e' anche chiave di raggruppamento

    tutte_riparate = ((ep["N_KO_h"] > 0) & (ep["N_RR_h"] == ep["N_KO_h"])
                      & (ep["N_KO_o"] > 0) & (ep["N_RR_o"] == ep["N_KO_o"]))
    ep_df = ep[tutte_riparate]
    ep_de = ep_df

    def per_cella(gruppo: list[str]) -> pd.DataFrame:
        ff = ep.groupby(gruppo).apply(lambda s: pd.Series({
            "ff_h": (s["N_KO_h"] - s["N_RR_h"]).sum() / s["flussi_tot"].sum() * 100,
            "ff_o": (s["N_KO_o"] - s["N_RR_o"]).sum() / s["flussi_tot"].sum() * 100,
            "N_FF": len(s)}), include_groups=False)
        df_ = ep_df.groupby(gruppo).apply(lambda s: pd.Series({
            "df_h": s["diff_simm_tot_h"].sum() / s["N_RR_h"].sum(),
            "df_o": s["diff_simm_tot_o"].sum() / s["N_RR_o"].sum(),
            "N_DF": len(s)}), include_groups=False)
        de = ep_de.groupby(gruppo).agg(de_h=("avg_latency_h", "mean"), de_o=("avg_latency_o", "mean"),
                                        N_DE=("epoch", "size"))
        r = ff.join(df_, how="left").join(de, how="left").reset_index()
        r["diff_flussi_persi_pct"] = (r["ff_h"] - r["ff_o"]).where(r["N_FF"] >= soglia)
        r["pct_diff_costo"] = ((r["df_h"] - r["df_o"]) / r["df_o"] * 100).where(r["N_DF"] >= soglia)
        r["pct_diff_lat"] = ((r["de_h"] - r["de_o"]) / r["de_o"] * 100).where(r["N_DE"] >= soglia)
        r[["N_FF", "N_DF", "N_DE"]] = r[["N_FF", "N_DF", "N_DE"]].fillna(0).astype(int)
        return r

    per_topo = per_cella(["topology", "n", "num_flows"])
    generale = per_cella(["n", "num_flows"]).assign(topology="tutte")
    tutto = pd.concat([per_topo, generale], ignore_index=True).assign(strategy=euristica)
    conteggi = tutto[["topology", "n", "num_flows", "N_FF", "N_DF", "N_DE"]]
    return tutto, conteggi


def _tabella_divario_epoca(dati: pd.DataFrame, nome_file: str) -> None:
    """Stesso layout di _tabella_divario_semplice, ma con tutte le celle
    (n, flussi, metrica) anche se vuote, e il numero di epoche usate (N)
    sotto ogni valore."""
    METRICHE = [("diff_flussi_persi_pct", "N_FF"), ("pct_diff_costo", "N_DF"), ("pct_diff_lat", "N_DE")]
    etichette = [g.ETICHETTA_METRICA[c] for c, _ in METRICHE]
    dati = dati.set_index(["num_flows", "n"])
    num_flussi, nodi, n_metrica = NUM_FLOWS, NODI, len(METRICHE)

    LARGH_ETICH, LARGH_DATO, ALT_HEADER, ALT_DATO = 1.15, 1.0, 0.85, 0.72
    x_bordi = [0, LARGH_ETICH] + [LARGH_ETICH + k * LARGH_DATO for k in range(1, len(nodi) * n_metrica + 1)]
    y_bordi = [0, ALT_HEADER, 2 * ALT_HEADER] + [2 * ALT_HEADER + k * ALT_DATO for k in range(1, len(num_flussi) + 1)]
    largh_tot, alt_tot = x_bordi[-1], y_bordi[-1]

    ALT_LEGENDA = 0.35
    fig, ax = plt.subplots(figsize=(largh_tot * 0.9 + 0.5, (alt_tot + ALT_LEGENDA) * 0.9 + 0.5))
    ax.set_xlim(0, largh_tot)
    ax.set_ylim(alt_tot, -ALT_LEGENDA)
    ax.axis("off")
    ax.text((x_bordi[0] + x_bordi[-1]) / 2, -ALT_LEGENDA / 2, LEGENDA_METRICHE + "      N = epochs used",
            ha="center", va="center", fontsize=8.5, color=g.INK)

    def testo(x0, y0, x1, y1, s, grassetto=False, fontsize=9, colore=g.INK, dy=0.0):
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, s, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if grassetto else "normal", color=colore, zorder=3)

    testo(x_bordi[0], y_bordi[0], x_bordi[1], y_bordi[2], "Flussi /\nNodi", grassetto=True, fontsize=8)
    for i, n in enumerate(nodi):
        testo(x_bordi[1 + i * n_metrica], y_bordi[0], x_bordi[1 + (i + 1) * n_metrica], y_bordi[1], f"n = {n}", grassetto=True)
        for j, m in enumerate(etichette):
            testo(x_bordi[1 + i * n_metrica + j], y_bordi[1], x_bordi[1 + i * n_metrica + j + 1], y_bordi[2], m)

    for i, nf in enumerate(num_flussi):
        testo(x_bordi[0], y_bordi[2 + i], x_bordi[1], y_bordi[3 + i], f"flussi = {nf}", grassetto=True)
        for k, n in enumerate(nodi):
            riga = dati.loc[(nf, n)] if (nf, n) in dati.index else None
            for j, (col, col_n) in enumerate(METRICHE):
                x0, x1 = x_bordi[1 + k * n_metrica + j], x_bordi[2 + k * n_metrica + j]
                v = riga[col] if riga is not None else float("nan")
                cnt = int(riga[col_n]) if riga is not None else 0
                testo(x0, y_bordi[2 + i], x1, y_bordi[3 + i], f"{v:+.1f}%" if pd.notna(v) else "--",
                      grassetto=(pd.notna(v) and round(v, 1) == 0.0), dy=-0.1)
                testo(x0, y_bordi[2 + i], x1, y_bordi[3 + i], f"N={cnt}", fontsize=6.5, colore=g.INK_SOFT, dy=0.17)

    y_piene = {y_bordi[0]} | set(y_bordi[2:])
    x_piene = {x_bordi[0], x_bordi[1], x_bordi[-1]} | {x_bordi[1 + i * n_metrica] for i in range(len(nodi) + 1)}
    L = dict(color=g.INK, solid_capstyle="butt", clip_on=False, zorder=2)
    for y in y_bordi:
        if y in y_piene:
            ax.plot([x_bordi[0], x_bordi[-1]], [y, y], lw=1.8, **L)
        else:
            ax.plot([x_bordi[1], x_bordi[-1]], [y, y], lw=0.4, **L)
    for x in x_bordi:
        if x in x_piene:
            ax.plot([x, x], [y_bordi[0], y_bordi[-1]], lw=1.8, **L)
        else:
            ax.plot([x, x], [y_bordi[1], y_bordi[-1]], lw=0.4, **L)

    percorso = g.OUT_DIR / nome_file
    fig.savefig(percorso, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Salvato: {percorso}")


def plot_tabelle_divario_per_epoca(df: pd.DataFrame, solo_generale: bool = False) -> pd.DataFrame:
    """Come plot_tabelle_divario ma con get_data_divario_per_epoca; file con
    suffisso _epoca, le tabelle per sequenza restano invariate."""
    dati, conteggi = get_data_divario_per_epoca(df)
    for topo in (["tutte"] if solo_generale else [*g.TOPOLOGIE_ORDINATE, "tutte"]):
        nome = "generale" if topo == "tutte" else topo
        _tabella_divario_epoca(dati[dati["topology"] == topo], f"tabella_divario_epoca_{nome}.pdf")
    return conteggi


def main() -> None:
    df = pd.read_csv(g.CSV_PATH)
    plot_accuratezza(df)
    plot_tabelle_divario_per_epoca(df, solo_generale=True)
    plot_tabelle_divario(df, solo_generale=True)
    plot_tempo_griglia_flussi_topologia(df)
    plot_tempo_griglia_topologia_flussi(df)   # produce anche la vista flussi6
    print("fatto")


if __name__ == "__main__":
    main()
