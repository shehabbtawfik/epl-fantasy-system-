#!/usr/bin/env python3
"""
FPL Optimizer — Live Dashboard
Fetches real-time data from the Fantasy Premier League API and runs
an Integer Linear Programming optimizer to build the best possible squad.

No CSV files required. Works anywhere, including Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FPL Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.8rem; color: #aaa; margin-top: 0.2rem; }
    .player-card {
        background: #1a1a2e;
        border-left: 3px solid #00d4aa;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .captain-badge { background: #ffd700; color: #000; border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 700; }
    .vc-badge { background: #c0c0c0; color: #000; border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 700; }
    .position-GKP { border-left-color: #FFD700; }
    .position-DEF { border-left-color: #00FF88; }
    .position-MID { border-left-color: #00BFFF; }
    .position-FWD { border-left-color: #FF6B6B; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
FPL_API = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_COLORS = {"GKP": "#FFD700", "DEF": "#00FF88", "MID": "#00BFFF", "FWD": "#FF6B6B"}

# ─── Data layer ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_fpl_data() -> pd.DataFrame:
    """Pull live player data from the FPL bootstrap API."""
    resp = requests.get(FPL_API, timeout=15,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    raw = resp.json()

    teams = {t["id"]: t["short_name"] for t in raw["teams"]}
    players = []
    for p in raw["elements"]:
        players.append({
            "id":            p["id"],
            "name":          p["web_name"],
            "full_name":     f"{p['first_name']} {p['second_name']}",
            "position":      POSITION_MAP[p["element_type"]],
            "team":          teams[p["team"]],
            "price":         p["now_cost"] / 10,
            "ep_next":       float(p["ep_next"] or 0),
            "ep_this":       float(p["ep_this"] or 0),
            "form":          float(p["form"] or 0),
            "total_points":  p["total_points"],
            "points_per_game": float(p["points_per_game"] or 0),
            "selected_pct":  float(p["selected_by_percent"] or 0),
            "status":        p["status"],
            "chance_next":   p["chance_of_playing_next_round"],
            "minutes":       p["minutes"],
            "goals":         p["goals_scored"],
            "assists":       p["assists"],
            "clean_sheets":  p["clean_sheets"],
            "bonus":         p["bonus"],
            "transfers_in":  p["transfers_in_event"],
            "transfers_out": p["transfers_out_event"],
        })

    df = pd.DataFrame(players)
    df["value_score"] = df["ep_next"] / df["price"].clip(lower=0.1)
    df["combined_score"] = (df["ep_next"] * 0.6 + df["form"] * 0.4)
    return df


def available(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "a"].copy()


# ─── Optimizer ───────────────────────────────────────────────────────────────
@dataclass
class Squad:
    squad: List[dict] = field(default_factory=list)
    starting_xi: List[dict] = field(default_factory=list)
    bench: List[dict] = field(default_factory=list)
    captain: Optional[dict] = None
    vice_captain: Optional[dict] = None
    total_cost: float = 0.0
    total_xpts: float = 0.0
    formation: str = ""
    valid: bool = False
    error: str = ""


def optimize_squad(
    df: pd.DataFrame,
    budget: float = 100.0,
    strategy: str = "balanced",
    max_per_club: int = 3,
) -> Squad:
    avail = available(df)

    # Score per strategy
    score_col = {
        "balanced":     "ep_next",
        "premium":      "total_points",
        "value":        "value_score",
        "differential": "combined_score",
        "form":         "form",
    }.get(strategy, "ep_next")

    players = avail.reset_index(drop=True)
    n = len(players)
    if n < 15:
        sq = Squad(); sq.error = "Not enough available players."; return sq

    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    # Objective
    prob += pulp.lpSum(players.loc[i, score_col] * x[i] for i in range(n))

    # Budget
    prob += pulp.lpSum(players.loc[i, "price"] * x[i] for i in range(n)) <= budget

    # Squad size = 15
    prob += pulp.lpSum(x) == 15

    # Positional quotas
    for pos, (mn, mx) in {"GKP": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)}.items():
        idx = players[players["position"] == pos].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) >= mn
        prob += pulp.lpSum(x[i] for i in idx) <= mx

    # Max per club
    for team in players["team"].unique():
        idx = players[players["team"] == team].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) <= max_per_club

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        sq = Squad(); sq.error = "Optimisation failed — try adjusting budget."; return sq

    selected = players[[pulp.value(x[i]) == 1 for i in range(n)]].copy()
    selected = selected.reset_index(drop=True)

    # Build starting XI: best 11 by score (1 GKP, valid formation)
    gk = selected[selected["position"] == "GKP"].nlargest(1, score_col)
    field_players = selected[selected["position"] != "GKP"]
    best_field = field_players.nlargest(10, score_col)
    starting = pd.concat([gk, best_field])
    bench_players = selected[~selected.index.isin(starting.index)]

    # Validate formation
    def_count = len(starting[starting["position"] == "DEF"])
    mid_count = len(starting[starting["position"] == "MID"])
    fwd_count = len(starting[starting["position"] == "FWD"])
    formation = f"1-{def_count}-{mid_count}-{fwd_count}"

    captain = starting.nlargest(1, score_col).iloc[0].to_dict()
    vc_pool = starting[starting["name"] != captain["name"]]
    vice_captain = vc_pool.nlargest(1, score_col).iloc[0].to_dict() if len(vc_pool) else captain

    return Squad(
        squad=selected.to_dict("records"),
        starting_xi=starting.to_dict("records"),
        bench=bench_players.to_dict("records"),
        captain=captain,
        vice_captain=vice_captain,
        total_cost=round(selected["price"].sum(), 1),
        total_xpts=round(selected[score_col].sum(), 2),
        formation=formation,
        valid=True,
    )


# ─── UI helpers ──────────────────────────────────────────────────────────────
def fmt_price(v): return f"£{v:.1f}m"
def fmt_pct(v):   return f"{v:.1f}%"
def pos_emoji(p): return {"GKP": "🥅", "DEF": "🛡️", "MID": "⚽", "FWD": "🎯"}.get(p, "")

def player_row(p: dict, is_captain=False, is_vc=False):
    badge = ""
    if is_captain:   badge = ' <span class="captain-badge">C</span>'
    elif is_vc:      badge = ' <span class="vc-badge">VC</span>'
    pos_class = f"position-{p['position']}"
    st.markdown(
        f'<div class="player-card {pos_class}">'
        f'{pos_emoji(p["position"])} <b>{p["name"]}</b> {badge}'
        f'<span style="float:right; color:#aaa;">{fmt_price(p["price"])} &nbsp;|&nbsp; '
        f'xPts: <b style="color:#00d4aa">{p["ep_next"]:.1f}</b></span>'
        f'<br><small style="color:#777">{p["team"]} · {p["position"]}</small>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_squad(squad: Squad):
    col1, col2, col3 = st.columns(3)
    col1.metric("Formation", squad.formation)
    col2.metric("Total Cost", fmt_price(squad.total_cost))
    col3.metric("Total xPts", f"{squad.total_xpts:.1f}")

    st.markdown("#### Starting XI")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        players = [p for p in squad.starting_xi if p["position"] == pos]
        for p in players:
            player_row(
                p,
                is_captain=(squad.captain and p["name"] == squad.captain["name"]),
                is_vc=(squad.vice_captain and p["name"] == squad.vice_captain["name"]),
            )

    st.markdown("#### Bench")
    for p in squad.bench:
        player_row(p)

    with st.expander("📥 Download squad as CSV"):
        df_dl = pd.DataFrame(squad.squad)
        st.download_button(
            "Download",
            df_dl.to_csv(index=False),
            f"fpl_squad_{squad.formation}.csv",
            "text/csv",
        )


# ─── Pages ───────────────────────────────────────────────────────────────────
def page_home(df: pd.DataFrame):
    st.title("⚽ FPL Optimizer Dashboard")
    st.caption("Live data from the official Fantasy Premier League API · Refreshes every 5 min")

    avail = available(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Players",     len(df))
    c2.metric("Available",         len(avail))
    c3.metric("Avg Price",         fmt_price(avail["price"].mean()))
    c4.metric("Avg xPts (next GW)", f"{avail['ep_next'].mean():.2f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🌟 Top 5 by xPts")
        top5 = avail.nlargest(5, "ep_next")[["name", "position", "team", "price", "ep_next", "selected_pct"]]
        top5.columns = ["Player", "Pos", "Team", "Price", "xPts", "Ownership%"]
        top5["Price"] = top5["Price"].apply(fmt_price)
        top5["Ownership%"] = top5["Ownership%"].apply(fmt_pct)
        st.dataframe(top5, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("💎 Top 5 by Value (xPts/£)")
        top_val = avail.nlargest(5, "value_score")[["name", "position", "team", "price", "ep_next", "value_score"]]
        top_val.columns = ["Player", "Pos", "Team", "Price", "xPts", "xPts/£"]
        top_val["Price"] = top_val["Price"].apply(fmt_price)
        top_val["xPts/£"] = top_val["xPts/£"].round(2)
        st.dataframe(top_val, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📈 Price vs xPts (all available players)")
    fig = px.scatter(
        avail, x="price", y="ep_next", color="position",
        hover_data=["name", "team", "selected_pct"],
        color_discrete_map=POSITION_COLORS,
        labels={"price": "Price (£m)", "ep_next": "Expected Points (next GW)"},
        height=420,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ccc", legend_title="Position")
    st.plotly_chart(fig, use_container_width=True)


def page_optimizer(df: pd.DataFrame):
    st.title("🎯 Squad Optimizer")
    st.markdown("Builds the best 15-man squad using **Integer Linear Programming** — fully FPL-rules compliant.")

    with st.sidebar:
        st.header("⚙️ Settings")
        budget = st.slider("Budget (£m)", 90.0, 100.0, 100.0, 0.5)
        strategy = st.selectbox(
            "Strategy",
            ["balanced", "premium", "value", "differential", "form"],
            help="balanced=ep_next | premium=total pts | value=pts/£ | differential=low ownership | form=recent",
        )
        max_per_club = st.slider("Max per club", 1, 3, 3)
        run = st.button("🚀 Optimize", type="primary", use_container_width=True)

    if run:
        with st.spinner("Running ILP optimizer…"):
            result = optimize_squad(df, budget=budget, strategy=strategy, max_per_club=max_per_club)
        if result.valid:
            st.success("✅ Optimal squad found!")
            render_squad(result)
        else:
            st.error(f"❌ {result.error}")
    else:
        st.info("👈 Configure settings in the sidebar and hit **Optimize** to generate your squad.")


def page_players(df: pd.DataFrame):
    st.title("📊 Player Analysis")

    avail = available(df)

    with st.sidebar:
        st.header("🔍 Filters")
        positions = st.multiselect("Position", ["GKP", "DEF", "MID", "FWD"],
                                   default=["GKP", "DEF", "MID", "FWD"])
        price_range = st.slider("Price (£m)", float(avail["price"].min()),
                                float(avail["price"].max()),
                                (float(avail["price"].min()), float(avail["price"].max())))
        sort_by = st.selectbox("Sort by", ["ep_next", "total_points", "form", "value_score", "selected_pct"],
                               format_func=lambda x: {
                                   "ep_next": "xPts (next GW)", "total_points": "Total Points",
                                   "form": "Form", "value_score": "Value (xPts/£)",
                                   "selected_pct": "Ownership %"
                               }[x])

    filtered = avail[
        (avail["position"].isin(positions)) &
        (avail["price"] >= price_range[0]) &
        (avail["price"] <= price_range[1])
    ].sort_values(sort_by, ascending=False)

    st.caption(f"{len(filtered)} players shown")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered, x="price", y="ep_next", color="position",
            hover_data=["name", "team"],
            color_discrete_map=POSITION_COLORS,
            title="Price vs xPts",
            labels={"price": "Price (£m)", "ep_next": "xPts"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            filtered.groupby("position")["ep_next"].mean().reset_index(),
            x="position", y="ep_next", color="position",
            color_discrete_map=POSITION_COLORS,
            title="Average xPts by Position",
            labels={"ep_next": "Avg xPts", "position": ""},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#ccc", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    display = filtered[["name", "position", "team", "price", "ep_next", "form",
                         "total_points", "selected_pct", "value_score"]].copy()
    display.columns = ["Player", "Pos", "Team", "Price", "xPts", "Form",
                       "Total Pts", "Ownership%", "xPts/£"]
    display["Price"] = display["Price"].apply(fmt_price)
    display["Ownership%"] = display["Ownership%"].apply(fmt_pct)
    display["xPts/£"] = display["xPts/£"].round(2)

    st.dataframe(display.head(50), use_container_width=True, hide_index=True)


def page_watchlists(df: pd.DataFrame):
    st.title("📋 Positional Watchlists")
    avail = available(df)

    tabs = st.tabs(["🥅 Goalkeepers", "🛡️ Defenders", "⚽ Midfielders", "🎯 Forwards"])
    limits = {"GKP": 15, "DEF": 25, "MID": 25, "FWD": 20}

    for tab, pos in zip(tabs, ["GKP", "DEF", "MID", "FWD"]):
        with tab:
            top = avail[avail["position"] == pos].nlargest(limits[pos], "ep_next")
            display = top[["name", "team", "price", "ep_next", "form",
                           "total_points", "selected_pct"]].copy()
            display.columns = ["Player", "Team", "Price", "xPts", "Form", "Total Pts", "Ownership%"]
            display["Price"] = display["Price"].apply(fmt_price)
            display["Ownership%"] = display["Ownership%"].apply(fmt_pct)
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button(
                f"📥 Download {pos} watchlist",
                top.to_csv(index=False),
                f"watchlist_{pos.lower()}.csv",
                "text/csv",
            )


def page_top50(df: pd.DataFrame):
    st.title("🏆 Top 50 Players")
    avail = available(df)

    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.slider("Show top N", 10, 50, 25)
    with col2:
        pos_filter = st.multiselect("Filter by position", ["GKP", "DEF", "MID", "FWD"],
                                    default=["GKP", "DEF", "MID", "FWD"])

    top = avail[avail["position"].isin(pos_filter)].nlargest(n, "ep_next")
    top = top.reset_index(drop=True)
    top.index += 1

    display = top[["name", "position", "team", "price", "ep_next", "form",
                   "total_points", "selected_pct", "value_score"]].copy()
    display.columns = ["Player", "Pos", "Team", "Price", "xPts", "Form",
                       "Total Pts", "Ownership%", "xPts/£"]
    display["Price"] = display["Price"].apply(fmt_price)
    display["Ownership%"] = display["Ownership%"].apply(fmt_pct)
    display["xPts/£"] = display["xPts/£"].round(2)
    display.index.name = "Rank"

    st.dataframe(display, use_container_width=True)

    fig = px.bar(
        top.head(20), x="name", y="ep_next", color="position",
        color_discrete_map=POSITION_COLORS,
        title="Top 20 Players — xPts Next GW",
        labels={"name": "", "ep_next": "xPts"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", showlegend=True, xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


def page_transfers(df: pd.DataFrame):
    st.title("🔄 Transfer Trends")
    avail = available(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Most Transferred In")
        top_in = avail.nlargest(15, "transfers_in")[
            ["name", "position", "team", "price", "ep_next", "transfers_in"]
        ].copy()
        top_in.columns = ["Player", "Pos", "Team", "Price", "xPts", "Transfers In ↑"]
        top_in["Price"] = top_in["Price"].apply(fmt_price)
        st.dataframe(top_in, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📉 Most Transferred Out")
        top_out = avail.nlargest(15, "transfers_out")[
            ["name", "position", "team", "price", "ep_next", "transfers_out"]
        ].copy()
        top_out.columns = ["Player", "Pos", "Team", "Price", "xPts", "Transfers Out ↓"]
        top_out["Price"] = top_out["Price"].apply(fmt_price)
        st.dataframe(top_out, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔥 Net Transfer Gain (In − Out)")
    avail = avail.copy()
    avail["net_transfers"] = avail["transfers_in"] - avail["transfers_out"]
    net = avail.nlargest(20, "net_transfers")
    fig = px.bar(
        net, x="name", y="net_transfers", color="position",
        color_discrete_map=POSITION_COLORS,
        title="Top 20 Net Transfer Gainers This GW",
        labels={"name": "", "net_transfers": "Net Transfers"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    # Sidebar nav
    st.sidebar.title("⚽ FPL Optimizer")
    st.sidebar.caption("Live data · ILP-powered")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "🎯 Squad Optimizer", "📊 Player Analysis",
         "📋 Watchlists", "🏆 Top 50", "🔄 Transfer Trends"],
    )
    st.sidebar.divider()
    st.sidebar.info(
        "Data is fetched live from the **official FPL API** and cached for 5 minutes. "
        "Squad optimisation uses **Integer Linear Programming** (PuLP/CBC)."
    )

    # Load data
    with st.spinner("🔄 Fetching live FPL data…"):
        try:
            df = fetch_fpl_data()
        except Exception as e:
            st.error(f"Failed to fetch FPL data: {e}")
            st.stop()

    # Route
    if page == "🏠 Home":
        page_home(df)
    elif page == "🎯 Squad Optimizer":
        page_optimizer(df)
    elif page == "📊 Player Analysis":
        page_players(df)
    elif page == "📋 Watchlists":
        page_watchlists(df)
    elif page == "🏆 Top 50":
        page_top50(df)
    elif page == "🔄 Transfer Trends":
        page_transfers(df)


if __name__ == "__main__":
    main()
