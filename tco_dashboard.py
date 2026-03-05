import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Quant16 Portfolio Optimization Model - Digital Assets", layout="wide", initial_sidebar_state="expanded")

DARK_BG, CARD_BG, BORDER, TEXT, MUTED, ACCENT = "#e8edf2", "#f0f4f8", "#c9d3de", "#1e293b", "#64748b", "#2563eb"
DEPT_COLORS = {"Buildings":"#ea580c","Finance":"#0891b2","HR":"#7c3aed","IT":"#2563eb","Public Health":"#dc2626","Public Safety":"#ca8a04","Transportation":"#059669","Water":"#2563eb"}
WT_COLORS = {"Access":"#db2777","Analysis":"#7c3aed","Collaboration":"#0891b2","Communications":"#ca8a04","Data":"#059669","External Facing":"#ea580c","Integration":"#dc2626","Workflow":"#2563eb"}
COMP_NAMES = ["Core","Interface","Workflow","Logic","Integration","Security"]
SECTOR_COLORS = {"Regulatory":"#ea580c","Community Services":"#0891b2","Legislative and Election":"#7c3aed",
    "Public Safety":"#dc2626","Infrastructure Services":"#ca8a04","Finance and Administration":"#059669","City Development":"#2563eb"}
SIZE_COLORS = {"Large":"#2563eb","Mid-Size":"#ca8a04","Small":"#059669","Internal":"#64748b"}

EARLY_PAY_PRESETS = {
    "Conservative": {
        "desc": "Minimal negotiation, standard terms",
        "disc": {"Large": 1.0, "Mid-Size": 2.0, "Small": 3.0},
        "accept": {"Large": 30, "Mid-Size": 50, "Small": 70},
        "coc": 6.0,
    },
    "Moderate": {
        "desc": "Active procurement, market-rate discounts",
        "disc": {"Large": 1.5, "Mid-Size": 3.0, "Small": 5.0},
        "accept": {"Large": 50, "Mid-Size": 70, "Small": 90},
        "coc": 5.0,
    },
    "Aggressive": {
        "desc": "Strong cash position, maximum leverage",
        "disc": {"Large": 2.5, "Mid-Size": 5.0, "Small": 8.0},
        "accept": {"Large": 70, "Mid-Size": 90, "Small": 95},
        "coc": 4.0,
    },
}

MARGIN_TIERS = {
    "Low":  {"max": 15, "mult": 0.7},   # margin < 15% → 0.7x discount
    "Med":  {"max": 30, "mult": 1.0},   # margin 15–30% → 1.0x (baseline)
    "High": {"max": 100, "mult": 1.5},  # margin > 30% → 1.5x discount
}

def margin_multiplier(margin_pct):
    """Return discount multiplier based on vendor gross margin tier."""
    if pd.isna(margin_pct) or margin_pct <= 0:
        return 1.0
    if margin_pct < 15:
        return 0.7
    if margin_pct <= 30:
        return 1.0
    return 1.5

NET_MARGIN_RATIO = 0.6  # net ≈ 60% of gross (SGA ~40%)

INCENTIVE_TIERS = {
    "None":   {"max": 0,  "rebate": 0.0},
    "Low":    {"max": 5,  "rebate": 0.02},   # 0–5pp above TSM → 2% rebate
    "Medium": {"max": 15, "rebate": 0.05},   # 5–15pp above TSM → 5% rebate
    "High":   {"max": 100,"rebate": 0.10},   # 15pp+ above TSM → 10% rebate
}

INCENTIVE_SCENARIOS = {
    "Conservative": {"Low": 0.01, "Medium": 0.03, "High": 0.05},
    "Aggressive":   {"Low": 0.02, "Medium": 0.05, "High": 0.10},
}
CHURN_SCENARIOS = {
    "Conservative": {"Low": 0.005, "Medium": 0.015, "High": 0.03},
    "Aggressive":   {"Low": 0.01,  "Medium": 0.03,  "High": 0.05},
}

TIER_COLORS = {"None": "#94a3b8", "Low": "#2563eb", "Medium": "#ca8a04", "High": "#dc2626"}

CHURN_TIERS = {
    "None":   {"churn": 0.00, "protection": 0.00},  # no headroom → no leverage
    "Low":    {"churn": 0.10, "protection": 0.01},   # 0–5pp → 10% churn risk → 1% protection
    "Medium": {"churn": 0.25, "protection": 0.03},   # 5–15pp → 25% churn risk → 3% protection
    "High":   {"churn": 0.40, "protection": 0.05},   # 15pp+ → 40% churn risk → 5% protection
}

QUADRANT_COLORS = {
    "High GM · High NM": "#059669",  # green — strong leverage + LTC candidate
    "Low GM · High NM":  "#2563eb",  # blue — operationally efficient, limited price lever
    "High GM · Low NM":  "#ca8a04",  # yellow — price lever exists but thin ops
    "Low GM · Low NM":   "#94a3b8",  # gray — no headroom, monitor only
}

st.markdown(f"""<style>
    .stApp {{ background-color: {DARK_BG}; }}
    .stSidebar > div {{ background-color: {CARD_BG}; }}
    div[data-testid="stMetric"] label {{ color: {MUTED} !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-family: 'JetBrains Mono', monospace !important; }}
    h1, h2, h3 {{ color: {TEXT} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ background: {CARD_BG}; border-radius: 8px 8px 0 0; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED} !important; }}
    .stTabs [aria-selected="true"] {{ color: {TEXT} !important; }}
    .stRadio label, .stSelectbox label, .stSlider label, .stCheckbox label {{ color: {TEXT} !important; }}
    .stSidebar details summary span {{ white-space: pre-line !important; }}
</style>""", unsafe_allow_html=True)

def fmt(v):
    if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"

def hex_to_rgba(hx, a=0.5):
    h = hx.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def dark_layout(fig, height=380):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11), height=height, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(gridcolor=BORDER, showline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, showline=False, tickfont=dict(color=MUTED)))
    return fig


_CARD_ICONS = [
    "&#128295;", "&#128202;", "&#128176;", "&#128640;", "&#9889;", "&#128161;",
    "&#127942;", "&#128270;", "&#128200;", "&#128736;", "&#128640;", "&#9881;",
    "&#128279;", "&#128274;", "&#128300;", "&#128194;", "&#127793;", "&#128204;",
    "&#128203;", "&#128201;", "&#128187;", "&#128260;", "&#128269;", "&#128218;",
]
_SEG_ICONS = {
    "Budget": "&#128181;", "Vendor": "&#129309;", "Risk": "&#128737;",
    "Component": "&#128300;", "Categorization": "&#128194;",
    "Retrun Analytics": "&#128200;", "Investment Analytics": "&#128176;", "Risk Analytics": "&#128737;",
    "Financial Optimization": "&#128181;", "Work Modernization": "&#9889;", "Risk Mitigation": "&#128737;",
}

def _alpha_cards_html(categories, alpha_label, score_range_lo, score_range_hi, cat_labels=None, detail_rows=None):
    """Build HTML for Alpha Performer differentiator cards, segmented by category.
    detail_rows: optional dict {cat_name: [row dicts with KPI/org_fmt/score/alpha]} for KPI detail table."""
    has_any = any(r.get("scenario_desc") for rows in categories.values() for r in rows)
    if not has_any:
        return ""

    all_scores = [r["alpha_score"] for rows in categories.values() for r in rows if r.get("alpha_score")]
    if score_range_lo is None and all_scores:
        score_range_lo = min(all_scores)
    if score_range_hi is None and all_scores:
        score_range_hi = max(all_scores)
    range_str = f"{score_range_lo:.1f}–{score_range_hi:.1f}" if score_range_lo and score_range_hi else ""

    dimension_blocks = ""
    icon_idx = 0
    n_segments = 0
    for cat_name, cat_rows in categories.items():
        rows_with_desc = [r for r in cat_rows if r.get("scenario_desc")]
        if not rows_with_desc:
            continue
        n_segments += 1
        display_name = cat_labels.get(cat_name, cat_name) if cat_labels else cat_name
        seg_icon = _SEG_ICONS.get(cat_name, "&#128202;")

        # Segment summary stats
        seg_scores = [r["alpha_score"] for r in rows_with_desc]
        seg_avg = round(sum(seg_scores) / len(seg_scores), 1) if seg_scores else 0
        seg_reps = sorted({r.get("alpha_rep", "") for r in rows_with_desc if r.get("alpha_rep")})
        seg_reps_str = ", ".join(seg_reps[:3]) if seg_reps else "Top Performers"
        seg_avg_color = "#059669" if seg_avg >= 7 else "#ca8a04" if seg_avg >= 4 else "#dc2626"

        # Individual KPI cards
        n_cards = len(rows_with_desc)
        cols = 2 if n_cards <= 4 else 3
        kpi_cards = ""
        for r in rows_with_desc:
            icon = _CARD_ICONS[icon_idx % len(_CARD_ICONS)]
            icon_idx += 1
            sc = r["alpha_score"]
            sc_color = "#059669" if sc >= 7 else "#ca8a04" if sc >= 4 else "#dc2626"
            rep = r.get("alpha_rep", "")
            rep_html = f"<span style='color:{MUTED}; font-size:10px;'> ({rep})</span>" if rep else ""
            alpha_val = r.get("alpha", "")
            alpha_kpi_html = (f"<div style='font-size:11px; color:#0284c7; font-weight:600; "
                              f"margin-top:2px;'>Alpha KPI: {alpha_val}</div>") if alpha_val else ""
            kpi_cards += (
                f"<div style='background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; "
                f"padding:16px 18px; display:flex; flex-direction:column; gap:6px; min-width:0;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='font-size:13px; font-weight:600; color:{TEXT};'>"
                f"<span style='margin-right:6px;'>{icon}</span>{r['KPI']}{rep_html}</span>"
                f"<span style='background:{sc_color}; color:#ffffff; font-size:12px; font-weight:700; "
                f"padding:2px 10px; border-radius:12px; white-space:nowrap;'>{sc}</span>"
                f"</div>"
                f"<div style='font-size:11px; color:{MUTED}; line-height:1.5;'>{r['scenario_desc']}</div>"
                f"{alpha_kpi_html}"
                f"</div>"
            )

        # Each dimension = a <details> where <summary> IS the card
        dimension_blocks += (
            f"<details style='margin-bottom:10px;'>"
            f"<summary style='cursor:pointer; list-style:none; outline:none;'>"
            f"<div style='background:linear-gradient(135deg, rgba(37,99,235,0.08), rgba(5,150,105,0.06)); "
            f"border:1px solid {BORDER}; border-radius:10px; padding:14px 18px; "
            f"display:flex; align-items:center; justify-content:space-between; "
            f"transition:border-color 0.2s;'>"
            f"<div style='display:flex; align-items:center; gap:10px;'>"
            f"<span style='font-size:18px;'>{seg_icon}</span>"
            f"<div>"
            f"<div style='font-size:14px; font-weight:700; color:{TEXT};'>{display_name}</div>"
            f"<div style='font-size:11px; color:{MUTED};'>"
            f"{n_cards} KPI{'s' if n_cards != 1 else ''} &middot; "
            f"Led by {seg_reps_str} &middot; "
            f"<span style='font-style:italic;'>click to expand</span></div>"
            f"</div></div>"
            f"<span style='background:{seg_avg_color}; color:#ffffff; font-size:13px; font-weight:700; "
            f"padding:3px 14px; border-radius:12px;'>{seg_avg} avg</span>"
            f"</div></summary>"
            f"<div style='display:grid; grid-template-columns:repeat({cols}, 1fr); gap:12px; "
            f"margin:12px 4px 4px 4px;'>"
            f"{kpi_cards}</div>"
        )

        # Optional: KPI detail table inside the expanded section
        if detail_rows and cat_name in detail_rows:
            dt_rows = detail_rows[cat_name]
            if dt_rows:
                tbl_hdr = (
                    f"<tr>"
                    f"<th style='text-align:left; color:{MUTED}; font-size:11px; text-transform:uppercase; "
                    f"padding:8px 12px; border-bottom:1px solid {BORDER};'>KPI</th>"
                    f"<th style='text-align:center; color:{MUTED}; font-size:11px; text-transform:uppercase; "
                    f"padding:8px 12px; border-bottom:1px solid {BORDER};'>Organization KPI Value</th>"
                    f"<th style='text-align:center; color:{MUTED}; font-size:11px; text-transform:uppercase; "
                    f"padding:8px 12px; border-bottom:1px solid {BORDER};'>KPI Score</th>"
                    f"<th style='text-align:center; color:{MUTED}; font-size:11px; text-transform:uppercase; "
                    f"padding:8px 12px; border-bottom:1px solid {BORDER};'>Alpha Performer</th>"
                    f"</tr>"
                )
                tbl_body = ""
                for dr in dt_rows:
                    sc_val = dr.get("score", 0)
                    sc_color = "#059669" if sc_val >= 7 else "#ca8a04" if sc_val >= 4 else "#dc2626"
                    tbl_body += (
                        f"<tr>"
                        f"<td style='padding:8px 12px; color:{TEXT}; font-size:12px; "
                        f"border-bottom:1px solid {BORDER};'>{dr.get('KPI', '')}</td>"
                        f"<td style='text-align:center; padding:8px 12px; color:{TEXT}; font-size:12px; "
                        f"border-bottom:1px solid {BORDER};'>{dr.get('org_fmt', '')}</td>"
                        f"<td style='text-align:center; padding:8px 12px; color:{sc_color}; font-size:12px; "
                        f"font-weight:600; border-bottom:1px solid {BORDER};'>{sc_val}</td>"
                        f"<td style='text-align:center; padding:8px 12px; color:{MUTED}; font-size:12px; "
                        f"border-bottom:1px solid {BORDER};'>{dr.get('alpha', '')}</td>"
                        f"</tr>"
                    )
                dimension_blocks += (
                    f"<details style='margin:10px 4px 4px 4px;'>"
                    f"<summary style='cursor:pointer; color:{MUTED}; font-size:11px; padding:4px 0; "
                    f"list-style:revert;'>{display_name} — KPI Detail Table</summary>"
                    f"<table style='width:100%; border-collapse:collapse; margin-top:8px;'>"
                    f"{tbl_hdr}{tbl_body}</table>"
                    f"</details>"
                )

        dimension_blocks += "</details>"

    if not dimension_blocks:
        return ""

    return (
        f"<style>details > summary::-webkit-details-marker {{ display:none; }}</style>"
        f"<div style='background:rgba(240,244,248,0.95); border:1px solid {BORDER}; border-radius:14px; "
        f"padding:24px 28px; margin:16px 0 24px 0;'>"
        f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>"
        f"<span style='font-size:24px;'>&#127942;</span>"
        f"<span style='font-size:16px; font-weight:700; color:{TEXT};'>"
        f"Alpha Best Performer &mdash; {alpha_label}</span></div>"
        f"<div style='color:{MUTED}; font-size:12px; margin-bottom:18px;'>"
        f"Why these organizations score {range_str} across all dimensions</div>"
        f"{dimension_blocks}"
        f"</div>"
    )


# ── Roadmap renderer (shared by all P tabs) ──
_ROADMAP_STAGES = [
    ("Foundation",   "avg",   "#ca8a04", "&#128204;"),   # → Benchmark tier
    ("Advancement",  "top",   "#6d28d9", "&#128640;"),   # → Top tier
    ("Leadership",   "alpha", "#059669", "&#127942;"),   # → Alpha tier
]

def _render_roadmap(categories, tier_scores, cat_labels=None):
    """Render a 3-stage roadmap using tier_data from parse_benchmark_sheets().
    categories: {cat_name: [row dicts with 'KPI', 'score', 'tier_data']}
    tier_scores: {cat_name: {tier_key: score}}
    cat_labels: optional display-name overrides for category names.
    """
    cats = list(tier_scores.keys())
    if not cats:
        return ""

    # Current org score (weighted avg across KPIs)
    org_scores = [r["score"] for rows in categories.values() for r in rows if r.get("score") is not None]
    org_avg = round(np.mean(org_scores), 1) if org_scores else 0

    _tier_display = {"avg": "Benchmark", "top": "Top Performance", "alpha": "Alpha"}

    stage_blocks = ""
    for stage_name, target_tier, stage_color, stage_icon in _ROADMAP_STAGES:
        kpi_targets = []
        dim_blocks = ""
        n_achieved = 0
        n_total = 0

        for cat_name, cat_rows in categories.items():
            display_cat = cat_labels.get(cat_name, cat_name) if cat_labels else cat_name
            seg_icon = _SEG_ICONS.get(cat_name, "&#128202;")

            # Build per-KPI cards for this dimension
            dim_kpi_html = ""
            dim_cur_scores = []
            dim_tgt_scores = []
            dim_achieved = 0
            for r in cat_rows:
                n_total += 1
                cur = r.get("score") or 0
                td = r.get("tier_data", {}).get(target_tier, {})
                tgt = td.get("score")
                if tgt is None:
                    tgt = tier_scores.get(cat_name, {}).get(target_tier)
                tgt = tgt if tgt is not None else cur
                achieved = cur >= tgt
                if achieved:
                    n_achieved += 1
                    dim_achieved += 1
                kpi_targets.append(tgt if not achieved else cur)
                dim_cur_scores.append(cur)
                dim_tgt_scores.append(tgt if not achieved else cur)

                gap = round(tgt - cur, 1) if not achieved else 0
                scenario = td.get("scenario_desc", "")
                practice = td.get("practice_diff", "")

                opacity = "0.35" if achieved else "1"
                badge = (f"<span style='background:#059669; color:#fff; font-size:10px; font-weight:700; "
                         f"padding:1px 8px; border-radius:8px; margin-left:8px;'>Achieved</span>") if achieved else ""
                scenario_html = (f"<div style='font-size:11px; color:{MUTED}; line-height:1.4; margin-top:4px;'>"
                                 f"<b>Scenario:</b> {scenario}</div>") if scenario else ""
                practice_html = (f"<div style='font-size:11px; color:{MUTED}; line-height:1.4; margin-top:2px;'>"
                                 f"<b>Differentiator:</b> {practice}</div>") if practice else ""
                gap_html = (f"  <span style='color:{MUTED}; font-size:11px;'>(+{gap})</span>"
                            if gap > 0 else "")

                dim_kpi_html += (
                    f"<div style='opacity:{opacity}; background:{CARD_BG}; border:1px solid {BORDER}; "
                    f"border-radius:8px; padding:12px 16px; margin-bottom:8px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<span style='font-size:12px; font-weight:600; color:{TEXT};'>"
                    f"{r['KPI']}{badge}</span>"
                    f"<span style='font-size:12px; color:{TEXT};'>"
                    f"<span style='color:{MUTED};'>{cur:.1f}</span>"
                    f" &rarr; <span style='color:{stage_color}; font-weight:700;'>{tgt:.1f}</span>"
                    f"{gap_html}"
                    f"</span></div>"
                    f"{scenario_html}{practice_html}"
                    f"</div>"
                )

            # Dimension-level summary
            dim_cur_avg = round(np.mean(dim_cur_scores), 1) if dim_cur_scores else 0
            dim_tgt_avg = round(np.mean(dim_tgt_scores), 1) if dim_tgt_scores else 0
            dim_n = len(cat_rows)
            dim_action = dim_n - dim_achieved
            dim_pct = round(dim_achieved / dim_n * 100) if dim_n else 0
            dim_avg_color = "#059669" if dim_tgt_avg >= 7 else "#ca8a04" if dim_tgt_avg >= 4 else "#dc2626"

            dim_progress = (
                f"<div style='display:flex; align-items:center; gap:8px; margin-top:6px;'>"
                f"<div style='flex:1; height:4px; background:{BORDER}; border-radius:2px; overflow:hidden;'>"
                f"<div style='width:{dim_pct}%; height:100%; background:{stage_color}; border-radius:2px;'></div>"
                f"</div>"
                f"<span style='font-size:10px; color:{MUTED};'>{dim_pct}%</span>"
                f"</div>"
            )

            dim_blocks += (
                f"<details style='margin-bottom:8px;'>"
                f"<summary style='cursor:pointer; list-style:none; outline:none;'>"
                f"<div style='background:{CARD_BG}; border:1px solid {BORDER}; border-radius:8px; "
                f"padding:10px 14px; display:flex; align-items:center; justify-content:space-between;'>"
                f"<div style='display:flex; align-items:center; gap:8px;'>"
                f"<span style='font-size:16px;'>{seg_icon}</span>"
                f"<div>"
                f"<div style='font-size:13px; font-weight:600; color:{TEXT};'>{display_cat}</div>"
                f"<div style='font-size:10px; color:{MUTED};'>"
                f"{dim_action} of {dim_n} KPIs to improve &middot; "
                f"<span style='font-style:italic;'>click to expand</span></div>"
                f"</div></div>"
                f"<div style='text-align:right;'>"
                f"<div style='font-size:13px; font-weight:700; color:{dim_avg_color};'>"
                f"{dim_cur_avg} &rarr; {dim_tgt_avg}</div>"
                f"<div style='font-size:10px; color:{MUTED};'>Dimension Avg</div>"
                f"</div>"
                f"</div>"
                f"{dim_progress}"
                f"</summary>"
                f"<div style='margin:8px 4px 4px 12px;'>{dim_kpi_html}</div>"
                f"</details>"
            )

        projected = round(np.mean(kpi_targets), 1) if kpi_targets else org_avg
        n_action = n_total - n_achieved
        pct_done = round(n_achieved / n_total * 100) if n_total else 0

        # Progress bar
        progress_html = (
            f"<div style='display:flex; align-items:center; gap:10px; margin-top:8px;'>"
            f"<div style='flex:1; height:6px; background:{BORDER}; border-radius:3px; overflow:hidden;'>"
            f"<div style='width:{pct_done}%; height:100%; background:{stage_color}; border-radius:3px;'></div>"
            f"</div>"
            f"<span style='font-size:11px; color:{MUTED};'>{pct_done}% achieved</span>"
            f"</div>"
        )

        tier_label = _tier_display.get(target_tier, target_tier)
        stage_blocks += (
            f"<details style='margin-bottom:12px;'>"
            f"<summary style='cursor:pointer; list-style:none; outline:none;'>"
            f"<div style='background:linear-gradient(135deg, {hex_to_rgba(stage_color, 0.08)}, {hex_to_rgba(stage_color, 0.04)}); "
            f"border:1px solid {BORDER}; border-radius:10px; padding:14px 18px; "
            f"display:flex; align-items:center; justify-content:space-between; "
            f"transition:border-color 0.2s;'>"
            f"<div style='display:flex; align-items:center; gap:10px;'>"
            f"<span style='font-size:20px;'>{stage_icon}</span>"
            f"<div>"
            f"<div style='font-size:14px; font-weight:700; color:{TEXT};'>Stage: {stage_name}</div>"
            f"<div style='font-size:11px; color:{MUTED};'>"
            f"Target tier: <b>{tier_label}</b> &middot; "
            f"{n_action} KPI{'s' if n_action != 1 else ''} to improve &middot; "
            f"<span style='font-style:italic;'>click to expand</span></div>"
            f"</div></div>"
            f"<div style='text-align:right;'>"
            f"<div style='font-size:16px; font-weight:700; color:{stage_color};'>{projected} / 10</div>"
            f"<div style='font-size:10px; color:{MUTED};'>Projected Score</div>"
            f"</div>"
            f"</div>"
            f"{progress_html}"
            f"</summary>"
            f"<div style='margin:10px 4px 4px 4px;'>{dim_blocks}</div>"
            f"</details>"
        )

    return (
        f"<style>details > summary::-webkit-details-marker {{ display:none; }}</style>"
        f"<div style='background:rgba(240,244,248,0.95); border:1px solid {BORDER}; border-radius:14px; "
        f"padding:24px 28px; margin:16px 0 24px 0;'>"
        f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>"
        f"<span style='font-size:24px;'>&#128506;</span>"
        f"<span style='font-size:16px; font-weight:700; color:{TEXT};'>"
        f"Improvement Roadmap</span></div>"
        f"<div style='color:{MUTED}; font-size:12px; margin-bottom:18px;'>"
        f"Current org score: <b>{org_avg}</b> &mdash; 3 stages to reach Alpha performance</div>"
        f"{stage_blocks}"
        f"</div>"
    )


# ══════════════════════════
# DATA
# ══════════════════════════
@st.cache_data
def load_data(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    df = sheets["Digital_Asset_Inventory"]
    wt_model = sheets.get("Worktype_Component_Model", pd.DataFrame())
    vendor_ext = sheets.get("Vendor_Extended", pd.DataFrame())
    asset_rels = sheets.get("Asset_Relationships", pd.DataFrame())
    app_spec = None
    infra_spec = None
    for key in sheets:
        if "Application_Component_Specific" in key:
            app_spec = sheets[key]
        if "Infrastructure_Component_Specif" in key:
            infra_spec = sheets[key]
    infra_model = sheets.get("Infrastructure_Component_Model", pd.DataFrame())

    if not vendor_ext.empty and "Vendor_ID" in df.columns:
        df = df.merge(vendor_ext, on="Vendor_ID", how="left", suffixes=("", "_ve"))
        # Drop duplicate columns from vendor_ext that already exist in inventory
        ve_dups = [c for c in df.columns if c.endswith("_ve")]
        if ve_dups:
            df.drop(columns=ve_dups, inplace=True)

    # Year-by-year
    rows = []
    for _, r in df.iterrows():
        start = int(r["Initial_Year"])
        end = start + int(r["Estimated_Lifecycle_Years"]) - 1
        for y in range(int(df["Initial_Year"].min()), end + 1):
            init = float(r["Initial_Cost"]) if y == start else 0
            run = float(r["Annual_Run_Cost"]) if start <= y <= end else 0
            dev = float(r["Annual_Dev_Cost"]) if start <= y <= end else 0
            if init + run + dev > 0:
                rows.append({"year": y, "department": r["Department"], "worktype": r["Worktype"],
                             "layer": r["Layer"], "sector": r.get("Sector", "Unknown"),
                             "initCost": init, "runCost": run, "devCost": dev})
    ydf = pd.DataFrame(rows)

    # Component allocation
    alloc = None
    if not wt_model.empty and app_spec is not None:
        apps = df[df["Layer"] == "Applications"].copy()
        apps["TotalCost"] = apps["Initial_Cost"] + apps["Annual_Run_Cost"] + apps["Annual_Dev_Cost"]
        apps = apps.merge(wt_model, on="Worktype", how="left", suffixes=("", "_w"))
        id_col = [c for c in app_spec.columns if "ID" in c or "Id" in c][0]
        apps = apps.merge(app_spec, left_on="Asset_ID", right_on=id_col, how="left", suffixes=("", "_s"))
        alloc_rows = []
        for _, r in apps.iterrows():
            for comp in COMP_NAMES:
                w_col = comp + "_w" if comp + "_w" in r.index else comp
                s_col = comp + "_s" if comp + "_s" in r.index else comp
                weight = float(r[w_col])
                spec = int(r[s_col]) if pd.notna(r[s_col]) else 1
                alloc_rows.append({
                    "Asset_ID": r["Asset_ID"], "Department": r["Department"],
                    "Worktype": r["Worktype"], "Component": comp,
                    "Weight": weight, "Spec": spec,
                    "Init_Alloc": weight * float(r["Initial_Cost"]),
                    "Run_Alloc": weight * float(r["Annual_Run_Cost"]),
                    "Dev_Alloc": weight * float(r["Annual_Dev_Cost"]),
                    "Total_Alloc": weight * float(r["TotalCost"]),
                    "BV": r["Business_Value_Score_1to5"], "TH": r["Technical_Health_Score_1to5"],
                })
        alloc = pd.DataFrame(alloc_rows)

    # Infrastructure component allocation
    infra_alloc = None
    if not infra_model.empty and infra_spec is not None:
        infra_assets = df[df["Layer"] == "Infrastructure"].copy()
        infra_assets["TotalCost"] = infra_assets["Initial_Cost"] + infra_assets["Annual_Run_Cost"] + infra_assets["Annual_Dev_Cost"]
        infra_assets = infra_assets.merge(infra_model, on="Worktype", how="left", suffixes=("", "_w"))
        id_col_i = [c for c in infra_spec.columns if "ID" in c or "Id" in c][0]
        infra_assets = infra_assets.merge(infra_spec, left_on="Asset_ID", right_on=id_col_i, how="left", suffixes=("", "_s"))
        infra_comps = [c for c in infra_model.columns if c != "Worktype"]
        ialloc_rows = []
        for _, r in infra_assets.iterrows():
            for comp in infra_comps:
                w_col = comp + "_w" if comp + "_w" in r.index else comp
                s_col = comp + "_s" if comp + "_s" in r.index else comp
                weight = float(r.get(w_col, 0))
                if weight == 0:
                    continue
                spec = int(r.get(s_col, 1)) if pd.notna(r.get(s_col, 1)) else 1
                ialloc_rows.append({
                    "Asset_ID": r["Asset_ID"], "Department": r["Department"],
                    "Worktype": r["Worktype"], "Component": comp,
                    "Weight": weight, "Spec": spec,
                    "Init_Alloc": weight * float(r["Initial_Cost"]),
                    "Run_Alloc": weight * float(r["Annual_Run_Cost"]),
                    "Dev_Alloc": weight * float(r["Annual_Dev_Cost"]),
                    "Total_Alloc": weight * float(r["TotalCost"]),
                    "BV": r["Business_Value_Score_1to5"], "TH": r["Technical_Health_Score_1to5"],
                })
        if ialloc_rows:
            infra_alloc = pd.DataFrame(ialloc_rows)

    # AI Reduction factors from AI_Reduction.xlsx
    ai_reduction_app = {}
    ai_reduction_infra = {}
    ai_task_by_comp_app = {}   # {component: {task: factor}}
    ai_task_by_comp_infra = {}
    ai_task_categories = []
    ai_category_gap_app = {}
    ai_category_gap_infra = {}
    ai_file = Path("AI_Reduction.xlsx")
    if ai_file.exists():
        ai_sheets = pd.read_excel(ai_file, sheet_name=None)
        for sheet_name, target_dict, cat_dict, task_by_comp in [
            ("Application vs AI", ai_reduction_app, ai_category_gap_app, ai_task_by_comp_app),
            ("Infrastructure vs AI", ai_reduction_infra, ai_category_gap_infra, ai_task_by_comp_infra),
        ]:
            if sheet_name in ai_sheets:
                ai_df = ai_sheets[sheet_name]
                task_cols = [c for c in ai_df.columns if c != "Automatizable Task"]
                if not ai_task_categories:
                    ai_task_categories = task_cols
                for _, row in ai_df.iterrows():
                    comp = row["Automatizable Task"]
                    target_dict[comp] = row[task_cols].sum()
                    task_by_comp[comp] = {tc: row[tc] for tc in task_cols}
                    for tc in task_cols:
                        cat_dict[tc] = cat_dict.get(tc, 0) + row[tc]

    return df, ydf, alloc, infra_alloc, ai_reduction_app, ai_reduction_infra, ai_task_categories, ai_category_gap_app, ai_category_gap_infra, ai_task_by_comp_app, ai_task_by_comp_infra, asset_rels


@st.cache_data
def load_budget(file_path="Digital_Budget.xlsx"):
    bp = Path(file_path)
    if not bp.exists():
        return pd.DataFrame()
    raw = pd.read_excel(bp)
    raw = raw.rename(columns={
        "FY": "year", "Department": "department", "Layer": "layer", "Worktype": "worktype",
        "Initial_Cost": "budget_init", "Annual_Run_Cost": "budget_run",
        "Annual_Dev_Cost": "budget_dev", "Total_Cost": "budget_total",
    })
    return raw


@st.cache_data
def load_market_benchmarks(file_path="Performance_Benchmarks.xlsx"):
    bp = Path(file_path)
    if not bp.exists():
        return {}
    try:
        return pd.read_excel(bp, sheet_name=None)
    except Exception:
        return {}

@st.cache_data
def load_prediction_benchmarks(file_path="Prediction_Benchmarks.xlsx"):
    bp = Path(file_path)
    if not bp.exists():
        return {}
    try:
        return pd.read_excel(bp, sheet_name=None)
    except Exception:
        return {}

@st.cache_data
def load_prescription_benchmarks(file_path="Prescription_Benchmarks.xlsx"):
    bp = Path(file_path)
    if not bp.exists():
        return {}
    try:
        return pd.read_excel(bp, sheet_name=None)
    except Exception:
        return {}


APP_WORKTYPES = {"Collaboration", "Workflow", "Analysis", "Transaction", "External Facing"}
INFRA_WORKTYPES = {"Access", "Communications", "Integration", "Data"}

@st.cache_data
def load_benchmarks(file_path):
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception:
        return pd.DataFrame(), {}
    app_rows = []
    infra_bench = {}
    for wt, sdf in sheets.items():
        if sdf.empty:
            continue
        if wt in APP_WORKTYPES:
            if "Min Cost / MAU" not in sdf.columns:
                continue
            clean = sdf.dropna(subset=["Vendor"]).copy()
            clean["Min Cost / MAU"] = pd.to_numeric(clean["Min Cost / MAU"], errors="coerce")
            clean["Max Cost / MAU"] = pd.to_numeric(clean["Max Cost / MAU"], errors="coerce")
            clean = clean.dropna(subset=["Min Cost / MAU", "Max Cost / MAU"])
            clean["Worktype"] = wt
            clean["Midpoint"] = (clean["Min Cost / MAU"] + clean["Max Cost / MAU"]) / 2
            if "Reference URLs" in clean.columns:
                app_rows.append(clean[["Worktype", "Vendor", "Min Cost / MAU", "Max Cost / MAU", "Midpoint",
                                       "Key Infrastructure Drivers", "Reference URLs"]])
            else:
                clean["Reference URLs"] = ""
                app_rows.append(clean[["Worktype", "Vendor", "Min Cost / MAU", "Max Cost / MAU", "Midpoint",
                                       "Key Infrastructure Drivers", "Reference URLs"]])
        elif wt in INFRA_WORKTYPES:
            clean = sdf.copy()
            vendor_col = next((c for c in clean.columns if "Vendor" in c or "Primary" in c), None)
            if vendor_col:
                clean = clean.dropna(subset=[vendor_col])
            infra_bench[wt] = clean
    app_bench = pd.concat(app_rows, ignore_index=True) if app_rows else pd.DataFrame()
    return app_bench, infra_bench


@st.cache_data
def compute_agg(ydf, last_year=2025):
    hist = ydf[ydf["year"] <= last_year]
    by_year = hist.groupby("year")[["initCost","runCost","devCost"]].sum().reset_index()
    by_year["total"] = by_year[["initCost","runCost","devCost"]].sum(axis=1)
    dp = hist.groupby(["year","department"])[["initCost","runCost","devCost"]].sum().reset_index()
    dp["total"] = dp[["initCost","runCost","devCost"]].sum(axis=1)
    wp = hist.groupby(["year","worktype"])[["initCost","runCost","devCost"]].sum().reset_index()
    wp["total"] = wp[["initCost","runCost","devCost"]].sum(axis=1)

    # Map worktype → layer
    wt_layer = hist.groupby("worktype")["layer"].first().to_dict()

    def cagr(pdf, col, val, s=max(last_year-3, int(hist["year"].min())), e=last_year):
        d = pdf[pdf[col]==val].groupby("year")["total"].sum()
        return max(min((d[e]/d[s])**(1/(e-s))-1, 0.15), -0.10) if s in d.index and e in d.index and d[s]>0 else 0.03

    depts = sorted(hist["department"].unique())
    wts = sorted(hist["worktype"].unique())
    return by_year, dp, wp, {d: cagr(dp,"department",d) for d in depts}, {w: cagr(wp,"worktype",w) for w in wts}, depts, wts, by_year[by_year["year"]>=2023]["initCost"].mean(), wt_layer


def gen_proj(by_year, dp, wp, dg, wg, depts, wts, avg_init, noise, n, last_year=2025):
    rng = np.random.RandomState(42)
    base = by_year.iloc[-1]
    pt = pd.DataFrame([{"year":last_year+i,
        "initCost": avg_init*(1+rng.uniform(-noise,noise)),
        "runCost": base["runCost"]*(1+0.05+rng.uniform(-noise,noise))**i,
        "devCost": base["devCost"]*(1+0.05+rng.uniform(-noise,noise))**i} for i in range(1,n+1)])
    pt["total"] = pt[["initCost","runCost","devCost"]].sum(axis=1)
    db = dp[dp["year"]==last_year].set_index("department")["total"]
    dpj = pd.DataFrame([{"year":last_year+i, **{d:max(0,db.get(d,0)*(1+dg.get(d,.03)+rng.uniform(-noise,noise))**i) for d in depts}} for i in range(1,n+1)])
    wb = wp[wp["year"]==last_year].set_index("worktype")["total"]
    wpj = pd.DataFrame([{"year":last_year+i, **{w:max(0,wb.get(w,0)*(1+wg.get(w,.03)+rng.uniform(-noise,noise))**i) for w in wts}} for i in range(1,n+1)])
    return pt, dpj, wpj


SPEC_LABELS = {1: "Common", 2: "Shared", 3: "Specialized"}

def compute_savings(df, alloc, scenario):
    comp_details, consol_savings = [], 0
    if alloc is not None and len(alloc) > 0:
        comp_names = sorted(alloc["Component"].unique())
        for comp in comp_names:
            for spec in [1,2,3]:
                grp = alloc[(alloc["Component"]==comp)&(alloc["Spec"]==spec)]
                if len(grp)==0: continue
                n_apps, total_cost = len(grp), grp["Total_Alloc"].sum()
                avg_cost = total_cost/n_apps if n_apps>0 else 0
                if spec==1: n_keep = grp["Worktype"].nunique()
                elif spec==2: n_keep = max(1,n_apps//2) if scenario=="Conservative" else grp["Department"].nunique()
                else: n_keep = n_apps
                savings = max(0, total_cost - avg_cost*min(n_keep, n_apps))
                consol_savings += savings
                comp_details.append({"Component":comp,"Specificity":SPEC_LABELS[spec],"Apps":n_apps,"Keep":n_keep,
                    "Total Cost":total_cost,"Savings":savings,"Savings %":savings/total_cost*100 if total_cost>0 else 0})
    return pd.DataFrame(comp_details), consol_savings


def compute_savings_by_year(df, alloc, scenario):
    """Distribute consolidation savings across asset lifecycle years."""
    if alloc is None or len(alloc) == 0:
        return pd.DataFrame(columns=["year", "savings"])
    asset_info = df.set_index("Asset_ID")[["Initial_Year", "Estimated_Lifecycle_Years"]].to_dict("index")
    comp_names = sorted(alloc["Component"].unique())
    yearly_sav = {}
    for comp in comp_names:
        for spec in [1, 2, 3]:
            grp = alloc[(alloc["Component"] == comp) & (alloc["Spec"] == spec)]
            if len(grp) == 0:
                continue
            n_apps = len(grp)
            if spec == 1:
                n_keep = grp["Worktype"].nunique()
            elif spec == 2:
                n_keep = max(1, n_apps // 2) if scenario == "Conservative" else grp["Department"].nunique()
            else:
                n_keep = n_apps
            n_eliminate = max(0, n_apps - min(n_keep, n_apps))
            if n_eliminate == 0:
                continue
            # Eliminate the lowest-cost assets first
            ranked = grp.sort_values("Total_Alloc", ascending=True)
            eliminated = ranked.head(n_eliminate)
            for _, r in eliminated.iterrows():
                info = asset_info.get(r["Asset_ID"])
                if info is None:
                    continue
                start = int(info["Initial_Year"])
                life = int(info["Estimated_Lifecycle_Years"])
                end = start + life - 1
                for y in range(start, end + 1):
                    init_sav = r["Init_Alloc"] if y == start else 0
                    run_dev_sav = r["Run_Alloc"] + r["Dev_Alloc"]
                    yearly_sav[y] = yearly_sav.get(y, 0) + init_sav + run_dev_sav
    result = pd.DataFrame(sorted(yearly_sav.items()), columns=["year", "savings"])
    return result


def compute_ai_gap_by_year(df, alloc, infra_alloc, ai_reduction_app, ai_reduction_infra, scale=1.0):
    """Compute AI-driven work modernization gap distributed across asset lifecycle years."""
    asset_info = df.set_index("Asset_ID")[["Initial_Year", "Estimated_Lifecycle_Years"]].to_dict("index")
    yearly_app = {}
    yearly_infra = {}

    # Application layer
    if alloc is not None and len(alloc) > 0:
        for _, r in alloc.iterrows():
            factor = ai_reduction_app.get(r["Component"], 0) * scale
            if factor == 0:
                continue
            info = asset_info.get(r["Asset_ID"])
            if info is None:
                continue
            start = int(info["Initial_Year"])
            life = int(info["Estimated_Lifecycle_Years"])
            end = start + life - 1
            for y in range(start, end + 1):
                init_gap = r["Init_Alloc"] * factor if y == start else 0
                rd_gap = (r["Run_Alloc"] + r["Dev_Alloc"]) * factor
                yearly_app[y] = yearly_app.get(y, 0) + init_gap + rd_gap

    # Infrastructure layer
    if infra_alloc is not None and len(infra_alloc) > 0:
        for _, r in infra_alloc.iterrows():
            factor = ai_reduction_infra.get(r["Component"], 0) * scale
            if factor == 0:
                continue
            info = asset_info.get(r["Asset_ID"])
            if info is None:
                continue
            start = int(info["Initial_Year"])
            life = int(info["Estimated_Lifecycle_Years"])
            end = start + life - 1
            for y in range(start, end + 1):
                init_gap = r["Init_Alloc"] * factor if y == start else 0
                rd_gap = (r["Run_Alloc"] + r["Dev_Alloc"]) * factor
                yearly_infra[y] = yearly_infra.get(y, 0) + init_gap + rd_gap

    all_years = sorted(set(list(yearly_app.keys()) + list(yearly_infra.keys())))
    rows = [{"year": y, "ai_gap_app": yearly_app.get(y, 0), "ai_gap_infra": yearly_infra.get(y, 0)} for y in all_years]
    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["year", "ai_gap_app", "ai_gap_infra"])
    if len(result) > 0:
        result["ai_gap"] = result["ai_gap_app"] + result["ai_gap_infra"]
    else:
        result["ai_gap"] = 0
    return result


def compute_vendor_incentives(df, vi_scenario="Aggressive"):
    """Compute vendor margin headroom vs TSM average, price negotiation, and prevented price increase."""
    if "Gross_Margin_%" not in df.columns or "Deployment_Model" not in df.columns:
        return pd.DataFrame()

    ext = df[df["Deployment_Model"].isin(["Vendor", "Cloud"])].copy()
    ext = ext.dropna(subset=["Gross_Margin_%"])
    if ext.empty:
        return pd.DataFrame()

    ext["Total_Spend"] = ext["Initial_Cost"] + ext["Annual_Run_Cost"] + ext["Annual_Dev_Cost"]
    ext["Annual_Spend"] = ext["Annual_Run_Cost"] + ext["Annual_Dev_Cost"]

    rows = []
    for layer in ["Applications", "Infrastructure"]:
        ldf = ext[ext["Layer"] == layer]
        if ldf.empty:
            continue
        tsm_gross = ldf["Gross_Margin_%"].mean()
        tsm_net = tsm_gross * NET_MARGIN_RATIO

        for vendor, vdf in ldf.groupby("Vendor"):
            gm = vdf["Gross_Margin_%"].mean()
            gross_headroom = gm - tsm_gross  # allow negative for matrix
            est_net = gm * NET_MARGIN_RATIO
            net_headroom = est_net - tsm_net  # allow negative for matrix
            total_spend = vdf["Total_Spend"].sum()
            annual_spend = vdf["Annual_Spend"].sum()

            # Determine incentive tier from gross headroom
            if gross_headroom <= 0:
                tier, rebate_rate = "None", 0.0
            elif gross_headroom <= 5:
                tier, rebate_rate = "Low", INCENTIVE_SCENARIOS[vi_scenario]["Low"]
            elif gross_headroom <= 15:
                tier, rebate_rate = "Medium", INCENTIVE_SCENARIOS[vi_scenario]["Medium"]
            else:
                tier, rebate_rate = "High", INCENTIVE_SCENARIOS[vi_scenario]["High"]

            # Churn tier for prevented price increase — uses abs(headroom):
            # vendors above TSM face churn risk (we can switch away),
            # vendors below TSM are underperformers willing to share risk to retain contract.
            abs_headroom = abs(gross_headroom)
            if abs_headroom == 0:
                churn_tier, protection_rate = "None", 0.0
            elif abs_headroom <= 5:
                churn_tier, protection_rate = "Low", CHURN_SCENARIOS[vi_scenario]["Low"]
            elif abs_headroom <= 15:
                churn_tier, protection_rate = "Medium", CHURN_SCENARIOS[vi_scenario]["Medium"]
            else:
                churn_tier, protection_rate = "High", CHURN_SCENARIOS[vi_scenario]["High"]

            price_negotiation = rebate_rate * total_spend if gross_headroom > 0 else 0
            prevented_increase = protection_rate * annual_spend
            total_incentive = price_negotiation + prevented_increase

            # Quadrant assignment
            gm_label = "High GM" if gross_headroom > 0 else "Low GM"
            nm_label = "High NM" if net_headroom > 0 else "Low NM"
            quadrant = f"{gm_label} · {nm_label}"

            rows.append({
                "Vendor": vendor, "Layer": layer,
                "Gross_Margin": gm, "TSM_Gross": tsm_gross,
                "Gross_Headroom": gross_headroom,
                "Est_Net_Margin": est_net, "TSM_Net": tsm_net,
                "Net_Headroom": net_headroom,
                "Incentive_Tier": tier, "Rebate_Rate": rebate_rate,
                "Churn_Tier": churn_tier, "Protection_Rate": protection_rate,
                "Total_Spend": total_spend, "Annual_Spend": annual_spend,
                "Price_Negotiation": price_negotiation,
                "Prevented_Increase": prevented_increase,
                "Total_Incentive": total_incentive,
                "Quadrant": quadrant,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_incentive_savings_by_year(df, vi_scenario="Aggressive"):
    """Distribute vendor optimization savings (negotiation + prevented) across asset lifecycles."""
    if "Gross_Margin_%" not in df.columns or "Deployment_Model" not in df.columns:
        return pd.DataFrame(columns=["year", "negotiation", "prevented"])

    ext = df[df["Deployment_Model"].isin(["Vendor", "Cloud"])].copy()
    ext = ext.dropna(subset=["Gross_Margin_%"])
    if ext.empty:
        return pd.DataFrame(columns=["year", "negotiation", "prevented"])

    # Pre-compute TSM averages per layer
    tsm_by_layer = {}
    for layer in ["Applications", "Infrastructure"]:
        ldf = ext[ext["Layer"] == layer]
        if not ldf.empty:
            tsm_by_layer[layer] = ldf["Gross_Margin_%"].mean()

    yearly_neg = {}
    yearly_prev = {}
    for _, r in ext.iterrows():
        layer = r["Layer"]
        if layer not in tsm_by_layer:
            continue
        gm = r["Gross_Margin_%"]
        headroom = gm - tsm_by_layer[layer]
        abs_headroom = abs(headroom)

        # Determine rebate rate (only for positive headroom)
        if headroom <= 0:
            rebate_rate = 0.0
        elif headroom <= 5:
            rebate_rate = INCENTIVE_SCENARIOS[vi_scenario]["Low"]
        elif headroom <= 15:
            rebate_rate = INCENTIVE_SCENARIOS[vi_scenario]["Medium"]
        else:
            rebate_rate = INCENTIVE_SCENARIOS[vi_scenario]["High"]

        # Protection rate uses abs(headroom) — both sides of TSM
        if abs_headroom == 0:
            protection_rate = 0.0
        elif abs_headroom <= 5:
            protection_rate = CHURN_SCENARIOS[vi_scenario]["Low"]
        elif abs_headroom <= 15:
            protection_rate = CHURN_SCENARIOS[vi_scenario]["Medium"]
        else:
            protection_rate = CHURN_SCENARIOS[vi_scenario]["High"]

        if rebate_rate == 0 and protection_rate == 0:
            continue

        start = int(r["Initial_Year"])
        life = int(r["Estimated_Lifecycle_Years"])
        end = start + life - 1
        run_dev = float(r["Annual_Run_Cost"]) + float(r["Annual_Dev_Cost"])

        for y in range(start, end + 1):
            # Negotiation: rebate on all costs per active year (positive headroom only)
            if rebate_rate > 0:
                init_neg = rebate_rate * float(r["Initial_Cost"]) if y == start else 0
                rundev_neg = rebate_rate * run_dev
                yearly_neg[y] = yearly_neg.get(y, 0) + init_neg + rundev_neg
            # Prevented: protection rate on run+dev only (both sides of TSM)
            yearly_prev[y] = yearly_prev.get(y, 0) + protection_rate * run_dev

    all_years = sorted(set(yearly_neg.keys()) | set(yearly_prev.keys()))
    result = pd.DataFrame({
        "year": all_years,
        "negotiation": [yearly_neg.get(y, 0) for y in all_years],
        "prevented": [yearly_prev.get(y, 0) for y in all_years],
    })
    return result


def compute_market_repricing_by_year(df, app_bench, infra_bench):
    """Compute market repricing opportunity distributed across asset lifecycle years.

    For each asset with a matching benchmark worktype, the repricing opportunity
    is: opp_pct * (Annual_Run_Cost + Annual_Dev_Cost) for each active year,
    where opp_pct = 1 - (mkt_mean / org_kpi) = premium / (1 + premium).
    """
    rng = np.random.default_rng(42)
    org_premium = rng.uniform(0.20, 0.30)

    yearly_opp = {}

    # ── Applications ──
    if not app_bench.empty:
        bench_agg = app_bench.groupby("Worktype").agg(
            mkt_mean_mid=("Midpoint", "mean")).reset_index()
        bench_agg["org_kpi"] = bench_agg["mkt_mean_mid"] * (1 + org_premium)
        bench_agg["opp_pct"] = 1 - (bench_agg["mkt_mean_mid"] / bench_agg["org_kpi"])

        app_df = df[df["Layer"] == "Applications"].copy()
        for _, r in app_df.iterrows():
            wt = r.get("Worktype")
            mr = bench_agg[bench_agg["Worktype"] == wt]
            if len(mr) == 0:
                continue
            opp_pct = mr.iloc[0]["opp_pct"]
            if opp_pct <= 0:
                continue
            start = int(r["Initial_Year"])
            life = int(r["Estimated_Lifecycle_Years"])
            end = start + life - 1
            run_dev = float(r["Annual_Run_Cost"]) + float(r["Annual_Dev_Cost"])
            for y in range(start, end + 1):
                yearly_opp[y] = yearly_opp.get(y, 0) + opp_pct * run_dev

    # ── Infrastructure (Access only) ──
    if infra_bench and "Access" in infra_bench:
        acc = infra_bench["Access"].copy()
        acc_cols = list(acc.columns)
        if len(acc_cols) >= 3:
            acc["_min"] = pd.to_numeric(acc[acc_cols[1]], errors="coerce")
            acc["_max"] = pd.to_numeric(acc[acc_cols[2]], errors="coerce")
            acc_num = acc.dropna(subset=["_min", "_max"]).copy()
            if len(acc_num) > 0:
                acc_num["_mid"] = (acc_num["_min"] + acc_num["_max"]) / 2
                i_mean = acc_num["_mid"].mean()
                i_org = i_mean * (1 + org_premium)
                opp_pct = 1 - (i_mean / i_org)
                if opp_pct > 0:
                    infra_df = df[(df["Layer"] == "Infrastructure") & (df["Worktype"] == "Access")]
                    for _, r in infra_df.iterrows():
                        start = int(r["Initial_Year"])
                        life = int(r["Estimated_Lifecycle_Years"])
                        end = start + life - 1
                        run_dev = float(r["Annual_Run_Cost"]) + float(r["Annual_Dev_Cost"])
                        for y in range(start, end + 1):
                            yearly_opp[y] = yearly_opp.get(y, 0) + opp_pct * run_dev

    if not yearly_opp:
        return pd.DataFrame(columns=["year", "mkt_opp"])
    all_years = sorted(yearly_opp.keys())
    return pd.DataFrame({
        "year": all_years,
        "mkt_opp": [yearly_opp[y] for y in all_years],
    })


def compute_early_pay(df, preset, fine_tune, days_early=20):
    if "Estimated_Size" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), 0, 0, 0

    p = EARLY_PAY_PRESETS[preset]
    multiplier = 1 + fine_tune / 100
    ext = df[df["Deployment_Model"].isin(["Vendor","Cloud"])].copy()
    ext["TotalCost"] = ext["Initial_Cost"] + ext["Annual_Run_Cost"] + ext["Annual_Dev_Cost"]

    size_disc = {s: p["disc"].get(s, 0) * multiplier / 100 for s in ["Large","Mid-Size","Small"]}
    size_acc = {s: min(p["accept"].get(s, 0) * multiplier, 100) / 100 for s in ["Large","Mid-Size","Small"]}
    coc_rate = max(0.5, p["coc"] / multiplier) / 100

    ext["Discount_Rate"] = ext["Estimated_Size"].map(size_disc).fillna(0)
    if "Gross_Margin_%" in ext.columns:
        ext["Margin_Mult"] = ext["Gross_Margin_%"].apply(margin_multiplier)
    else:
        ext["Margin_Mult"] = 1.0
    ext["Discount_Rate"] = ext["Discount_Rate"] * ext["Margin_Mult"]
    ext["Acceptance"] = ext["Estimated_Size"].map(size_acc).fillna(0)
    ext["Gross_Discount"] = ext["TotalCost"] * ext["Discount_Rate"] * ext["Acceptance"]
    ext["Cost_of_Cash"] = ext["TotalCost"] * (days_early / 365) * coc_rate * ext["Acceptance"]
    ext["Net_Benefit"] = ext["Gross_Discount"] - ext["Cost_of_Cash"]

    vendor_summary = ext.groupby(["Vendor","Estimated_Size"]).agg(
        Assets=("Asset_ID","count"), Total_Cost=("TotalCost","sum"),
        Gross_Discount=("Gross_Discount","sum"), Cost_of_Cash=("Cost_of_Cash","sum"),
        Net_Benefit=("Net_Benefit","sum")).reset_index()

    # Yearly waterfall
    yearly_rows = []
    for _, r in ext.iterrows():
        start, end = int(r["Initial_Year"]), int(r["Initial_Year"])+int(r["Estimated_Lifecycle_Years"])-1
        dr, ac, cocr = r["Discount_Rate"], r["Acceptance"], coc_rate
        for y in range(start, end+1):
            run_g = r["Annual_Run_Cost"]*dr*ac
            dev_g = r["Annual_Dev_Cost"]*dr*ac
            init_g = r["Initial_Cost"]*dr*ac if y==start else 0
            coc_y = (r["Annual_Run_Cost"]+r["Annual_Dev_Cost"]+(r["Initial_Cost"] if y==start else 0))*(days_early/365)*cocr*ac
            yearly_rows.append({"year":y, "Gross":run_g+dev_g+init_g, "CoC":coc_y, "Net":run_g+dev_g+init_g-coc_y})
    yearly_df = pd.DataFrame(yearly_rows)

    # Effective rates for display
    eff_rates = {s: {"disc": size_disc.get(s,0)*100, "acc": size_acc.get(s,0)*100, "coc": coc_rate*100} for s in ["Large","Mid-Size","Small"]}

    return vendor_summary, yearly_df, vendor_summary["Gross_Discount"].sum(), vendor_summary["Cost_of_Cash"].sum(), vendor_summary["Net_Benefit"].sum(), eff_rates


# ══════════════════════════
# LOAD
# ══════════════════════════
uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
file_path = uploaded or next((p for p in [
    Path("/mnt/user-data/uploads/Digital_Asset_Inventory.xlsx"),
    Path("Digital_Asset_Inventory.xlsx")] if p.exists()), None)
if file_path is None:
    st.warning("Please upload a Digital Asset Inventory Excel file.")
    st.stop()

df, ydf, alloc, infra_alloc, ai_reduction_app, ai_reduction_infra, ai_task_categories, ai_cat_gap_app, ai_cat_gap_infra, ai_task_by_comp_app, ai_task_by_comp_infra, asset_rels = load_data(file_path)
LAST_HIST_YEAR = int(df["Initial_Year"].max())  # auto-detect from latest asset deployment year
by_year, dp, wp, dg, wg, depts, wts, avg_init, wt_layer = compute_agg(ydf, LAST_HIST_YEAR)
bench_path = Path("Benchmark.xlsx")
app_bench, infra_bench = load_benchmarks(bench_path) if bench_path.exists() else (pd.DataFrame(), {})

# ══════════════════════════
# SIDEBAR
# ══════════════════════════
# ── Projections ──
if "proj_toggle" not in st.session_state:
    st.session_state["proj_toggle"] = True
if "proj_noise" not in st.session_state:
    st.session_state["proj_noise"] = 0.05
if "proj_years" not in st.session_state:
    st.session_state["proj_years"] = 5
_proj_label = "Off"
if st.session_state["proj_toggle"]:
    _proj_label = f'{st.session_state["proj_years"]}yr'
with st.sidebar.expander(f"⚙️ Projections\n{_proj_label}", expanded=False):
    st.caption(f"Projections start on: **{LAST_HIST_YEAR + 1}** (last data year: {LAST_HIST_YEAR})")
    st.toggle("Enable Projections", value=st.session_state["proj_toggle"], key="proj_toggle")
    if st.session_state["proj_toggle"]:
        st.slider("Noise Level", 0.0, 0.20, st.session_state["proj_noise"], 0.01, key="proj_noise")
        st.slider("Projection Years", 1, 10, st.session_state["proj_years"], key="proj_years")
show_proj = st.session_state["proj_toggle"]
noise = st.session_state["proj_noise"]
proj_years = st.session_state["proj_years"]

# ── Asset Management ──
if "consol_scenario" not in st.session_state:
    st.session_state["consol_scenario"] = "Conservative"
if "consol_pct_global" not in st.session_state:
    st.session_state["consol_pct_global"] = 50
_consol_label = st.session_state["consol_scenario"]
with st.sidebar.expander(f"🧩 Asset Management\n{_consol_label}", expanded=False):
    st.radio("Scenario", ["Conservative","Aggressive"],
        captions=["Moderate consolidation — keeps half of shared components",
                  "Maximum consolidation — one shared component per department"],
        key="consol_scenario")
    st.slider("Savings capture %", 0, 100, 50, 5, key="consol_pct_global",
        help="Scale consolidation opportunity from 0% (none) to 100% (full)")
scenario = st.session_state["consol_scenario"]
consol_scale = st.session_state["consol_pct_global"] / 100

# ── AI Work Modernization ──
if "ai_scenario" not in st.session_state:
    st.session_state["ai_scenario"] = "Moderate"
AI_SCENARIO_SCALE = {"Conservative": 0.50, "Moderate": 0.75, "Aggressive": 1.0}
_ai_label = st.session_state["ai_scenario"]
with st.sidebar.expander(f"🤖 Work Modernization\n{_ai_label}", expanded=False):
    st.radio("Adoption Scenario", ["Conservative", "Moderate", "Aggressive"],
        index=1,
        captions=["50% of AI reduction potential", "75% of AI reduction potential", "100% of AI reduction potential"],
        key="ai_scenario")
ai_scenario = st.session_state["ai_scenario"]
ai_scale = AI_SCENARIO_SCALE[ai_scenario]

# ── Vendor Incentives ──
if "vi_scenario" not in st.session_state:
    st.session_state["vi_scenario"] = "Aggressive"
_vi_label = st.session_state["vi_scenario"]
with st.sidebar.expander(f"🏷️ Vendor Incentives\n{_vi_label}", expanded=False):
    st.radio("Scenario", ["Conservative", "Aggressive"],
        index=1,
        captions=["Lower rates: rebate 1/3/5% · protection 0.5/1.5/3%",
                  "Full rates: rebate 2/5/10% · protection 1/3/5%"],
        key="vi_scenario")
vi_scenario = st.session_state["vi_scenario"]

# ── Payment Management ──
PM_SCENARIOS = {"Conservative": 0.015, "Moderate": 0.03, "Aggressive": 0.05}
if "pm_scenario" not in st.session_state:
    st.session_state["pm_scenario"] = "Conservative"
_pm_label = st.session_state["pm_scenario"]
with st.sidebar.expander(f"💳 Payment Management\n{_pm_label}", expanded=False):
    st.radio("Scenario", list(PM_SCENARIOS.keys()),
        index=0,
        captions=["1.5% of spend — baseline payment process improvements",
                  "3.0% of spend — + payment timing optimization, dynamic discounting",
                  "5.0% of spend — + full payment automation, virtual cards, rebate capture"],
        key="pm_scenario")
pm_scenario = st.session_state["pm_scenario"]
pm_rate = PM_SCENARIOS[pm_scenario]

# ── Early Pay ──
if "ep_preset" not in st.session_state:
    st.session_state["ep_preset"] = "Aggressive"
if "ep_fine_tune" not in st.session_state:
    st.session_state["ep_fine_tune"] = 0
_ep_label = st.session_state["ep_preset"]
with st.sidebar.expander(f"💰 Early Pay\n{_ep_label}", expanded=False):
    st.radio("Strategy", list(EARLY_PAY_PRESETS.keys()),
        captions=[v["desc"] for v in EARLY_PAY_PRESETS.values()],
        key="ep_preset")
    st.slider("Fine-tune adjustment", -50, 50, 0, 5,
        help="Shift all early pay rates up or down from preset",
        key="ep_fine_tune")
ep_preset = st.session_state["ep_preset"]
ep_fine_tune = st.session_state["ep_fine_tune"]

# ══════════════════════════
# COMPUTE
# ══════════════════════════
comp_det, consol_sav_raw = compute_savings(df, alloc, scenario)
savings_by_year = compute_savings_by_year(df, alloc, scenario)
# Apply global consolidation scale
consol_sav = consol_sav_raw * consol_scale
savings_by_year["savings"] = savings_by_year["savings"] * consol_scale if len(savings_by_year) > 0 else savings_by_year["savings"]
total_sav = consol_sav
annual_costs = df["Annual_Run_Cost"].sum() + df["Annual_Dev_Cost"].sum()
portfolio_total = df["Initial_Cost"].sum() + annual_costs
vendor_summary, ep_yearly, ep_gross, ep_coc, ep_net, ep_rates = compute_early_pay(df, ep_preset, ep_fine_tune)
inc_by_year_raw = compute_incentive_savings_by_year(df, vi_scenario)
mkt_repricing_by_year = compute_market_repricing_by_year(df, app_bench, infra_bench)
if not mkt_repricing_by_year.empty and "mkt_opp" in mkt_repricing_by_year.columns:
    mkt_repricing_by_year["mkt_opp"] = mkt_repricing_by_year["mkt_opp"] * 0.3
ai_gap_by_year = compute_ai_gap_by_year(df, alloc, infra_alloc, ai_reduction_app, ai_reduction_infra, ai_scale)

# Budget variance
budget_df = load_budget()
if not budget_df.empty:
    actual_agg = ydf.groupby(["year", "department", "layer", "worktype"])[["initCost", "runCost", "devCost"]].sum().reset_index()
    actual_agg["actual_total"] = actual_agg[["initCost", "runCost", "devCost"]].sum(axis=1)
    var_df = actual_agg.merge(budget_df, on=["year", "department", "layer", "worktype"], how="inner")
    var_df["variance_pct"] = np.where(var_df["budget_total"] != 0,
        (var_df["actual_total"] - var_df["budget_total"]) / var_df["budget_total"] * 100, 0)
    var_df["score"] = (10 - var_df["variance_pct"].abs()).clip(lower=0)
else:
    var_df = pd.DataFrame()

# ══════════════════════════════════════
# Generic Benchmark Parsing (all P tabs)
# ══════════════════════════════════════

def _perf_score(val, best_extreme, worst_extreme, lower_better):
    """Linear 0–10 across full range from worst_extreme (0) to best_extreme (10)."""
    if best_extreme == worst_extreme:
        return 5.0
    if lower_better:
        raw = (worst_extreme - val) / (worst_extreme - best_extreme)
    else:
        raw = (val - worst_extreme) / (best_extreme - worst_extreme)
    return round(max(0.0, min(10.0, raw * 10.0)), 1)

def _parse_pct(v):
    """Parse percentage string like '30%', '0.35+', or numeric to float."""
    if pd.isna(v): return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().rstrip("+").rstrip("%")
    try:
        val = float(s)
        if "%" in str(v): val = val / 100.0
        return val
    except ValueError:
        return None

def _auto_fmt(kpi_name, sdf):
    """Auto-detect format function from KPI name and data. No hardcoding needed."""
    name_lower = kpi_name.lower()
    kdf = sdf[sdf["KPI"] == kpi_name]
    # Check if Min/Max values in the sheet contain '%' strings (actual data, not KPI name)
    has_pct_str = any(
        isinstance(v, str) and "%" in v
        for v in pd.concat([kdf["Min"], kdf["Max"]]).dropna()
    )
    if has_pct_str:
        return lambda v: f"{v*100:.1f}%"
    if "years" in name_lower or "period" in name_lower:
        return lambda v: f"{v:.1f} yrs"
    if "density" in name_lower:
        return lambda v: f"{v:,.0f}"
    # Check actual data range to decide format
    vals = [_parse_pct(v) for v in pd.concat([kdf["Min"], kdf["Max"]]).dropna()]
    vals = [v for v in vals if v is not None]
    if vals:
        max_abs = max(abs(v) for v in vals)
        if max_abs > 1000:
            return lambda v: fmt(v)  # dollar/large number
        if max_abs <= 1:
            if "index" in name_lower or "hhi" in name_lower:
                return lambda v: f"{v:.3f}"
            # KPI name ends with (%) — stored as decimals
            if "(%)" in kpi_name:
                return lambda v: f"{v*100:.1f}%"
            return lambda v: f"{v:.2f}"
        # Values 1-1000: could be ratio, count, or percent stored as whole numbers
        if "(%)" in kpi_name or "fragmentation" in name_lower:
            return lambda v: f"{v:.1f}%"  # already whole-number percentages
        if "ratio" in name_lower:
            return lambda v: f"{v:.2f}"
    return lambda v: f"{v:.2f}"

def _detect_tier_map(sdf):
    """Auto-detect tier name mapping from Performance Level column values."""
    pl_col = "Performance Level" if "Performance Level" in sdf.columns else "Value"
    levels = [str(v).strip() for v in sdf[pl_col].dropna().unique()]
    tier_map = {}
    for lv in levels:
        ll = lv.lower()
        if "alpha" in ll or "best performer" in ll:
            tier_map["alpha"] = lv
        elif "top" in ll:
            tier_map["top"] = lv
        elif "average" in ll or "benchmark" in ll:
            tier_map["avg"] = lv
        elif "bottom" in ll or "late" in ll:
            tier_map["bot"] = lv
    return tier_map, pl_col

def _get_str_col(row, col):
    """Safely get a string column value from a row."""
    if col not in row.index:
        return ""
    v = row[col]
    if pd.isna(v):
        return ""
    return str(v).strip()

def parse_benchmark_sheets(bench_sheets):
    """Generic parser for any benchmark Excel file (Performance, Prediction, Prescription).
    Auto-detects sheets, tiers, formats, and reads all available columns.
    Returns (categories, tier_scores, overall, alpha_label)."""
    if not bench_sheets:
        return {}, {}, 0, "Best Performers"

    import random as _rnd
    _rnd.seed(42)

    categories = {}    # {sheet: [row_dicts]}
    tier_scores = {}   # {sheet: {tier_key: avg_score}}
    alpha_reps = {}    # {sheet: set of names}

    # Detect representative column name (varies by file)
    rep_col_candidates = ["Alpha Performer Representative", "Sample"]

    for sheet_name, sdf in bench_sheets.items():
        if sdf is None or sdf.empty or "KPI" not in sdf.columns:
            continue
        t_map, pl_col = _detect_tier_map(sdf)
        if not all(k in t_map for k in ["alpha", "top", "avg", "bot"]):
            continue  # skip sheets without full tier structure

        # Detect representative column
        rep_col = next((c for c in rep_col_candidates if c in sdf.columns), None)

        sheet_rows = []
        t_scores = {"alpha": [], "top": [], "avg": [], "bot": []}
        alpha_names = set()

        for kpi_name in sdf["KPI"].unique():
            kdf = sdf[sdf["KPI"] == kpi_name]
            fmt_fn = _auto_fmt(kpi_name, sdf)

            def _mid(tier_key, _kdf=kdf, _t_map=t_map, _pl=pl_col):
                r = _kdf[_kdf[_pl] == _t_map[tier_key]]
                if r.empty: return None
                mn, mx = _parse_pct(r.iloc[0]["Min"]), _parse_pct(r.iloc[0]["Max"])
                if mn is not None and mx is not None: return (mn + mx) / 2
                return mn if mn is not None else mx

            def _extremes(tier_key, _kdf=kdf, _t_map=t_map, _pl=pl_col):
                r = _kdf[_kdf[_pl] == _t_map[tier_key]]
                if r.empty: return None, None
                return _parse_pct(r.iloc[0]["Min"]), _parse_pct(r.iloc[0]["Max"])

            best_v, top_v, avg_v, bot_v = _mid("alpha"), _mid("top"), _mid("avg"), _mid("bot")
            if best_v is None: best_v = top_v
            if any(v is None for v in [best_v, avg_v, bot_v]):
                continue
            lower_better = top_v < bot_v if (top_v is not None and bot_v is not None) else best_v < bot_v

            alpha_mn, alpha_mx = _extremes("alpha")
            bot_mn, bot_mx = _extremes("bot")
            if lower_better:
                best_extreme = alpha_mn if alpha_mn is not None else best_v
                worst_extreme = bot_mx if bot_mx is not None else (bot_mn if bot_mn is not None else bot_v)
            else:
                best_extreme = alpha_mx if alpha_mx is not None else (alpha_mn if alpha_mn is not None else best_v)
                worst_extreme = bot_mn if bot_mn is not None else bot_v

            for tk, tv in [("alpha", best_v), ("top", top_v), ("avg", avg_v), ("bot", bot_v)]:
                if tv is not None:
                    t_scores[tk].append(_perf_score(tv, best_extreme, worst_extreme, lower_better))

            # Placeholder org value
            _t = _rnd.uniform(0.2, 0.8)
            org_val = bot_v + _t * (top_v - bot_v) if not lower_better else bot_v - _t * (bot_v - top_v)
            score = _perf_score(org_val, best_extreme, worst_extreme, lower_better)

            # Alpha representative
            alpha_row = kdf[kdf[pl_col] == t_map["alpha"]]
            alpha_sample = ""
            if not alpha_row.empty and rep_col and rep_col in alpha_row.columns:
                _av = alpha_row.iloc[0][rep_col]
                if pd.notna(_av):
                    alpha_sample = str(_av).strip()
                    alpha_names.add(alpha_sample)
            alpha_val = _parse_pct(alpha_row.iloc[0]["Min"]) if not alpha_row.empty else best_v
            if alpha_val is None: alpha_val = best_v
            alpha_fmt = fmt_fn(alpha_val)
            alpha_str = f"{alpha_sample} ({alpha_fmt})" if alpha_sample else alpha_fmt
            alpha_score_val = round(_perf_score(best_v, best_extreme, worst_extreme, lower_better), 1)

            # Alpha scenario desc (backward compat)
            scenario_desc = _get_str_col(alpha_row.iloc[0], "Scenario Description") if not alpha_row.empty else ""

            # KPI description (from first non-empty row)
            kpi_desc = ""
            if "KPI Description" in kdf.columns:
                descs = kdf["KPI Description"].dropna()
                if len(descs) > 0:
                    kpi_desc = str(descs.iloc[0]).strip()

            # Per-tier rich data: scenario_desc, practice_differentiator, score, mid_val
            tier_data = {}
            for tk in ["alpha", "top", "avg", "bot"]:
                t_row = kdf[kdf[pl_col] == t_map[tk]]
                td = {"scenario_desc": "", "practice_diff": "", "score": None, "mid_val": _mid(tk)}
                if not t_row.empty:
                    r0 = t_row.iloc[0]
                    td["scenario_desc"] = _get_str_col(r0, "Scenario Description")
                    td["practice_diff"] = _get_str_col(r0, "Practice Differentiator")
                    mv = _mid(tk)
                    td["score"] = round(_perf_score(mv, best_extreme, worst_extreme, lower_better), 1) if mv is not None else None
                tier_data[tk] = td

            sheet_rows.append({
                "KPI": kpi_name, "kpi_desc": kpi_desc, "org_val": org_val,
                "org_fmt": fmt_fn(org_val),
                "score": round(score, 1), "alpha": alpha_str,
                "alpha_score": alpha_score_val, "scenario_desc": scenario_desc,
                "alpha_rep": alpha_sample, "tier_data": tier_data,
            })

        if sheet_rows:
            categories[sheet_name] = sheet_rows
        tier_scores[sheet_name] = {k: round(np.mean(v), 1) if v else None for k, v in t_scores.items()}
        tier_scores[sheet_name]["org"] = round(np.mean([r["score"] for r in sheet_rows]), 1) if sheet_rows else None
        alpha_reps[sheet_name] = alpha_names

    all_scores = [r["score"] for rows in categories.values() for r in rows]
    overall = np.mean(all_scores) if all_scores else 0

    all_alpha = set()
    for names in alpha_reps.values():
        all_alpha.update(names)
    alpha_label = ", ".join(sorted(all_alpha)[:3]) if all_alpha else "Best Performers"

    return categories, tier_scores, overall, alpha_label

# ── Parse all three benchmark files ──
mkt_bench_sheets = load_market_benchmarks()
perf_categories, perf_tier_scores, perf_overall, _alpha_label = parse_benchmark_sheets(mkt_bench_sheets)

pred_bench_sheets = load_prediction_benchmarks()
pred_categories, pred_tier_scores, pred_overall, _pred_alpha_label = parse_benchmark_sheets(pred_bench_sheets)

presc_bench_sheets = load_prescription_benchmarks()
presc_categories, presc_tier_scores, presc_overall, _presc_alpha_label = parse_benchmark_sheets(presc_bench_sheets)

# ══════════════════════════
# HEADER
# ══════════════════════════
st.markdown("# Quant16 Portfolio Optimization Model - Digital Assets")
st.markdown(f"<span style='color:{MUTED}; font-size:13px;'>{len(df)} assets · "
            f"{int(by_year['year'].min())}–{LAST_HIST_YEAR}"
            f"{'  ·  Projected → '+str(LAST_HIST_YEAR+proj_years) if show_proj else ''}</span>",
            unsafe_allow_html=True)

# ══════════════════════════
# PRE-COMPUTE spend_comp (used by multiple tabs)
# ══════════════════════════
hist_spend = by_year[["year", "total"]].copy()
if show_proj:
    _pt, _, _ = gen_proj(by_year, dp, wp, dg, wg, depts, wts, avg_init, noise, proj_years, LAST_HIST_YEAR)
    all_spend = pd.concat([hist_spend, _pt[["year", "total"]]], ignore_index=True)
else:
    all_spend = hist_spend.copy()

ep_by_year = pd.DataFrame()
if len(ep_yearly) > 0:
    ep_by_year = ep_yearly.groupby("year")["Net"].sum().reset_index()
    ep_by_year.columns = ["year", "ep_savings"]

spend_comp = all_spend.copy()
spend_comp = spend_comp.merge(savings_by_year, on="year", how="left")
spend_comp = spend_comp.merge(ep_by_year, on="year", how="left")
spend_comp["savings"] = spend_comp["savings"].fillna(0)
spend_comp["ep_savings"] = spend_comp["ep_savings"].fillna(0)

if not inc_by_year_raw.empty:
    vi_by_year = inc_by_year_raw.copy()
    vi_by_year["vi_internal"] = vi_by_year["negotiation"] + vi_by_year["prevented"]
else:
    vi_by_year = pd.DataFrame(columns=["year", "vi_internal"])
vi_by_year = vi_by_year.merge(mkt_repricing_by_year, on="year", how="outer").fillna(0)
if "vi_internal" not in vi_by_year.columns:
    vi_by_year["vi_internal"] = 0.0
if "mkt_opp" not in vi_by_year.columns:
    vi_by_year["mkt_opp"] = 0.0
vi_by_year["vi_savings"] = vi_by_year["vi_internal"] + vi_by_year["mkt_opp"]
spend_comp = spend_comp.merge(vi_by_year[["year", "vi_savings"]], on="year", how="left")
spend_comp["vi_savings"] = spend_comp["vi_savings"].fillna(0)

spend_comp = spend_comp.merge(ai_gap_by_year[["year", "ai_gap"]], on="year", how="left")
spend_comp["ai_savings"] = spend_comp["ai_gap"].fillna(0)
spend_comp.drop(columns=["ai_gap"], inplace=True, errors="ignore")

# Payment Management — fixed % of current trend spend
spend_comp["pm_savings"] = spend_comp["total"] * pm_rate

last_hist = spend_comp[spend_comp["year"] <= LAST_HIST_YEAR]
if len(last_hist) > 0:
    last_row = last_hist.iloc[-1]
    last_total = last_row["total"]
    consol_ratio = last_row["savings"] / last_total if last_total > 0 else 0
    ep_ratio = last_row["ep_savings"] / last_total if last_total > 0 else 0
    vi_ratio = last_row["vi_savings"] / last_total if last_total > 0 else 0
    ai_ratio = last_row["ai_savings"] / last_total if last_total > 0 else 0
    proj_mask = spend_comp["year"] > LAST_HIST_YEAR
    spend_comp.loc[proj_mask, "savings"] = spend_comp.loc[proj_mask, "total"] * consol_ratio
    spend_comp.loc[proj_mask, "ep_savings"] = spend_comp.loc[proj_mask, "total"] * ep_ratio
    spend_comp.loc[proj_mask, "vi_savings"] = spend_comp.loc[proj_mask, "total"] * vi_ratio
    spend_comp.loc[proj_mask, "ai_savings"] = spend_comp.loc[proj_mask, "total"] * ai_ratio

spend_comp["total_savings"] = spend_comp["savings"] + spend_comp["ep_savings"] + spend_comp["vi_savings"] + spend_comp["ai_savings"] + spend_comp["pm_savings"]
spend_comp["current_trend"] = spend_comp["total"]
spend_comp["optimal"] = (spend_comp["total"] - spend_comp["total_savings"]).clip(lower=0)

def optimized_ramp(row):
    yr = row["year"]
    if yr <= LAST_HIST_YEAR:
        return 0
    offset = yr - LAST_HIST_YEAR
    if offset == 1:
        return row["total_savings"] * 0.33
    elif offset == 2:
        return row["total_savings"] * 0.66
    else:
        return row["total_savings"]

spend_comp["optimized"] = (spend_comp["total"] - spend_comp.apply(optimized_ramp, axis=1)).clip(lower=0)

# ══════════════════════════
# TABS
# ══════════════════════════
tab_perf, tab_pred, tab_presc, tab_totalsav, tab_effcomp, tab_others = st.tabs([
    "🏆 Performance", "🔮 Prediction", "💊 Prescription",
    "🎯 Efficiency Opportunities", "🔧 Efficiency Components (Temp)", "📁 Others (Temp)"])

with tab_presc:
    if not presc_categories:
        st.warning("No prescription benchmark data available. Place Prescription_Benchmarks.xlsx in the app directory.")
    else:
        st.markdown(
            "<h3 style='margin-bottom:0'>Prescription</h3>"
            "<p style='color:#94a3b8;font-size:0.95rem;margin-top:0.25rem'>"
            "<b>Prescription</b> evaluates the organization's capacity to translate analysis into actionable "
            "optimization — financial restructuring, process modernization, and risk mitigation. Scores reflect "
            "readiness to prescribe and execute strategic action, not the recommendations themselves.<br><br>"
            "A score of 0 indicates no prescriptive optimization in place; "
            "10 represents full strategic execution maturity across all measured dimensions.</p>",
            unsafe_allow_html=True,
        )

        # Build data structures
        _presc_cats = list(presc_tier_scores.keys())
        _presc_tier_rows = [
            ("Maximum Performance",                       "max"),
            (f"Alpha Performer ({_presc_alpha_label})",   "alpha"),
            ("Top Performance",                            "top"),
            ("Benchmark Performance",                      "avg"),
            ("Late Adopters",                              "bot"),
            ("Observed",                                   "org"),
        ]

        def _dot_color_presc(score):
            if score is None: return "#94a3b8"
            if score >= 7: return "#059669"
            if score >= 4: return "#ca8a04"
            return "#dc2626"

        _presc_tier_colors = {
            "Maximum Performance": "#059669",
            "Alpha Performer": "#0284c7",
            "Top Performance": "#6d28d9",
            "Benchmark Performance": "#ca8a04",
            "Late Adopters": "#dc2626",
            "Observed": "#c026d3",
        }

        # ── 0) Summary metric cards ──
        _presc_obs_avg = round(np.mean([presc_tier_scores[c].get("org", 0) or 0 for c in _presc_cats]), 1)
        _presc_alpha_avg = round(np.mean([presc_tier_scores[c].get("alpha", 0) or 0 for c in _presc_cats]), 1)
        _presc_gap = round(_presc_alpha_avg - _presc_obs_avg, 1)
        _presc_cat_gaps = {c: round((presc_tier_scores[c].get("alpha", 0) or 0) - (presc_tier_scores[c].get("org", 0) or 0), 1) for c in _presc_cats}
        _presc_riskiest = max(_presc_cat_gaps, key=_presc_cat_gaps.get) if _presc_cat_gaps else ""
        _presc_riskiest_gap = _presc_cat_gaps.get(_presc_riskiest, 0)
        _presc_five_yr = spend_comp[(spend_comp["year"] >= LAST_HIST_YEAR+1) & (spend_comp["year"] <= LAST_HIST_YEAR+5)]
        _presc_five_yr_sav = (_presc_five_yr["current_trend"].sum() - _presc_five_yr["optimal"].sum()) if len(_presc_five_yr) > 0 else 0

        _presc_cards_ctr = st.container(border=True)
        rmc1, rmc2, rmc3, rmc4 = _presc_cards_ctr.columns(4)
        rmc1.metric("Organization Score", f"{_presc_obs_avg} / 10", "Avg across all dimensions")
        rmc2.metric("Gap to Alpha", f"{_presc_gap} pts", f"Alpha avg: {_presc_alpha_avg}")
        rmc3.metric("Highest Risk Area", _presc_riskiest, f"{_presc_riskiest_gap} pts gap to alpha")
        rmc4.metric("5-Year Savings Potential", fmt(_presc_five_yr_sav), f"Baseline − Optimal ({LAST_HIST_YEAR+1}–{LAST_HIST_YEAR+5})")

        # ── 1) Line chart ──
        _presc_chart_ctr = st.container(border=True)
        _presc_chart_ctr.markdown(f"**Observed Score: {_presc_obs_avg} avg**")
        fig_presc = go.Figure()
        for label, tier_key in _presc_tier_rows:
            y_vals = []
            for cat in _presc_cats:
                s = 10.0 if tier_key == "max" else presc_tier_scores.get(cat, {}).get(tier_key)
                y_vals.append(s)
            is_benchmark = tier_key == "avg"
            fig_presc.add_trace(go.Scatter(
                x=_presc_cats, y=y_vals, name=label, mode="lines+markers",
                line=dict(color=_presc_tier_colors.get(label, ACCENT), width=2,
                          dash="dash" if is_benchmark else "solid", shape="spline"),
                marker=dict(size=7),
            ))
        fig_presc.update_yaxes(range=[0, 10.5], dtick=2)
        dark_layout(fig_presc, height=400)
        fig_presc.update_layout(margin=dict(b=100), legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10)))
        _presc_chart_ctr.plotly_chart(fig_presc, use_container_width=True)

        # ── 2) Scorecard table (sorted by avg score descending) ──
        _presc_hdr = "".join(f"<th style='text-align:center; color:{MUTED}; font-size:12px; "
                             f"text-transform:uppercase; padding:8px 16px; border-bottom:1px solid {BORDER};'>{c}</th>"
                             for c in _presc_cats + ["Avg"])
        def _presc_tier_avg(tier_key):
            scores = [10.0 if tier_key == "max" else presc_tier_scores.get(c, {}).get(tier_key) for c in _presc_cats]
            valid = [s for s in scores if s is not None]
            return round(np.mean(valid), 1) if valid else 0
        _presc_tier_sorted = sorted(_presc_tier_rows, key=lambda t: _presc_tier_avg(t[1]), reverse=True)
        _presc_html_rows = ""
        for label, tier_key in _presc_tier_sorted:
            scores = []
            for cat in _presc_cats:
                s = 10.0 if tier_key == "max" else presc_tier_scores.get(cat, {}).get(tier_key)
                scores.append(s)
            avg_s = round(np.mean([s for s in scores if s is not None]), 1) if any(s is not None for s in scores) else None
            scores.append(avg_s)
            _base_label = label.split(" (")[0] if " (" in label else label
            dot_c = _presc_tier_colors.get(_base_label, _dot_color_presc(avg_s))
            label_html = (f"<td style='padding:10px 12px; border-bottom:1px solid {BORDER}; white-space:nowrap;'>"
                          f"<span style='color:{dot_c}; font-size:16px; margin-right:8px;'>&#9632;</span>"
                          f"<span style='color:#1e293b; font-size:13px;'>{label}</span></td>")
            cells = ""
            for s in scores:
                val = f"{s:.1f}" if s is not None else "—"
                c = _dot_color_presc(s)
                cells += (f"<td style='text-align:center; padding:10px 16px; color:{c}; "
                          f"font-weight:600; font-size:14px; border-bottom:1px solid {BORDER};'>{val}</td>")
            _presc_html_rows += f"<tr>{label_html}{cells}</tr>\n"

        _presc_table_html = (
            f"<table style='width:100%; border-collapse:collapse; background:rgba(0,0,0,0);'>"
            f"<thead><tr>"
            f"<th style='text-align:left; color:{MUTED}; font-size:12px; text-transform:uppercase; "
            f"padding:8px 12px; border-bottom:1px solid {BORDER};'>Participant</th>"
            f"{_presc_hdr}"
            f"</tr></thead>"
            f"<tbody>{_presc_html_rows}</tbody>"
            f"</table>"
        )
        _presc_chart_ctr.markdown(_presc_table_html, unsafe_allow_html=True)

        # ── 3) Alpha Performer differentiator cards ──
        _presc_alpha_scores = [r["alpha_score"] for rows in presc_categories.values() for r in rows if r.get("alpha_score")]
        _presc_detail = {cat: [{"KPI": r["KPI"], "org_fmt": r["org_fmt"], "score": r["score"],
                                "alpha": r["alpha"]} for r in rows]
                        for cat, rows in presc_categories.items()}
        _presc_cards_html = _alpha_cards_html(
            presc_categories, _presc_alpha_label,
            min(_presc_alpha_scores) if _presc_alpha_scores else None,
            max(_presc_alpha_scores) if _presc_alpha_scores else None,
            detail_rows=_presc_detail)
        if _presc_cards_html:
            st.markdown(_presc_cards_html, unsafe_allow_html=True)

        # ── 4) Improvement Roadmap ──
        _presc_roadmap_html = _render_roadmap(presc_categories, presc_tier_scores)
        if _presc_roadmap_html:
            st.markdown(_presc_roadmap_html, unsafe_allow_html=True)

with tab_others:
    tab_spend, tab_benchmarks, tab_bizcase, tab_budvar = st.tabs([
        "📊 Spend Overview", "📈 Benchmarks",
        "📂 Asset ROI Analysis", "📋 Budget Variance Performance"])

# ════════════════════════
# TAB 1: SPEND
# ════════════════════════
with tab_spend:
    total_spend = by_year["total"].sum()
    last, prev = by_year.iloc[-1], by_year.iloc[-2] if len(by_year)>1 else None
    yoy = (last["total"]-prev["total"])/prev["total"]*100 if prev is not None else 0
    peak = by_year.loc[by_year["total"].idxmax()]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Cumulative Spend", fmt(total_spend), f"{int(by_year['year'].min())}–{LAST_HIST_YEAR}")
    c2.metric(f"{LAST_HIST_YEAR} Total", fmt(last["total"]), f"{'▼' if yoy<0 else '▲'} {abs(yoy):.1f}% YoY")
    c3.metric("Peak Year", str(int(peak["year"])), fmt(peak["total"]))
    c4.metric("Annual Run Rate", fmt(df["Annual_Run_Cost"].sum()), "Current portfolio")

    st.markdown("---")
    st.subheader("Cost Composition")
    cost_data = by_year.copy()
    if show_proj:
        pt,_,_ = gen_proj(by_year,dp,wp,dg,wg,depts,wts,avg_init,noise,proj_years,LAST_HIST_YEAR)
        cost_data = pd.concat([cost_data,pt], ignore_index=True)
    cost_data = cost_data[cost_data["year"] >= 2020]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cost_data["year"], y=cost_data["total"],
        line=dict(color="#2563eb", width=2.5), mode="lines",
        fill="tozeroy", fillcolor=hex_to_rgba("#2563eb", 0.15),
        hovertemplate="Year %{x}<br>Total: %{y:$,.0f}<extra></extra>"))
    if show_proj: fig.add_vline(x=LAST_HIST_YEAR,line_dash="dash",line_color=MUTED,annotation_text="Projected →",annotation_font_color=MUTED,annotation_font_size=10)
    dark_layout(fig,340)
    fig.update_layout(showlegend=False, yaxis_title="Total Cost")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spend Breakdown")
    view = st.radio("View by",["Sector / Department","Worktype"],horizontal=True,label_visibility="collapsed")

    # Build breakdown data for last historical year
    hist_ydf = ydf[ydf["year"] <= LAST_HIST_YEAR]
    if view == "Sector / Department":
        bd_raw = hist_ydf[hist_ydf["year"] == LAST_HIST_YEAR].groupby(["sector", "department"])[["initCost","runCost","devCost"]].sum().reset_index()
        bd_raw["spend"] = bd_raw[["initCost","runCost","devCost"]].sum(axis=1)
        bd_raw = bd_raw[bd_raw["spend"] > 0]
        # Sector-level for pie
        bd_sector = bd_raw.groupby("sector")["spend"].sum().reset_index()
        bd_sector.columns = ["category", "spend"]
        bd_sector = bd_sector.sort_values("spend", ascending=False)
        colors_map = SECTOR_COLORS
        bd_colors = [colors_map.get(c, ACCENT) for c in bd_sector["category"]]
    else:
        bd = wp[wp["year"] == LAST_HIST_YEAR].groupby("worktype")["total"].sum().reset_index()
        bd.columns = ["category", "spend"]
        bd["layer"] = bd["category"].map(wt_layer)
        colors_map = WT_COLORS
        bd = bd[bd["spend"] > 0].sort_values("spend", ascending=False)
        bd_colors = [colors_map.get(c, ACCENT) for c in bd["category"]]

    col_pie, col_tree = st.columns(2)

    # ── Left: Pie chart ──
    with col_pie:
        pie_data = bd_sector if view == "Sector / Department" else bd
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_data["category"], values=pie_data["spend"],
            marker=dict(colors=[colors_map.get(c, ACCENT) for c in pie_data["category"]],
                        line=dict(color=DARK_BG, width=2)),
            textinfo="label+percent", textfont=dict(size=11, color=TEXT),
            hovertemplate="%{label}<br>%{value:$,.0f}<br>%{percent}<extra></extra>",
            hole=0.4)])
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=11), height=400,
            margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Right: Treemap (drilldown) ──
    with col_tree:
        if view == "Sector / Department":
            fig_tm = px.treemap(bd_raw, path=["sector", "department"], values="spend",
                color="sector", color_discrete_map=SECTOR_COLORS,
                hover_data={"spend": ":$,.0f"})
        elif "layer" in bd.columns:
            fig_tm = px.treemap(bd, path=["layer", "category"], values="spend",
                color="category", color_discrete_map=colors_map,
                hover_data={"spend": ":$,.0f"})
        else:
            fig_tm = px.treemap(bd, path=["category"], values="spend",
                color="category", color_discrete_map=colors_map,
                hover_data={"spend": ":$,.0f"})
        fig_tm.update_traces(
            textinfo="label+value+percent root",
            texttemplate="%{label}<br>%{value:$,.0f}<br>%{percentRoot:.1%}",
            textfont=dict(size=11))
        fig_tm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=11), height=400,
            margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_tm, use_container_width=True)

    _cap_drilldown = " · Treemap: click a sector to drill into departments" if view == "Sector / Department" else ""
    _cap_drilldown = " · Treemap grouped by Application/Infrastructure layer" if view == "Worktype" else _cap_drilldown
    st.caption(f"{LAST_HIST_YEAR} spend distribution" + _cap_drilldown)


# ════════════════════════
# TAB 2: OPTIMAL PERFORMANCE & SAVINGS
# ════════════════════════
with tab_totalsav:

    # spend_comp is pre-computed above the tabs

    # Metric cards — projected years only
    proj_data = spend_comp[spend_comp["year"] > LAST_HIST_YEAR]
    sum_trend = proj_data["current_trend"].sum()
    sum_optimized = proj_data["optimized"].sum()
    sum_optimal = proj_data["optimal"].sum()
    proj_label = f"Projected {proj_years}yr" if show_proj else "No projections"

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Trend (Projected)", fmt(sum_trend) if show_proj else "—", proj_label)
    c2.metric("Optimized (Projected)", fmt(sum_optimized) if show_proj else "—",
        f"−{fmt(sum_trend - sum_optimized)} vs trend" if show_proj and sum_trend > 0 else "")
    c3.metric("Optimal (Projected)", fmt(sum_optimal) if show_proj else "—",
        f"−{fmt(sum_trend - sum_optimal)} vs trend" if show_proj and sum_trend > 0 else "")

    st.markdown("---")

    # Spend comparison chart — only show from 2020 onward
    st.subheader("Spend Comparison: Current Trend vs. Optimized vs. Optimal")
    st.caption("Current Trend = as-is · Optimized = 3yr ramp (33%→66%→100%) · Optimal = full consolidation + early pay + payment management + vendor optimization + AI work modernization opportunities applied")

    plot_data = spend_comp[spend_comp["year"] >= 2020]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=plot_data["year"], y=plot_data["current_trend"], name="Current Trend",
        line=dict(color="#dc2626", width=2.5), mode="lines"))
    fig_comp.add_trace(go.Scatter(x=plot_data["year"], y=plot_data["optimized"], name="Optimized",
        line=dict(color="#ca8a04", width=2.5, dash="dash"), mode="lines",
        fill="tonexty", fillcolor="rgba(202,138,4,0.12)"))
    fig_comp.add_trace(go.Scatter(x=plot_data["year"], y=plot_data["optimal"], name="Optimal",
        line=dict(color="#059669", width=2), mode="lines",
        fill="tonexty", fillcolor="rgba(5,150,105,0.12)"))

    if show_proj:
        fig_comp.add_vline(x=LAST_HIST_YEAR, line_dash="dash", line_color=MUTED,
            annotation_text="Projected →", annotation_font_color=MUTED, annotation_font_size=10)
    dark_layout(fig_comp, 420)
    fig_comp.update_layout(yaxis_title="Annual Spend", yaxis_rangemode="tozero")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption("Yellow fill = optimized opportunity (3yr ramp) · Green fill = additional optimal opportunity")

    st.markdown("---")

    # Year-by-year gaps breakdown (stacked bar)
    st.subheader("Year-by-Year Opportunities Breakdown")
    st.caption("Asset Management = eliminated component costs · Work Modernization = work modernization potential · Vendor Incentives = negotiation + prevented increase + market repricing · Payment Management = payment process optimization · Early Pay = gross discount − cost of cash")

    gap_years = spend_comp[spend_comp["year"] >= 2020].copy()
    gap_years["Asset Management"] = gap_years["savings"]
    gap_years["Work Modernization"] = gap_years["ai_savings"]
    gap_years["Vendor Incentives"] = gap_years["vi_savings"]
    gap_years["Payment Management"] = gap_years["pm_savings"]
    gap_years["Early Pay"] = gap_years["ep_savings"]
    gap_years["Total"] = gap_years["Asset Management"] + gap_years["Work Modernization"] + gap_years["Vendor Incentives"] + gap_years["Payment Management"] + gap_years["Early Pay"]
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(x=gap_years["year"], y=gap_years["Asset Management"], name="Asset Management",
        marker_color="rgba(59,130,246,0.7)"))
    fig_gap.add_trace(go.Bar(x=gap_years["year"], y=gap_years["Work Modernization"], name="Work Modernization",
        marker_color="rgba(34,211,238,0.7)"))
    fig_gap.add_trace(go.Bar(x=gap_years["year"], y=gap_years["Vendor Incentives"], name="Vendor Incentives",
        marker_color="rgba(168,85,247,0.7)"))
    fig_gap.add_trace(go.Bar(x=gap_years["year"], y=gap_years["Payment Management"], name="Payment Management",
        marker_color="rgba(251,191,36,0.7)"))
    fig_gap.add_trace(go.Bar(x=gap_years["year"], y=gap_years["Early Pay"], name="Early Pay",
        marker_color="rgba(52,211,153,0.7)"))
    fig_gap.update_layout(barmode="stack", legend=dict(traceorder="normal"))
    dark_layout(fig_gap, 360)
    st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("---")

    # Summary table
    with st.expander("Efficiency Opportunities", expanded=False):
        # Rebuild with raw numeric values (comp_det may have fmt applied in rationalization tab)
        raw_comp_det, _ = compute_savings(df, alloc, scenario)

        summary_rows = []
        if len(raw_comp_det) > 0:
            for _, r in raw_comp_det.iterrows():
                amt = r["Savings"] * consol_scale
                if amt > 0:
                    summary_rows.append({
                        "Source": f"{r['Component']} ({r['Specificity']})",
                        "Type": "Asset Management",
                        "Opportunity": amt,
                    })

        if len(vendor_summary) > 0:
            for size in ["Large", "Mid-Size", "Small"]:
                vs_grp = vendor_summary[vendor_summary["Estimated_Size"] == size]
                if len(vs_grp) > 0:
                    net = vs_grp["Net_Benefit"].sum()
                    if abs(net) > 0:
                        summary_rows.append({
                            "Source": f"{size} vendors",
                            "Type": "Early Pay",
                            "Opportunity": net,
                        })

        if not inc_by_year_raw.empty:
            total_neg = inc_by_year_raw["negotiation"].sum()
            total_prev = inc_by_year_raw["prevented"].sum()
            if total_neg > 0:
                summary_rows.append({
                    "Source": "Price Negotiation",
                    "Type": "Vendor Incentives",
                    "Opportunity": total_neg,
                })
            if total_prev > 0:
                summary_rows.append({
                    "Source": "Prevented Increase",
                    "Type": "Vendor Incentives",
                    "Opportunity": total_prev,
                })
        mkt_total_opp = mkt_repricing_by_year["mkt_opp"].sum() if len(mkt_repricing_by_year) > 0 else 0
        if mkt_total_opp > 0:
            summary_rows.append({
                "Source": "Market Repricing (vs Mkt Avg)",
                "Type": "Vendor Incentives",
                "Opportunity": mkt_total_opp,
            })

        # Work Modernization gap
        ai_total_gap = ai_gap_by_year["ai_gap"].sum() if len(ai_gap_by_year) > 0 else 0
        if ai_total_gap > 0:
            summary_rows.append({
                "Source": f"AI Adoption ({ai_scenario})",
                "Type": "Work Modernization",
                "Opportunity": ai_total_gap,
            })

        if summary_rows:
            stdf = pd.DataFrame(summary_rows)
            totals = pd.DataFrame([{"Source": "TOTAL", "Type": "", "Opportunity": stdf["Opportunity"].sum()}])
            stdf = pd.concat([stdf, totals], ignore_index=True)
            stdf["% Portfolio"] = (stdf["Opportunity"] / portfolio_total * 100).apply(lambda x: f"{x:.1f}%")
            stdf["Opportunity"] = stdf["Opportunity"].apply(fmt)
            st.dataframe(stdf, use_container_width=True, hide_index=True)
            if mkt_total_opp > 0:
                st.caption("Note: Vendor Incentives = negotiation + prevented increase + market repricing (additive).")
        else:
            st.info("No opportunity data available.")


# ════════════════════════

# ════════════════════════
# TAB 3: VENDOR BENCHMARKS
# ════════════════════════
with tab_benchmarks:
    tab_bench_int, tab_bench_mkt = st.tabs(["📈 Internal Benchmarks", "🌐 Market Benchmarks"])

    # ════════════════════════════════════════
    # SUB-TAB: INTERNAL BENCHMARKS
    # ════════════════════════════════════════
    with tab_bench_int:
        st.subheader("Internal Benchmarks")
        st.caption("How does each vendor compare against portfolio averages? "
                   "Per-worktype view with top performer details and cost translation.")

        bench_layer = st.radio("Layer", ["Applications", "Infrastructure"], horizontal=True, key="bench_layer")
        df_bench = df[df["Layer"] == bench_layer].copy()
        df_bench["TotalAnnual"] = df_bench["Annual_Run_Cost"] + df_bench["Annual_Dev_Cost"]

        # Only show vendors with at least 2 assets
        vendor_counts = df_bench.groupby("Vendor")["Asset_ID"].count()
        active_vendors = vendor_counts[vendor_counts >= 2].index.tolist()

        # Worktype selector
        _ib_wt_list = sorted(df_bench["Worktype"].unique())
        _ib_sel_wt = st.selectbox("Worktype", _ib_wt_list, key="ib_wt_sel")

        _ib_wt_df = df_bench[df_bench["Worktype"] == _ib_sel_wt].copy()
        _ib_wt_avg = _ib_wt_df["TotalAnnual"].mean()

        # Per-vendor stats for this worktype
        _ib_vw = _ib_wt_df.groupby("Vendor").agg(
            Assets=("Asset_ID", "count"),
            AvgCost=("TotalAnnual", "mean"),
            TotalSpend=("TotalAnnual", "sum")).reset_index()
        _ib_vw["Benchmark"] = _ib_wt_avg
        _ib_vw["Delta %"] = ((_ib_vw["AvgCost"] - _ib_vw["Benchmark"]) / _ib_vw["Benchmark"] * 100).round(1)
        _ib_vw = _ib_vw.sort_values("AvgCost")

        _ib_active = _ib_vw[_ib_vw["Vendor"].isin(active_vendors)].copy()

        # ── Internal Benchmark — Worktype ──
        st.markdown("### Internal Benchmark — Worktype")

        if len(_ib_active) > 0:
            _ib_top = _ib_active.iloc[0]
            _ib_top_vendor = _ib_top["Vendor"]
            _ib_top_avg = _ib_top["AvgCost"]
            _ib_top_assets = int(_ib_top["Assets"])
            _ib_top_delta = _ib_top["Delta %"]

            _ib_left, _ib_right = st.columns([1.3, 3])
            with _ib_left:
                st.markdown(
                    f"""<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
                    padding:16px 18px;height:100%;">
                    <span style="color:#34d399;font-weight:700;font-size:13px;">TOP PERFORMER</span>
                    <br><span style="color:{TEXT};font-weight:600;font-size:16px;">{_ib_top_vendor}</span>
                    <br><span style="color:{MUTED};font-size:13px;">Avg Cost: {fmt(_ib_top_avg)}/asset</span>
                    <br><span style="color:{MUTED};font-size:12px;">{_ib_top_assets} assets · {_ib_top_delta:+.1f}% vs avg</span>
                    <br><br><span style="color:{MUTED};font-size:11px;">Portfolio avg: {fmt(_ib_wt_avg)}/asset</span>
                    </div>""", unsafe_allow_html=True)

            with _ib_right:
                fig_ib = go.Figure()
                _ib_min_cost = _ib_active["AvgCost"].min()
                _ib_max_cost = _ib_active["AvgCost"].max()
                fig_ib.add_trace(go.Scatter(
                    x=[_ib_min_cost, _ib_max_cost], y=[_ib_sel_wt, _ib_sel_wt],
                    mode="lines", line=dict(color=MUTED, width=8),
                    showlegend=False,
                    hovertemplate=f"Vendor Range: {fmt(_ib_min_cost)} – {fmt(_ib_max_cost)}<extra></extra>"))
                for _, vrow in _ib_active.iterrows():
                    fig_ib.add_trace(go.Scatter(
                        x=[vrow["AvgCost"]], y=[_ib_sel_wt],
                        mode="markers", marker=dict(color=MUTED, size=8, symbol="circle"),
                        showlegend=False,
                        hovertemplate=f"{vrow['Vendor']}<br>Avg: {fmt(vrow['AvgCost'])}/asset<br>{vrow['Delta %']:+.1f}% vs avg<extra></extra>"))
                fig_ib.add_trace(go.Scatter(
                    x=[_ib_top_avg], y=[_ib_sel_wt],
                    mode="markers", marker=dict(color="#059669", size=16, symbol="star"),
                    name=f"Top: {_ib_top_vendor}",
                    hovertemplate=f"Top: {_ib_top_vendor} ({fmt(_ib_top_avg)})<extra></extra>"))
                fig_ib.add_trace(go.Scatter(
                    x=[_ib_wt_avg], y=[_ib_sel_wt],
                    mode="markers", marker=dict(color=ACCENT, size=14, symbol="diamond"),
                    name="Portfolio Average",
                    hovertemplate=f"Portfolio Avg: {fmt(_ib_wt_avg)}<extra></extra>"))
                _ib_worst = _ib_active.iloc[-1]
                fig_ib.add_trace(go.Scatter(
                    x=[_ib_worst["AvgCost"]], y=[_ib_sel_wt],
                    mode="markers", marker=dict(color="#dc2626", size=14, symbol="circle"),
                    name=f"Highest: {_ib_worst['Vendor']}",
                    hovertemplate=f"Highest: {_ib_worst['Vendor']} ({fmt(_ib_worst['AvgCost'])})<extra></extra>"))
                fig_ib.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                    line=dict(color=MUTED, width=6), name="Vendor Range"))
                dark_layout(fig_ib, 200)
                fig_ib.update_layout(xaxis_title="Avg Annual Cost / Asset ($)", margin=dict(l=150))
                st.plotly_chart(fig_ib, use_container_width=True)

            st.caption("Gray dots = individual vendors · Green star = top performer · "
                       "Blue diamond = portfolio avg · Red circle = highest cost vendor")

            _ib_actual_spend = _ib_wt_df["TotalAnnual"].sum()
            _ib_n_assets = len(_ib_wt_df)
            _ib_whatif_top = _ib_top_avg * _ib_n_assets
            _ib_whatif_avg = _ib_wt_avg * _ib_n_assets
            _ib_gap_top = _ib_actual_spend - _ib_whatif_top
            _ib_k1, _ib_k2, _ib_k3 = st.columns(3)
            _ib_k1.metric(f"{_ib_sel_wt} — Actual Spend", fmt(_ib_actual_spend))
            _ib_k2.metric("What-if at Portfolio Avg", fmt(_ib_whatif_avg))
            _ib_k3.metric("What-if at Top Performer", fmt(_ib_whatif_top),
                           delta=f"{_ib_gap_top / _ib_actual_spend * 100:.1f}% gap" if _ib_actual_spend else "")

            with st.expander(f"Vendor Details — {_ib_sel_wt}", expanded=False):
                _ib_disp = _ib_vw[["Vendor", "Assets", "AvgCost", "TotalSpend", "Benchmark", "Delta %"]].copy()
                _ib_disp["AvgCost"] = _ib_disp["AvgCost"].apply(fmt)
                _ib_disp["TotalSpend"] = _ib_disp["TotalSpend"].apply(fmt)
                _ib_disp["Benchmark"] = _ib_disp["Benchmark"].apply(fmt)
                _ib_disp["Delta %"] = _ib_disp["Delta %"].apply(lambda x: f"{x:+.1f}%")
                _ib_disp = _ib_disp.rename(columns={
                    "AvgCost": "Avg Cost/Asset", "TotalSpend": "Total Spend",
                    "Benchmark": "Portfolio Avg", "Delta %": "Delta vs Avg"})
                st.dataframe(_ib_disp, use_container_width=True, hide_index=True)
        else:
            st.info(f"Not enough vendors with 2+ assets in {_ib_sel_wt} for benchmarking.")

        # ── Internal Benchmark — Total ──
        st.markdown("### Internal Benchmark — Total")
        st.caption("All worktypes: actual spend vs what-if every asset were at top-performer cost.")

        _ib_all_rows = []
        for _wt in _ib_wt_list:
            _wdf = df_bench[df_bench["Worktype"] == _wt]
            _wvw = _wdf.groupby("Vendor").agg(AvgCost=("TotalAnnual", "mean")).reset_index()
            _wvw_active = _wvw[_wvw["Vendor"].isin(active_vendors)]
            if len(_wvw_active) == 0:
                continue
            _w_actual = _wdf["TotalAnnual"].sum()
            _w_avg = _wdf["TotalAnnual"].mean()
            _w_top_cost = _wvw_active["AvgCost"].min()
            _w_n = len(_wdf)
            _w_at_top = _w_top_cost * _w_n
            _w_at_avg = _w_avg * _w_n
            _ib_all_rows.append({
                "Worktype": _wt, "actual_spend": _w_actual,
                "spend_at_avg": _w_at_avg, "spend_at_top": _w_at_top,
                "gap_vs_top": _w_actual - _w_at_top})

        if _ib_all_rows:
            _ib_all = pd.DataFrame(_ib_all_rows).sort_values("actual_spend", ascending=True)

            _ib_tot_actual = _ib_all["actual_spend"].sum()
            _ib_tot_avg = _ib_all["spend_at_avg"].sum()
            _ib_tot_top = _ib_all["spend_at_top"].sum()
            _ib_tot_gap = _ib_tot_actual - _ib_tot_top
            _tc1, _tc2, _tc3, _tc4 = st.columns(4)
            _tc1.metric("Total Actual Spend", fmt(_ib_tot_actual))
            _tc2.metric("At Portfolio Avg", fmt(_ib_tot_avg))
            _tc3.metric("At Top Performer", fmt(_ib_tot_top))
            _tc4.metric("Gap vs Top", fmt(_ib_tot_gap))

            fig_ib_tot = go.Figure()
            fig_ib_tot.add_trace(go.Bar(
                y=_ib_all["Worktype"], x=_ib_all["actual_spend"],
                orientation="h", name="Actual Spend",
                marker_color="#dc2626", opacity=0.85,
                text=[fmt(v) for v in _ib_all["actual_spend"]],
                textposition="auto"))
            fig_ib_tot.add_trace(go.Bar(
                y=_ib_all["Worktype"], x=_ib_all["spend_at_avg"],
                orientation="h", name="At Portfolio Avg",
                marker_color=ACCENT, opacity=0.85,
                text=[fmt(v) for v in _ib_all["spend_at_avg"]],
                textposition="auto"))
            fig_ib_tot.add_trace(go.Bar(
                y=_ib_all["Worktype"], x=_ib_all["spend_at_top"],
                orientation="h", name="At Top Performer",
                marker_color="#059669", opacity=0.85,
                text=[fmt(v) for v in _ib_all["spend_at_top"]],
                textposition="auto"))
            dark_layout(fig_ib_tot, max(320, len(_ib_all) * 60))
            fig_ib_tot.update_layout(barmode="group", xaxis_title="Annual Spend ($)",
                                      margin=dict(l=150))
            st.plotly_chart(fig_ib_tot, use_container_width=True)
            st.caption("Red = actual spend · Blue = at portfolio avg · Green = at top performer rate")

            with st.expander("Internal Benchmark Detail", expanded=False):
                _ib_all_disp = _ib_all[["Worktype", "actual_spend", "spend_at_avg",
                                         "spend_at_top", "gap_vs_top"]].copy()
                for _c in ["actual_spend", "spend_at_avg", "spend_at_top", "gap_vs_top"]:
                    _ib_all_disp[_c] = _ib_all_disp[_c].apply(fmt)
                _ib_all_disp = _ib_all_disp.rename(columns={
                    "actual_spend": "Actual", "spend_at_avg": "At Avg",
                    "spend_at_top": "At Top", "gap_vs_top": "Gap vs Top"})
                st.dataframe(_ib_all_disp, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════
    # SUB-TAB: MARKET BENCHMARKS
    # ════════════════════════════════════════
    with tab_bench_mkt:
        st.subheader("Market Benchmarks")
        st.caption("Per-worktype KPI benchmarking against external vendor data, "
                   "with cost translation showing what org spend would be at market rates.")

        if not app_bench.empty or infra_bench:
            rng_bench = np.random.default_rng(42)
            org_premium = rng_bench.uniform(0.20, 0.30)
            org_premium_pct = round(org_premium * 100, 1)

            mkt_layer = st.radio("Layer", ["Applications", "Infrastructure"],
                                 horizontal=True, key="mkt_bench_layer")

            # ══════════════════════════
            # APPLICATIONS
            # ══════════════════════════
            if mkt_layer == "Applications":
                if not app_bench.empty:
                    app_wt_list = sorted(app_bench["Worktype"].unique())
                    sel_wt = st.selectbox("Worktype", app_wt_list, key="mkt_app_wt")

                    wt_data = app_bench[app_bench["Worktype"] == sel_wt].copy()
                    wt_data = wt_data.sort_values("Midpoint")

                    mkt_min = wt_data["Min Cost / MAU"].min()
                    mkt_max = wt_data["Max Cost / MAU"].max()
                    mkt_mean = wt_data["Midpoint"].mean()
                    top_idx = wt_data["Midpoint"].idxmin()
                    top_cost = wt_data.loc[top_idx, "Midpoint"]
                    top_vendor = wt_data.loc[top_idx, "Vendor"]
                    org_est = mkt_mean * (1 + org_premium)

                    _app_df = df[df["Layer"] == "Applications"]
                    _sel_wt_assets = _app_df[_app_df["Worktype"] == sel_wt]
                    _sel_actual = _sel_wt_assets["Annual_Run_Cost"].sum() + _sel_wt_assets["Annual_Dev_Cost"].sum()
                    _sel_vol = _sel_actual / org_est if org_est > 0 else 0
                    _sel_at_avg = mkt_mean * _sel_vol
                    _sel_at_top = top_cost * _sel_vol

                    # ── Market Benchmark — Worktype ──
                    st.markdown("### Market Benchmark — Worktype")
                    st.info(f"Org premium: **{org_premium_pct}%** above market mean (seeded)")

                    top_driver_raw = wt_data.loc[top_idx, "Key Infrastructure Drivers"]
                    top_driver = str(top_driver_raw) if pd.notna(top_driver_raw) else ""
                    top_url_raw = wt_data.loc[top_idx, "Reference URLs"]
                    top_url = str(top_url_raw).strip() if pd.notna(top_url_raw) else ""

                    _left_tp, _right_chart = st.columns([1.3, 3])
                    with _left_tp:
                        st.markdown(
                            f"""<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
                            padding:16px 18px;height:100%;">
                            <span style="color:#34d399;font-weight:700;font-size:13px;">TOP PERFORMER</span>
                            <br><span style="color:{TEXT};font-weight:600;font-size:16px;">{top_vendor}</span>
                            <br><span style="color:{MUTED};font-size:13px;">${top_cost:.2f}/MAU</span>
                            <br><br><span style="color:{MUTED};font-size:11px;">{top_driver}</span>
                            {"<br><br><a href='" + top_url + "' style='color:" + ACCENT + ";font-size:12px;' target='_blank'>Reference &rarr;</a>" if top_url.startswith("http") else ""}
                            </div>""", unsafe_allow_html=True)
                    with _right_chart:
                        fig_db = go.Figure()
                        fig_db.add_trace(go.Scatter(
                            x=[mkt_min, mkt_max], y=[sel_wt, sel_wt],
                            mode="lines", line=dict(color=MUTED, width=8),
                            showlegend=False,
                            hovertemplate=f"Market Range: ${mkt_min:.2f} – ${mkt_max:.2f}<extra></extra>"))
                        fig_db.add_trace(go.Scatter(
                            x=[top_cost], y=[sel_wt],
                            mode="markers", marker=dict(color="#059669", size=16, symbol="star"),
                            name=f"Top: {top_vendor}",
                            hovertemplate=f"Top: {top_vendor} (${top_cost:.2f}/MAU)<extra></extra>"))
                        fig_db.add_trace(go.Scatter(
                            x=[mkt_mean], y=[sel_wt],
                            mode="markers", marker=dict(color=ACCENT, size=14, symbol="diamond"),
                            name="Market Average",
                            hovertemplate=f"Market Avg: ${mkt_mean:.2f}/MAU<extra></extra>"))
                        fig_db.add_trace(go.Scatter(
                            x=[org_est], y=[sel_wt],
                            mode="markers", marker=dict(color="#dc2626", size=16, symbol="circle"),
                            name=f"Org Estimate (+{org_premium_pct}%)",
                            hovertemplate=f"Org Est: ${org_est:.2f}/MAU<extra></extra>"))
                        fig_db.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                            line=dict(color=MUTED, width=6), name="Market Range (Min–Max)"))
                        dark_layout(fig_db, 200)
                        fig_db.update_layout(xaxis_title="Cost / MAU ($)", margin=dict(l=150))
                        st.plotly_chart(fig_db, use_container_width=True)
                    st.caption("Gray = market range · Green star = top performer · "
                               "Blue diamond = market mean · Red circle = org estimate")

                    _k1, _k2, _k3 = st.columns(3)
                    _k1.metric(f"{sel_wt} — Actual Spend", fmt(_sel_actual))
                    _k2.metric("What-if at Market Avg", fmt(_sel_at_avg),
                               delta=f"{(_sel_actual - _sel_at_avg) / _sel_actual * 100:.1f}% gap" if _sel_actual else "")
                    _k3.metric("What-if at Top Performer", fmt(_sel_at_top),
                               delta=f"{(_sel_actual - _sel_at_top) / _sel_actual * 100:.1f}% gap" if _sel_actual else "")

                    with st.expander(f"Vendor Details — {sel_wt}", expanded=False):
                        vt_disp = wt_data[["Vendor", "Min Cost / MAU", "Max Cost / MAU",
                                           "Midpoint", "Key Infrastructure Drivers", "Reference URLs"]].copy()
                        def _vendor_link(row):
                            url = str(row["Reference URLs"]).strip() if pd.notna(row["Reference URLs"]) else ""
                            if url.startswith("http"):
                                return f"[{row['Vendor']}]({url})"
                            return row["Vendor"]
                        vt_disp.insert(0, "Vendor Link", vt_disp.apply(_vendor_link, axis=1))
                        vt_disp = vt_disp.drop(columns=["Vendor", "Reference URLs"])
                        vt_disp = vt_disp.rename(columns={
                            "Vendor Link": "Vendor",
                            "Key Infrastructure Drivers": "Key Drivers"})
                        vt_disp["Min Cost / MAU"] = vt_disp["Min Cost / MAU"].apply(lambda x: f"${x:,.2f}")
                        vt_disp["Max Cost / MAU"] = vt_disp["Max Cost / MAU"].apply(lambda x: f"${x:,.2f}")
                        vt_disp["Midpoint"] = vt_disp["Midpoint"].apply(lambda x: f"${x:,.2f}")
                        st.markdown(vt_disp.to_markdown(index=False), unsafe_allow_html=True)

                    # ── Market Benchmark — Total ──
                    st.markdown("### Market Benchmark — Total")
                    st.caption("Translates KPI benchmarks into dollar spend using implied MAU volumes.")

                    app_df = df[df["Layer"] == "Applications"].copy()
                    app_spend_by_wt = app_df.groupby("Worktype").agg(
                        actual_spend=("Annual_Run_Cost", "sum")).reset_index()
                    app_spend_by_wt["actual_spend"] += app_df.groupby("Worktype")["Annual_Dev_Cost"].sum().values

                    bench_agg = app_bench.groupby("Worktype").agg(
                        mkt_min_mid=("Midpoint", "min"),
                        mkt_mean_mid=("Midpoint", "mean")).reset_index()
                    bench_agg["org_kpi"] = bench_agg["mkt_mean_mid"] * (1 + org_premium)

                    cost_comp = app_spend_by_wt.merge(bench_agg, on="Worktype", how="inner")
                    cost_comp["implied_vol"] = cost_comp["actual_spend"] / cost_comp["org_kpi"]
                    cost_comp["spend_at_avg"] = cost_comp["mkt_mean_mid"] * cost_comp["implied_vol"]
                    cost_comp["spend_at_top"] = cost_comp["mkt_min_mid"] * cost_comp["implied_vol"]
                    cost_comp["gap_vs_avg"] = cost_comp["actual_spend"] - cost_comp["spend_at_avg"]
                    cost_comp["gap_vs_top"] = cost_comp["actual_spend"] - cost_comp["spend_at_top"]
                    cost_comp = cost_comp.sort_values("actual_spend", ascending=True)

                    if not cost_comp.empty:
                        tot_actual = cost_comp["actual_spend"].sum()
                        tot_at_avg = cost_comp["spend_at_avg"].sum()
                        tot_at_top = cost_comp["spend_at_top"].sum()
                        tot_gap_avg = tot_actual - tot_at_avg
                        tot_gap_top = tot_actual - tot_at_top
                        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                        tc1.metric("Total Actual Spend", fmt(tot_actual))
                        tc2.metric("At Market Avg", fmt(tot_at_avg))
                        tc3.metric("At Top Performer", fmt(tot_at_top))
                        tc4.metric("Gap vs Avg", fmt(tot_gap_avg))
                        tc5.metric("Gap vs Top", fmt(tot_gap_top))

                        fig_cost = go.Figure()
                        fig_cost.add_trace(go.Bar(
                            y=cost_comp["Worktype"], x=cost_comp["actual_spend"],
                            orientation="h", name="Org Actual",
                            marker_color="#dc2626", opacity=0.85,
                            text=[fmt(v) for v in cost_comp["actual_spend"]],
                            textposition="auto"))
                        fig_cost.add_trace(go.Bar(
                            y=cost_comp["Worktype"], x=cost_comp["spend_at_avg"],
                            orientation="h", name="At Market Avg KPI",
                            marker_color=ACCENT, opacity=0.85,
                            text=[fmt(v) for v in cost_comp["spend_at_avg"]],
                            textposition="auto"))
                        fig_cost.add_trace(go.Bar(
                            y=cost_comp["Worktype"], x=cost_comp["spend_at_top"],
                            orientation="h", name="At Top Performer KPI",
                            marker_color="#059669", opacity=0.85,
                            text=[fmt(v) for v in cost_comp["spend_at_top"]],
                            textposition="auto"))
                        dark_layout(fig_cost, max(320, len(cost_comp) * 60))
                        fig_cost.update_layout(barmode="group", xaxis_title="Annual Spend ($)",
                                               margin=dict(l=150))
                        st.plotly_chart(fig_cost, use_container_width=True)
                        st.caption("Red = org actual spend · Blue = repriced at market avg · "
                                   "Green = repriced at top performer KPI")

                        with st.expander("Cost Comparison Detail", expanded=False):
                            cc_disp = cost_comp[["Worktype", "actual_spend", "spend_at_avg",
                                                 "spend_at_top", "gap_vs_avg", "gap_vs_top"]].copy()
                            for c in ["actual_spend", "spend_at_avg", "spend_at_top", "gap_vs_avg", "gap_vs_top"]:
                                cc_disp[c] = cc_disp[c].apply(fmt)
                            cc_disp = cc_disp.rename(columns={
                                "actual_spend": "Org Actual", "spend_at_avg": "At Mkt Avg",
                                "spend_at_top": "At Top Perf", "gap_vs_avg": "Gap vs Avg",
                                "gap_vs_top": "Gap vs Top"})
                            st.dataframe(cc_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No application benchmark data found in Benchmark.xlsx.")

            # ══════════════════════════
            # INFRASTRUCTURE — consistent with Applications
            # ══════════════════════════
            else:
                if infra_bench:
                    infra_wt_list = sorted(infra_bench.keys())
                    sel_infra_wt = st.selectbox("Worktype", infra_wt_list, key="mkt_infra_wt")

                    _NUMERIC_INFRA = {"Access"}
                    _infra_rename = {
                        "Communications": {
                            "Vendor": "Vendor",
                            "KPI: Cost per SMS Message": "Cost / SMS",
                            "KPI: Cost per Voice Minute (Outbound)": "Cost / Voice Min",
                            "Key Infrastructure Cost Drivers": "Key Drivers",
                            "Reference URLs": "References"},
                        "Integration": {
                            "Primary Cost Driver / Vendor": "Vendor / Model",
                            "Min Cost KPI": "Min Cost",
                            "Max Cost KPI": "Max Cost",
                            "How the Cost Model Works (Infrastructure Driver)": "Cost Model & Drivers",
                            "Reference URLs": "References"},
                        "Data": {
                            "Primary Cost Driver / Vendor": "Vendor / Model",
                            "Min Cost KPI (Monthly)": "Min Monthly Cost",
                            "Max Cost KPI (Monthly)": "Max Monthly Cost",
                            "How the Cost Model Works (Infrastructure Driver)": "Cost Model & Drivers",
                            "Reference URLs": "References"},
                    }

                    idf_raw = infra_bench[sel_infra_wt].copy()

                    if sel_infra_wt in _NUMERIC_INFRA:
                        acc_cols = list(idf_raw.columns)
                        vcol = acc_cols[0]
                        min_col = acc_cols[1] if len(acc_cols) > 1 else None
                        max_col = acc_cols[2] if len(acc_cols) > 2 else None
                        drv_col = acc_cols[3] if len(acc_cols) > 3 else None
                        ref_col_a = acc_cols[4] if len(acc_cols) > 4 else None

                        if min_col and max_col:
                            idf_raw["_min"] = pd.to_numeric(idf_raw[min_col], errors="coerce")
                            idf_raw["_max"] = pd.to_numeric(idf_raw[max_col], errors="coerce")
                            acc_num = idf_raw.dropna(subset=["_min", "_max"]).copy()

                            if len(acc_num) > 0:
                                acc_num["_mid"] = (acc_num["_min"] + acc_num["_max"]) / 2
                                i_mkt_min = acc_num["_min"].min()
                                i_mkt_max = acc_num["_max"].max()
                                i_mkt_mean = acc_num["_mid"].mean()
                                i_top_idx = acc_num["_mid"].idxmin()
                                i_top_cost = acc_num.loc[i_top_idx, "_mid"]
                                i_top_vendor = acc_num.loc[i_top_idx, vcol]
                                i_org_est = i_mkt_mean * (1 + org_premium)

                                # Pre-compute actual spend
                                _infra_df = df[df["Layer"] == "Infrastructure"]
                                _infra_wt_assets = _infra_df[_infra_df["Worktype"] == sel_infra_wt]
                                _infra_actual = _infra_wt_assets["Annual_Run_Cost"].sum() + _infra_wt_assets["Annual_Dev_Cost"].sum()
                                _infra_vol = _infra_actual / i_org_est if i_org_est > 0 else 0
                                _infra_at_avg = i_mkt_mean * _infra_vol
                                _infra_at_top = i_top_cost * _infra_vol

                                # Top performer details
                                i_top_drv_raw = acc_num.loc[i_top_idx, drv_col] if drv_col else None
                                i_top_drv = str(i_top_drv_raw) if pd.notna(i_top_drv_raw) else ""
                                i_top_url_raw = acc_num.loc[i_top_idx, ref_col_a] if ref_col_a else None
                                i_top_url = str(i_top_url_raw).strip() if pd.notna(i_top_url_raw) else ""

                                # ── Market Benchmark — Worktype ──
                                st.markdown("### Market Benchmark — Worktype")
                                st.info(f"Org premium: **{org_premium_pct}%** above market mean (seeded)")

                                _i_left, _i_right = st.columns([1.3, 3])
                                with _i_left:
                                    st.markdown(
                                        f"""<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
                                        padding:16px 18px;height:100%;">
                                        <span style="color:#34d399;font-weight:700;font-size:13px;">TOP PERFORMER</span>
                                        <br><span style="color:{TEXT};font-weight:600;font-size:16px;">{i_top_vendor}</span>
                                        <br><span style="color:{MUTED};font-size:13px;">${i_top_cost:.2f}/user/month</span>
                                        <br><br><span style="color:{MUTED};font-size:11px;">{i_top_drv}</span>
                                        {"<br><br><a href='" + i_top_url + "' style='color:" + ACCENT + ";font-size:12px;' target='_blank'>Reference &rarr;</a>" if i_top_url.startswith("http") else ""}
                                        </div>""", unsafe_allow_html=True)
                                with _i_right:
                                    fig_iacc = go.Figure()
                                    fig_iacc.add_trace(go.Scatter(
                                        x=[i_mkt_min, i_mkt_max], y=[sel_infra_wt, sel_infra_wt],
                                        mode="lines", line=dict(color=MUTED, width=8),
                                        showlegend=False,
                                        hovertemplate=f"Range: ${i_mkt_min:.2f} – ${i_mkt_max:.2f}<extra></extra>"))
                                    fig_iacc.add_trace(go.Scatter(
                                        x=[i_top_cost], y=[sel_infra_wt],
                                        mode="markers", marker=dict(color="#059669", size=16, symbol="star"),
                                        name=f"Top: {i_top_vendor}",
                                        hovertemplate=f"Top: {i_top_vendor} (${i_top_cost:.2f})<extra></extra>"))
                                    fig_iacc.add_trace(go.Scatter(
                                        x=[i_mkt_mean], y=[sel_infra_wt],
                                        mode="markers", marker=dict(color=ACCENT, size=14, symbol="diamond"),
                                        name="Market Average",
                                        hovertemplate=f"Market Avg: ${i_mkt_mean:.2f}<extra></extra>"))
                                    fig_iacc.add_trace(go.Scatter(
                                        x=[i_org_est], y=[sel_infra_wt],
                                        mode="markers", marker=dict(color="#dc2626", size=16, symbol="circle"),
                                        name=f"Org Estimate (+{org_premium_pct}%)",
                                        hovertemplate=f"Org Est: ${i_org_est:.2f}<extra></extra>"))
                                    fig_iacc.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                        line=dict(color=MUTED, width=6), name="Market Range (Min–Max)"))
                                    dark_layout(fig_iacc, 200)
                                    fig_iacc.update_layout(xaxis_title="$/User/Month", margin=dict(l=150))
                                    st.plotly_chart(fig_iacc, use_container_width=True)
                                st.caption("Gray = market range · Green star = top performer · "
                                           "Blue diamond = market mean · Red circle = org estimate")

                                # Spend cards — same pattern as Applications
                                _ik1, _ik2, _ik3 = st.columns(3)
                                _ik1.metric(f"{sel_infra_wt} — Actual Spend", fmt(_infra_actual))
                                _ik2.metric("What-if at Market Avg", fmt(_infra_at_avg),
                                            delta=f"{(_infra_actual - _infra_at_avg) / _infra_actual * 100:.1f}% gap" if _infra_actual else "")
                                _ik3.metric("What-if at Top Performer", fmt(_infra_at_top),
                                            delta=f"{(_infra_actual - _infra_at_top) / _infra_actual * 100:.1f}% gap" if _infra_actual else "")

                                # Vendor detail table
                                with st.expander(f"Vendor Details — {sel_infra_wt}", expanded=False):
                                    vt_rows = []
                                    for _, row in acc_num.iterrows():
                                        url = str(row[ref_col_a]).strip() if ref_col_a and pd.notna(row.get(ref_col_a)) else ""
                                        vendor_name = row[vcol]
                                        link = f"[{vendor_name}]({url})" if url.startswith("http") else vendor_name
                                        drv = str(row[drv_col]) if drv_col and pd.notna(row.get(drv_col)) else ""
                                        vt_rows.append({
                                            "Vendor": link,
                                            "Min $/User/Mo": f"${row['_min']:.2f}",
                                            "Max $/User/Mo": f"${row['_max']:.2f}",
                                            "Midpoint": f"${row['_mid']:.2f}",
                                            "Key Drivers": drv})
                                    st.markdown(pd.DataFrame(vt_rows).to_markdown(index=False),
                                                unsafe_allow_html=True)

                                # ── Market Benchmark — Total (single worktype for Access) ──
                                st.markdown("### Market Benchmark — Total")
                                st.caption("Cost translation for numeric infrastructure worktypes.")

                                if _infra_actual > 0:
                                    _it1, _it2, _it3, _it4, _it5 = st.columns(5)
                                    _it1.metric("Total Actual Spend", fmt(_infra_actual))
                                    _it2.metric("At Market Avg", fmt(_infra_at_avg))
                                    _it3.metric("At Top Performer", fmt(_infra_at_top))
                                    _it4.metric("Gap vs Avg", fmt(_infra_actual - _infra_at_avg))
                                    _it5.metric("Gap vs Top", fmt(_infra_actual - _infra_at_top))

                                    fig_ic = go.Figure()
                                    fig_ic.add_trace(go.Bar(
                                        y=[sel_infra_wt], x=[_infra_actual], orientation="h",
                                        name="Org Actual", marker_color="#dc2626", opacity=0.85,
                                        text=[fmt(_infra_actual)], textposition="auto"))
                                    fig_ic.add_trace(go.Bar(
                                        y=[sel_infra_wt], x=[_infra_at_avg], orientation="h",
                                        name="At Market Avg", marker_color=ACCENT, opacity=0.85,
                                        text=[fmt(_infra_at_avg)], textposition="auto"))
                                    fig_ic.add_trace(go.Bar(
                                        y=[sel_infra_wt], x=[_infra_at_top], orientation="h",
                                        name="At Top Performer", marker_color="#059669", opacity=0.85,
                                        text=[fmt(_infra_at_top)], textposition="auto"))
                                    dark_layout(fig_ic, 200)
                                    fig_ic.update_layout(barmode="group", xaxis_title="Annual Spend ($)",
                                                         margin=dict(l=150))
                                    st.plotly_chart(fig_ic, use_container_width=True)
                                    st.caption("Red = org actual spend · Blue = repriced at market avg · "
                                               "Green = repriced at top performer KPI")

                    else:
                        # Reference-only worktypes
                        _subtitles = {
                            "Communications": "Communications — Per-Message & Per-Minute Pricing",
                            "Integration": "Integration — Platform Pricing Models",
                            "Data": "Data — Compute & Storage Pricing"}
                        subtitle = _subtitles.get(sel_infra_wt, sel_infra_wt)
                        st.markdown(f"### Market Benchmark — Worktype")
                        st.caption(f"{subtitle} — Reference table only, no single numeric KPI for cost comparison.")

                        disp_idf = idf_raw.copy()
                        if sel_infra_wt in _infra_rename:
                            disp_idf = disp_idf.rename(columns=_infra_rename[sel_infra_wt])

                        ref_key = "References"
                        vendor_key = next((c for c in disp_idf.columns if "Vendor" in c or "Model" in c), None)
                        if vendor_key and ref_key in disp_idf.columns:
                            def _ilink(row):
                                url = str(row[ref_key]).strip() if pd.notna(row[ref_key]) else ""
                                name = row[vendor_key]
                                return f"[{name}]({url})" if url.startswith("http") else name
                            disp_idf[vendor_key] = disp_idf.apply(_ilink, axis=1)
                            disp_idf = disp_idf.drop(columns=[ref_key])

                        st.markdown(disp_idf.to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.info("No infrastructure benchmark data found in Benchmark.xlsx.")

        else:
            st.info("Place Benchmark.xlsx alongside the dashboard to enable external market comparisons.")


# ════════════════════════
# TAB: EFFICIENCY COMPONENTS (top-level)
# ════════════════════════
with tab_effcomp:
  tab_asset, tab_aimod, tab_incentives, tab_paymgmt, tab_epay, tab_bench_opp = st.tabs([
      "🏗️ Asset Management", "🤖 Work Modernization",
      "🏷️ Vendor Incentives", "💳 Payment Management", "💰 Early Pay",
      "📊 Vendor Incentives (Benchmarks)"])

  # ── Asset Management ──
  with tab_asset:
    tab_rat, tab_comp = st.tabs(["✂️ Rationalization", "🧩 Components"])

    with tab_rat:
      rat_layer = st.radio("Layer", ["Applications", "Infrastructure"], horizontal=True, key="rat_layer")

      if rat_layer == "Applications":
          rat_alloc = alloc
          rat_label = "Application"
      else:
          rat_alloc = infra_alloc
          rat_label = "Infrastructure"

      rat_comp_det, rat_consol = compute_savings(df, rat_alloc, scenario)
      rat_sav_by_year = compute_savings_by_year(df, rat_alloc, scenario)

      scaled_rat_consol = rat_consol * consol_scale
      c1,c2,c3 = st.columns(3)
      c1.metric(f"{rat_label} Asset Management Opportunity", fmt(scaled_rat_consol), f"{scaled_rat_consol/portfolio_total*100:.1f}% of portfolio")
      c2.metric("Early Pay Opportunity", fmt(ep_net), ep_preset)
      c3.metric("Combined Opportunity", fmt(scaled_rat_consol+ep_net), f"{(scaled_rat_consol+ep_net)/portfolio_total*100:.1f}% of portfolio")

      st.markdown("---")
      st.subheader(f"{rat_label} Asset Management")
      if rat_alloc is not None and len(rat_alloc) > 0 and len(rat_comp_det) > 0:
          # ── Cards: most common + most specialized component ──
          spec_counts = rat_alloc.groupby("Component").agg(
              total_assets=("Asset_ID", "nunique"),
              avg_spec=("Spec", "mean"),
              total_cost=("Total_Alloc", "sum")).reset_index()
          most_common = spec_counts.sort_values("total_assets", ascending=False).iloc[0]
          most_specialized = spec_counts.sort_values("avg_spec", ascending=False).iloc[0]
          cc1, cc2 = st.columns(2)
          cc1.metric("Most Common Component", most_common["Component"],
              f"{int(most_common['total_assets'])} assets · {fmt(most_common['total_cost'])}")
          cc2.metric("Most Specialized Component", most_specialized["Component"],
              f"Avg specificity {most_specialized['avg_spec']:.1f} · {int(most_specialized['total_assets'])} assets")

          st.markdown("---")

          # ── Chart: assets per component by specificity ──
          spec_chart = rat_alloc.groupby(["Component", "Spec"]).agg(
              count=("Asset_ID", "nunique")).reset_index()
          spec_chart["Specificity"] = spec_chart["Spec"].map(SPEC_LABELS)
          spec_pivot = spec_chart.pivot_table(index="Component", columns="Specificity", values="count", fill_value=0)
          spec_pivot = spec_pivot.reindex(columns=["Common", "Shared", "Specialized"], fill_value=0)
          spec_pivot = spec_pivot.sort_values(by=["Common", "Shared", "Specialized"], ascending=[False, False, False])

          fig_spec = go.Figure()
          spec_colors = {"Common": "rgba(5,150,105,0.8)", "Shared": "rgba(202,138,4,0.7)", "Specialized": "rgba(220,38,38,0.7)"}
          for sp in ["Common", "Shared", "Specialized"]:
              if sp in spec_pivot.columns:
                  fig_spec.add_trace(go.Bar(
                      y=spec_pivot.index, x=spec_pivot[sp],
                      name=sp, orientation="h", marker_color=spec_colors[sp]))
          fig_spec.update_layout(barmode="stack")
          dark_layout(fig_spec, max(280, len(spec_pivot) * 28))
          fig_spec.update_layout(xaxis_title="Assets")
          st.plotly_chart(fig_spec, use_container_width=True)
          st.caption("Common = shared across worktypes (consolidate most) · Shared = partial overlap · Specialized = unique (keep all)")

          st.markdown("---")

          # ── Opportunity bar chart ──
          cs = rat_comp_det.groupby("Component")[["Savings","Total Cost"]].sum().reset_index()
          cs.rename(columns={"Savings": "Opportunity"}, inplace=True)
          cs["Opportunity"] = cs["Opportunity"] * consol_scale
          cs["Keep"] = cs["Total Cost"]-cs["Opportunity"]
          cs = cs.sort_values("Opportunity",ascending=True)
          fig_s = go.Figure()
          fig_s.add_trace(go.Bar(y=cs["Component"],x=cs["Keep"],name="Retained",orientation="h",marker_color="rgba(59,130,246,0.4)"))
          fig_s.add_trace(go.Bar(y=cs["Component"],x=cs["Opportunity"],name="Opportunity",orientation="h",marker_color="rgba(52,211,153,0.8)"))
          fig_s.update_layout(barmode="stack")
          dark_layout(fig_s, max(280, len(cs) * 30))
          st.plotly_chart(fig_s, use_container_width=True)

          # ── Detail table in expander ──
          with st.expander("Component Detail", expanded=False):
              det = rat_comp_det.sort_values(["Component","Specificity"]).copy()
              det["Savings"] = det["Savings"] * consol_scale
              det["Savings %"] = (det["Savings"] / det["Total Cost"] * 100).where(det["Total Cost"] > 0, 0)
              det.rename(columns={"Savings": "Opportunity", "Savings %": "Opportunity %"}, inplace=True)
              det["Total Cost"] = det["Total Cost"].apply(fmt)
              det["Opportunity"] = det["Opportunity"].apply(fmt)
              det["Opportunity %"] = det["Opportunity %"].apply(lambda x: f"{x:.0f}%")
              st.dataframe(det, use_container_width=True, hide_index=True)
      else:
          st.info(f"No {rat_label.lower()} component allocation data available.")


    # ════════════════════════
    # SUB-TAB: COMPONENTS
    # ════════════════════════
    with tab_comp:
      comp_layer = st.radio("Layer", ["Applications", "Infrastructure"], horizontal=True, key="comp_layer")

      if comp_layer == "Applications":
          active_alloc = alloc
          layer_label = "Application"
      else:
          active_alloc = infra_alloc
          layer_label = "Infrastructure"

      if active_alloc is not None and len(active_alloc) > 0:
          n_comps = active_alloc["Component"].nunique()
          n_assets = active_alloc["Asset_ID"].nunique()

          # For infra: group into top N + Others
          if n_comps > 10:
              top_n = st.slider("Top components to show", 3, min(n_comps, 20), 10, key="top_n_comp")
              top_list = active_alloc.groupby("Component")["Total_Alloc"].sum().nlargest(top_n).index.tolist()
              active_alloc = active_alloc.copy()
              active_alloc["Component"] = active_alloc["Component"].apply(lambda x: x if x in top_list else "Others")
              st.caption(f"{layer_label}: {n_assets} assets · {n_comps} component types (showing top {top_n} + Others) · Cost = Weight × Total Cost")
          else:
              st.caption(f"{layer_label}: {n_assets} assets · {n_comps} component types · Cost = Weight × Total Cost")

          st.subheader("Cost by Component")
          ct = active_alloc.groupby("Component").agg(Run=("Run_Alloc","sum"),Dev=("Dev_Alloc","sum"),Init=("Init_Alloc","sum"),
              Total=("Total_Alloc","sum"),Apps=("Asset_ID","nunique")).reset_index().sort_values("Total",ascending=False)

          n_display = ct["Component"].nunique()
          if n_display > 8:
              ct_plot = ct.sort_values("Total", ascending=True)
              fig_b = go.Figure()
              for col,nm,clr in [("Init","Initial","#ea580c"),("Run","Run","#2563eb"),("Dev","Dev","#0891b2")]:
                  fig_b.add_trace(go.Bar(y=ct_plot["Component"],x=ct_plot[col],name=nm,orientation="h",marker_color=clr))
              fig_b.update_layout(barmode="stack")
              dark_layout(fig_b, max(360, n_display * 30))
          else:
              fig_b = go.Figure()
              for col,nm,clr in [("Init","Initial","#ea580c"),("Run","Run","#2563eb"),("Dev","Dev","#0891b2")]:
                  fig_b.add_trace(go.Bar(x=ct["Component"],y=ct[col],name=nm,marker_color=clr))
              fig_b.update_layout(barmode="stack")
              dark_layout(fig_b, 320)
          st.plotly_chart(fig_b, use_container_width=True)

          col1,col2 = st.columns(2)
          for title,grp_col,cscale,col_target in [("By Department","Department",["#e8edf2","#93b5e0","#2563eb"],col1),
                                                    ("By Worktype","Worktype",["#e8edf2","#b4a0d9","#7c3aed"],col2)]:
              with col_target:
                  st.subheader(title)
                  h = active_alloc.groupby([grp_col,"Component"])["Total_Alloc"].sum().reset_index()
                  hp = h.pivot_table(index=grp_col,columns="Component",values="Total_Alloc",fill_value=0)
                  # Sort columns by total descending, Others last
                  col_order = hp.sum().sort_values(ascending=False).index.tolist()
                  if "Others" in col_order:
                      col_order.remove("Others")
                      col_order.append("Others")
                  hp = hp[col_order]
                  txt = [[fmt(v) for v in row] for row in hp.values]
                  fh = px.imshow(hp.values,x=hp.columns.tolist(),y=hp.index.tolist(),color_continuous_scale=cscale,labels=dict(color="Cost"))
                  fh.update_traces(text=txt,texttemplate="%{text}")
                  fh.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(color=MUTED),
                      height=max(300, len(hp) * 40),margin=dict(l=110,r=10,t=20,b=40),
                      xaxis=dict(tickfont=dict(color=TEXT, size=9)),yaxis=dict(tickfont=dict(color=TEXT)))
                  st.plotly_chart(fh, use_container_width=True)

          st.subheader("Specificity Distribution")
          st.caption("Common = generic (consolidatable) · Shared = moderate · Specialized = keep separate")
          sd = active_alloc.groupby(["Component","Spec"]).agg(Cost=("Total_Alloc","sum"),Apps=("Asset_ID","nunique")).reset_index()
          sd["Specificity"] = sd["Spec"].map(SPEC_LABELS)

          if n_display > 8:
              fig_sp = px.bar(sd,y="Component",x="Cost",color="Specificity",barmode="group",orientation="h",
                  color_discrete_map={"Common":"#059669","Shared":"#ca8a04","Specialized":"#dc2626"},
                  hover_data={"Apps":True,"Cost":":,.0f"},
                  category_orders={"Specificity":["Common","Shared","Specialized"]})
              dark_layout(fig_sp, max(360, n_display * 28))
          else:
              fig_sp = px.bar(sd,x="Component",y="Cost",color="Specificity",barmode="group",
                  color_discrete_map={"Common":"#059669","Shared":"#ca8a04","Specialized":"#dc2626"},
                  hover_data={"Apps":True,"Cost":":,.0f"},
                  category_orders={"Specificity":["Common","Shared","Specialized"]})
              dark_layout(fig_sp, 340)
          fig_sp.update_layout(legend_title_text="Specificity")
          st.plotly_chart(fig_sp, use_container_width=True)
      else:
          st.warning(f"No {layer_label.lower()} component data available.")


  # ════════════════════════
  # ── Payment Management ──
  # ════════════════════════
  with tab_paymgmt:
      st.subheader("Payment Management")
      st.info("Payment management content coming soon.")

  # ════════════════════════
  # ── Early Pay ──
  # ════════════════════════
  with tab_epay:
      if len(vendor_summary)>0:
          ext_total = vendor_summary["Total_Cost"].sum()

          c1,c2,c3,c4 = st.columns(4)
          c1.metric("Gross Discount", fmt(ep_gross), f"{ep_gross/ext_total*100:.1f}% of external" if ext_total>0 else "")
          c2.metric("Cost of Cash", fmt(ep_coc), f"20 days early")
          c3.metric("Net Benefit", fmt(ep_net), f"{ep_preset} + {ep_fine_tune:+d}% tune")
          c4.metric("External Spend", fmt(ext_total), f"{vendor_summary['Assets'].sum()} assets")

          st.markdown("---")

          # Effective rates
          st.subheader("Effective Rates")
          rate_cols = st.columns(3)
          for i,(size,rates) in enumerate(ep_rates.items()):
              rate_cols[i].markdown(f"**{size}**")
              rate_cols[i].markdown(f"Discount: {rates['disc']:.1f}% · Accept: {rates['acc']:.0f}%")
              rate_cols[i].markdown(f"Effective: {rates['disc']*rates['acc']/100:.2f}%")
          st.caption("Discount rates adjusted by vendor margin tier: Low (<15%) ×0.7, Med (15–30%) ×1.0, High (>30%) ×1.5")

          st.markdown("---")

          # Vendor bar: gross vs cost of cash vs net
          st.subheader("Savings by Vendor")
          vs = vendor_summary[vendor_summary["Gross_Discount"] > 0].sort_values("Net_Benefit",ascending=True)
          fig_v = go.Figure()
          fig_v.add_trace(go.Bar(y=vs["Vendor"],x=vs["Gross_Discount"],name="Gross Discount",orientation="h",marker_color="rgba(52,211,153,0.7)"))
          fig_v.add_trace(go.Bar(y=vs["Vendor"],x=-vs["Cost_of_Cash"],name="Cost of Cash",orientation="h",marker_color="rgba(239,68,68,0.6)"))
          fig_v.update_layout(barmode="relative")
          dark_layout(fig_v,max(280,len(vs)*32))
          st.plotly_chart(fig_v, use_container_width=True)
          st.caption("Green = gross discount earned · Red = cost of deploying cash early · Net = difference")

          # By vendor size donut
          col1,col2 = st.columns(2)
          with col1:
              st.subheader("Net Benefit by Size")
              ss = vendor_summary.groupby("Estimated_Size")[["Gross_Discount","Cost_of_Cash","Net_Benefit"]].sum().reset_index()
              ss = ss[ss["Net_Benefit"]>0]
              if len(ss)>0:
                  fig_d = go.Figure(go.Pie(labels=ss["Estimated_Size"],values=ss["Net_Benefit"],hole=0.5,
                      marker_colors=[SIZE_COLORS.get(s,ACCENT) for s in ss["Estimated_Size"]],
                      textinfo="label+percent",textfont=dict(color=TEXT)))
                  fig_d.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=MUTED),height=300,margin=dict(l=20,r=20,t=20,b=20),showlegend=False)
                  st.plotly_chart(fig_d, use_container_width=True)

          with col2:
              st.subheader("Net Benefit by Year")
              if len(ep_yearly)>0:
                  yy = ep_yearly.groupby("year")[["Net"]].sum().reset_index()
                  yy = yy[(yy["year"]>=2020)&(yy["year"]<=2035)]
                  fig_nb = go.Figure()
                  fig_nb.add_trace(go.Bar(x=yy["year"], y=yy["Net"],
                      marker_color=["rgba(52,211,153,0.7)" if v >= 0 else "rgba(239,68,68,0.6)" for v in yy["Net"]],
                      text=yy["Net"].apply(fmt), textposition="outside"))
                  dark_layout(fig_nb, 300)
                  fig_nb.update_layout(yaxis_title="Net Benefit")
                  st.plotly_chart(fig_nb, use_container_width=True)
              else:
                  st.info("No yearly early payment data available.")

          st.markdown("---")
          with st.expander("Vendor Detail", expanded=False):
              vd = vendor_summary.sort_values("Net_Benefit",ascending=False).copy()
              for c in ["Total_Cost","Gross_Discount","Cost_of_Cash","Net_Benefit"]:
                  vd[c] = vd[c].apply(fmt)
              vd.columns = ["Vendor","Size","Assets","Total Cost","Gross Discount","Cost of Cash","Net Benefit"]
              st.dataframe(vd, use_container_width=True, hide_index=True)
      else:
          st.warning("Vendor_Extended sheet not found.")


  # ════════════════════════
  # ── Vendor Incentives ──
  # ════════════════════════
  with tab_incentives:
      st.subheader("Vendor Incentives")

      # ── A. Narrative Header ──
      _reb = INCENTIVE_SCENARIOS[vi_scenario]
      _prot = CHURN_SCENARIOS[vi_scenario]
      st.markdown(f"""
  **How this works:** Every vendor's gross margin is benchmarked against the portfolio's
  **TSM (Total Serviceable Market) average** — the weighted mean margin per layer. Vendors
  above the TSM have pricing *headroom* that translates into two negotiation levers:

  | Lever | Mechanism | Tiers ({vi_scenario}) |
  |-------|-----------|-------|
  | **Price Negotiation** | Vendors above TSM have inflated pricing. The headroom gap = a negotiable rebate applied to total spend. | Low {_reb['Low']:.0%} · Med {_reb['Medium']:.0%} · High {_reb['High']:.0%} |
  | **Prevented Price Increase** | Any vendor far from the TSM baseline faces **risk-sharing pressure**. Vendors *above* TSM face churn risk (we can switch to cheaper alternatives). Vendors *below* TSM are underperformers willing to share risk to retain the contract. Both accept **price protection** (long-term commitment locking current prices). Greater distance from TSM = more willingness to protect. | Low {_prot['Low']:.1%} · Med {_prot['Medium']:.1%} · High {_prot['High']:.0%} of annual spend |

  > **Net margin** is estimated as Gross Margin × 0.6 (assuming ~40% SGA overhead).
  > Prevented Price Increase uses **absolute distance** from TSM — vendors on either side contribute.
  """)

      inc_df = compute_vendor_incentives(df, vi_scenario)
      if not inc_df.empty:
          # ── B. Layer radio ──
          inc_layer = st.radio("Layer", ["Applications", "Infrastructure"], horizontal=True, key="inc_layer")
          ldf = inc_df[inc_df["Layer"] == inc_layer].copy()

          if not ldf.empty:
              tsm_val = ldf["TSM_Gross"].iloc[0]

              # ── C. KPI row (layer-specific) ──
              k1, k2, k3, k4 = st.columns(4)
              k1.metric("TSM Gross Margin", f"{tsm_val:.1f}%")
              k2.metric("Total Price Negotiation", fmt(ldf["Price_Negotiation"].sum()))
              k3.metric("Total Prevented Increase", fmt(ldf["Prevented_Increase"].sum()))
              k4.metric("Total Combined Incentive", fmt(ldf["Total_Incentive"].sum()))

              st.markdown("---")

              # ── D. 2×2 Quadrant Matrix ──
              st.markdown("### Headroom Quadrant Matrix")
              st.caption("Gross margin headroom (x) vs net margin headroom (y). Size = total spend. Quadrant = negotiation posture.")

              has_spread = (ldf["Gross_Headroom"].abs().sum() > 0) or (ldf["Net_Headroom"].abs().sum() > 0)
              if len(ldf) < 2 or not has_spread:
                  st.info(f"Not enough vendor margin spread for {inc_layer} to render the quadrant matrix. "
                          f"All {len(ldf)} vendor(s) sit at the TSM average — no headroom differentiation.")
              else:
                  fig_q = go.Figure()
                  for quad, qcolor in QUADRANT_COLORS.items():
                      qdf = ldf[ldf["Quadrant"] == quad]
                      if qdf.empty:
                          continue
                      max_spend = ldf["Total_Spend"].max()
                      sizes = qdf["Total_Spend"].apply(lambda v: max(10, min(50, v / (max_spend + 1) * 50)))
                      fig_q.add_trace(go.Scatter(
                          x=qdf["Gross_Headroom"], y=qdf["Net_Headroom"],
                          mode="markers+text", name=quad,
                          marker=dict(color=qcolor, size=sizes, opacity=0.8, line=dict(width=1, color="white")),
                          text=qdf["Vendor"], textposition="top center",
                          textfont=dict(size=9, color=MUTED)))

                  # Crosshair at 0,0
                  fig_q.add_hline(y=0, line_dash="dash", line_color=MUTED, line_width=1)
                  fig_q.add_vline(x=0, line_dash="dash", line_color=MUTED, line_width=1)

                  # Quadrant annotations
                  x_range = max(abs(ldf["Gross_Headroom"].min()), abs(ldf["Gross_Headroom"].max()), 5)
                  y_range = max(abs(ldf["Net_Headroom"].min()), abs(ldf["Net_Headroom"].max()), 5)
                  annotations = [
                      dict(x=x_range * 0.6, y=y_range * 0.8, text="Strong leverage<br>+ LTC candidate", showarrow=False, font=dict(color="#059669", size=10)),
                      dict(x=-x_range * 0.6, y=y_range * 0.8, text="Efficient ops,<br>limited price lever", showarrow=False, font=dict(color="#2563eb", size=10)),
                      dict(x=x_range * 0.6, y=-y_range * 0.8, text="Price lever exists,<br>thin ops", showarrow=False, font=dict(color="#ca8a04", size=10)),
                      dict(x=-x_range * 0.6, y=-y_range * 0.8, text="No headroom<br>— monitor", showarrow=False, font=dict(color="#94a3b8", size=10)),
                  ]
                  dark_layout(fig_q, 440)
                  fig_q.update_layout(
                      xaxis_title="Gross Margin Headroom (pp)",
                      yaxis_title="Net Margin Headroom (pp)",
                      xaxis_range=[-x_range * 1.15, x_range * 1.15],
                      yaxis_range=[-y_range * 1.15, y_range * 1.15],
                      annotations=annotations)
                  st.plotly_chart(fig_q, use_container_width=True)

              st.markdown("---")

              # ── E. Two-column lever bars ──
              col_a, col_b = st.columns(2)
              ldf_sorted = ldf.sort_values("Price_Negotiation", ascending=True)

              with col_a:
                  st.markdown("**Price Negotiation $ by Vendor**")
                  fig_neg = go.Figure()
                  for tier in ["High", "Medium", "Low", "None"]:
                      tdf = ldf_sorted[ldf_sorted["Incentive_Tier"] == tier]
                      if not tdf.empty:
                          fig_neg.add_trace(go.Bar(
                              y=tdf["Vendor"], x=tdf["Price_Negotiation"],
                              orientation="h", name=tier,
                              marker_color=TIER_COLORS[tier],
                              text=tdf["Price_Negotiation"].apply(fmt),
                              textposition="outside"))
                  dark_layout(fig_neg, 380)
                  fig_neg.update_layout(barmode="group", yaxis=dict(categoryorder="total ascending"),
                                        xaxis_title="Price Negotiation ($)")
                  st.plotly_chart(fig_neg, use_container_width=True)

              ldf_sorted2 = ldf.sort_values("Prevented_Increase", ascending=True)
              with col_b:
                  st.markdown("**Prevented Price Increase $ by Vendor**")
                  fig_prev = go.Figure()
                  for tier in ["High", "Medium", "Low", "None"]:
                      tdf = ldf_sorted2[ldf_sorted2["Churn_Tier"] == tier]
                      if not tdf.empty:
                          fig_prev.add_trace(go.Bar(
                              y=tdf["Vendor"], x=tdf["Prevented_Increase"],
                              orientation="h", name=tier,
                              marker_color=TIER_COLORS[tier],
                              text=tdf["Prevented_Increase"].apply(fmt),
                              textposition="outside"))
                  dark_layout(fig_prev, 380)
                  fig_prev.update_layout(barmode="group", yaxis=dict(categoryorder="total ascending"),
                                         xaxis_title="Prevented Increase ($)")
                  st.plotly_chart(fig_prev, use_container_width=True)

              # ── F. Detail table ──
              st.markdown("---")
              with st.expander("Vendor Detail", expanded=False):
                  tbl = ldf[["Vendor", "Gross_Margin", "TSM_Gross", "Gross_Headroom",
                              "Est_Net_Margin", "TSM_Net", "Net_Headroom",
                              "Incentive_Tier", "Rebate_Rate", "Churn_Tier", "Protection_Rate",
                              "Total_Spend", "Price_Negotiation", "Prevented_Increase", "Total_Incentive"]].copy()
                  tbl = tbl.sort_values("Total_Incentive", ascending=False)
                  for c in ["Total_Spend", "Price_Negotiation", "Prevented_Increase", "Total_Incentive"]:
                      tbl[c] = tbl[c].apply(fmt)
                  for c in ["Gross_Margin", "TSM_Gross", "Gross_Headroom", "Est_Net_Margin", "TSM_Net", "Net_Headroom"]:
                      tbl[c] = tbl[c].apply(lambda v: f"{v:.1f}%")
                  tbl["Rebate_Rate"] = tbl["Rebate_Rate"].apply(lambda v: f"{v*100:.0f}%")
                  tbl["Protection_Rate"] = tbl["Protection_Rate"].apply(lambda v: f"{v*100:.0f}%")
                  tbl.columns = ["Vendor", "Gross Margin", "TSM Gross", "Gross Headroom",
                                 "Est Net Margin", "TSM Net", "Net Headroom",
                                 "Incentive Tier", "Rebate Rate", "Churn Tier", "Protection Rate",
                                 "Total Spend", "Price Negotiation", "Prevented Increase", "Total Incentive"]
                  st.dataframe(tbl, use_container_width=True, hide_index=True)
          else:
              st.info(f"No external vendors with margin data found for {inc_layer}.")
      else:
          st.warning("Margin data not available — ensure Vendor_Extended sheet contains Gross_Margin_% column.")



  # ── Work Modernization (AI) ──
  # ════════════════════════
  with tab_aimod:
      if len(ai_reduction_app) == 0 and len(ai_reduction_infra) == 0:
          st.warning("AI_Reduction.xlsx not found or empty. Place the file in the same directory as the dashboard.")
      else:
          # Build historical + projected spend
          ai_hist_spend = by_year[["year", "total"]].copy()
          if show_proj:
              ai_pt, _, _ = gen_proj(by_year, dp, wp, dg, wg, depts, wts, avg_init, noise, proj_years, LAST_HIST_YEAR)
              ai_all_spend = pd.concat([ai_hist_spend, ai_pt[["year", "total"]]], ignore_index=True)
          else:
              ai_all_spend = ai_hist_spend.copy()

          # Merge AI gap by year
          ai_spend = ai_all_spend.merge(ai_gap_by_year, on="year", how="left")
          ai_spend["ai_gap"] = ai_spend["ai_gap"].fillna(0)
          ai_spend["ai_gap_app"] = ai_spend["ai_gap_app"].fillna(0)
          ai_spend["ai_gap_infra"] = ai_spend["ai_gap_infra"].fillna(0)

          # Extend AI gaps into projected years using last historical ratio
          ai_last_hist = ai_spend[ai_spend["year"] <= LAST_HIST_YEAR]
          if len(ai_last_hist) > 0:
              ai_last_row = ai_last_hist.iloc[-1]
              ai_last_total = ai_last_row["total"]
              ai_ratio = ai_last_row["ai_gap"] / ai_last_total if ai_last_total > 0 else 0
              ai_app_ratio = ai_last_row["ai_gap_app"] / ai_last_total if ai_last_total > 0 else 0
              ai_infra_ratio = ai_last_row["ai_gap_infra"] / ai_last_total if ai_last_total > 0 else 0
              ai_proj_mask = ai_spend["year"] > LAST_HIST_YEAR
              ai_spend.loc[ai_proj_mask, "ai_gap"] = ai_spend.loc[ai_proj_mask, "total"] * ai_ratio
              ai_spend.loc[ai_proj_mask, "ai_gap_app"] = ai_spend.loc[ai_proj_mask, "total"] * ai_app_ratio
              ai_spend.loc[ai_proj_mask, "ai_gap_infra"] = ai_spend.loc[ai_proj_mask, "total"] * ai_infra_ratio

          # Current Trend vs AI-Optimized (3yr ramp) vs AI-Optimal (full)
          ai_spend["current_trend"] = ai_spend["total"]
          ai_spend["ai_optimal"] = (ai_spend["total"] - ai_spend["ai_gap"]).clip(lower=0)

          def ai_optimized_ramp(row):
              yr = row["year"]
              if yr <= LAST_HIST_YEAR:
                  return 0
              offset = yr - LAST_HIST_YEAR
              if offset == 1:
                  return row["ai_gap"] * 0.33
              elif offset == 2:
                  return row["ai_gap"] * 0.66
              else:
                  return row["ai_gap"]
          ai_spend["ai_optimized"] = (ai_spend["total"] - ai_spend.apply(ai_optimized_ramp, axis=1)).clip(lower=0)

          # ── Metric cards ──
          total_ai_gap = ai_spend["ai_gap"].sum()
          proj_ai_gap = ai_spend[ai_spend["year"] > LAST_HIST_YEAR]["ai_gap"].sum() if show_proj else 0
          ai_pct = total_ai_gap / portfolio_total * 100 if portfolio_total > 0 else 0

          c1, c2, c3 = st.columns(3)
          c1.metric("Total Work Modernization Opportunity", fmt(total_ai_gap), f"{ai_scenario} scenario ({ai_scale:.0%})")
          c2.metric("Projected Work Modernization Opportunity", fmt(proj_ai_gap) if show_proj else "—",
                    f"Next {proj_years} years" if show_proj else "Enable projections")
          c3.metric("Work Modernization as % of Portfolio", f"{ai_pct:.1f}%", "Lifecycle cost base")

          st.markdown("---")

          # ── Chart 1: Spend with Work Modernization (line chart) ──
          st.subheader("Spend Comparison: Current Trend vs. Modernization-Optimized vs. Modernization-Optimal")
          st.caption("Current Trend = as-is · Modernization-Optimized = 3yr ramp (33%→66%→100%) · Modernization-Optimal = full work modernization opportunities applied")

          ai_plot = ai_spend[ai_spend["year"] >= 2020]

          fig_ai = go.Figure()
          fig_ai.add_trace(go.Scatter(x=ai_plot["year"], y=ai_plot["current_trend"], name="Current Trend",
              line=dict(color="#dc2626", width=2.5), mode="lines"))
          fig_ai.add_trace(go.Scatter(x=ai_plot["year"], y=ai_plot["ai_optimized"], name="AI-Optimized",
              line=dict(color="#0891b2", width=2.5, dash="dash"), mode="lines",
              fill="tonexty", fillcolor="rgba(8,145,178,0.12)"))
          fig_ai.add_trace(go.Scatter(x=ai_plot["year"], y=ai_plot["ai_optimal"], name="AI-Optimal",
              line=dict(color="#059669", width=2), mode="lines",
              fill="tonexty", fillcolor="rgba(5,150,105,0.12)"))

          if show_proj:
              fig_ai.add_vline(x=LAST_HIST_YEAR, line_dash="dash", line_color=MUTED,
                  annotation_text="Projected →", annotation_font_color=MUTED, annotation_font_size=10)
          dark_layout(fig_ai, 420)
          fig_ai.update_layout(yaxis_title="Annual Spend")
          st.plotly_chart(fig_ai, use_container_width=True)
          st.caption("Cyan fill = AI-optimized opportunity (3yr ramp) · Green fill = additional AI-optimal opportunity")

          st.markdown("---")

          # ── Side by side: Task Category (bar) + App vs Infra (donut) ──
          col_task, col_donut = st.columns(2)

          with col_task:
              st.markdown("**Opportunity by Task Category**")
              # Combine app + infra category gaps (scaled by scenario)
              all_cat_gap = {}
              for tc in ai_task_categories:
                  all_cat_gap[tc] = (ai_cat_gap_app.get(tc, 0) + ai_cat_gap_infra.get(tc, 0)) * ai_scale
              cat_df = pd.DataFrame([{"Task Category": k, "Reduction Factor": v} for k, v in all_cat_gap.items()])
              cat_df = cat_df.sort_values("Reduction Factor", ascending=True)

              fig_cat = go.Figure()
              fig_cat.add_trace(go.Bar(
                  y=cat_df["Task Category"], x=cat_df["Reduction Factor"],
                  orientation="h", marker_color="rgba(34,211,238,0.7)",
                  text=cat_df["Reduction Factor"].apply(lambda x: f"{x:.3f}"),
                  textposition="outside", textfont=dict(color=MUTED, size=10)))
              dark_layout(fig_cat, max(300, len(cat_df) * 35))
              fig_cat.update_layout(xaxis_title="Aggregate Reduction Factor", yaxis_title="")
              st.plotly_chart(fig_cat, use_container_width=True)

          with col_donut:
              st.markdown("**Opportunity: Application vs. Infrastructure**")
              total_app = ai_spend["ai_gap_app"].sum()
              total_infra = ai_spend["ai_gap_infra"].sum()
              if total_app > 0 or total_infra > 0:
                  fig_donut = go.Figure(go.Pie(
                      labels=["Application", "Infrastructure"],
                      values=[total_app, total_infra],
                      hole=0.5,
                      marker_colors=["rgba(34,211,238,0.8)", "rgba(59,130,246,0.8)"],
                      textinfo="label+percent",
                      textfont=dict(color=TEXT),
                      hovertemplate="%{label}<br>%{value:$,.0f}<br>%{percent}<extra></extra>"))
                  fig_donut.update_layout(
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=MUTED), height=350,
                      margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
                      annotations=[dict(text=fmt(total_app + total_infra), x=0.5, y=0.5,
                          font_size=16, font_color=TEXT, showarrow=False)])
                  st.plotly_chart(fig_donut, use_container_width=True)
              else:
                  st.info("No AI opportunity data to display.")

          st.markdown("---")

          # ── Table: Component-level detail (collapsed) ──
          with st.expander("Component-Level AI Reduction Factors", expanded=False):
              detail_rows = []
              for comp, factor in sorted(ai_reduction_app.items(), key=lambda x: -x[1]):
                  scaled = factor * ai_scale
                  comp_cost = 0
                  if alloc is not None and len(alloc) > 0:
                      comp_cost = alloc[alloc["Component"] == comp][["Run_Alloc", "Dev_Alloc"]].sum().sum()
                  detail_rows.append({"Layer": "Application", "Component": comp,
                                      "AI Factor": f"{scaled:.1%}", "Annual Opportunity": comp_cost * scaled})

              for comp, factor in sorted(ai_reduction_infra.items(), key=lambda x: -x[1]):
                  scaled = factor * ai_scale
                  comp_cost = 0
                  if infra_alloc is not None and len(infra_alloc) > 0:
                      match = infra_alloc[infra_alloc["Component"] == comp]
                      comp_cost = match[["Run_Alloc", "Dev_Alloc"]].sum().sum() if len(match) > 0 else 0
                  detail_rows.append({"Layer": "Infrastructure", "Component": comp,
                                      "AI Factor": f"{scaled:.1%}", "Annual Opportunity": comp_cost * scaled})

              if detail_rows:
                  detail_df = pd.DataFrame(detail_rows)
                  detail_df = detail_df.sort_values("Annual Opportunity", ascending=False)
                  detail_df["Annual Opportunity"] = detail_df["Annual Opportunity"].apply(fmt)
                  st.dataframe(detail_df, use_container_width=True, hide_index=True)
              else:
                  st.info("No AI reduction data available.")

  # ── Vendor Incentives (Benchmarks) ──
  with tab_bench_opp:
    st.subheader("Vendor Incentives — Benchmark Opportunities")
    st.caption("Identifies optimization opportunities by comparing actual org spend "
               "to external market benchmark rates per worktype.")

    bo_layer = st.radio("Layer", ["Applications", "Infrastructure"], horizontal=True, key="bo_layer")

    # ── Build opportunity table (market benchmarks only) ──
    _bo_rows = []

    if bo_layer == "Applications":
        _bo_df = df[df["Layer"] == "Applications"].copy()
        _bo_df["TotalAnnual"] = _bo_df["Annual_Run_Cost"] + _bo_df["Annual_Dev_Cost"]

        if not app_bench.empty:
            _bo_rng = np.random.default_rng(42)
            _bo_premium = _bo_rng.uniform(0.20, 0.30)

            # Per-worktype market stats
            _bo_bench_agg = app_bench.groupby("Worktype").agg(
                mkt_top_mid=("Midpoint", "min"),
                mkt_mean_mid=("Midpoint", "mean"),
                n_vendors=("Vendor", "nunique")).reset_index()
            _bo_bench_agg["org_kpi"] = _bo_bench_agg["mkt_mean_mid"] * (1 + _bo_premium)

            for wt in sorted(_bo_df["Worktype"].unique()):
                wt_assets = _bo_df[_bo_df["Worktype"] == wt]
                actual = wt_assets["TotalAnnual"].sum()
                n_assets = len(wt_assets)
                if actual == 0:
                    continue

                mkt_row = _bo_bench_agg[_bo_bench_agg["Worktype"] == wt]
                if len(mkt_row) == 0:
                    continue
                mr = mkt_row.iloc[0]

                # Implied volume from org KPI estimate
                vol = actual / mr["org_kpi"] if mr["org_kpi"] > 0 else 0
                at_avg = mr["mkt_mean_mid"] * vol
                at_top = mr["mkt_top_mid"] * vol
                opp_vs_avg = actual - at_avg
                opp_vs_top = actual - at_top

                # Opportunity % = dollar opportunity as % of actual spend (varies per wt)
                opp_pct_avg = (opp_vs_avg / actual * 100) if actual > 0 else 0
                opp_pct_top = (opp_vs_top / actual * 100) if actual > 0 else 0

                _bo_rows.append({
                    "Worktype": wt, "Assets": n_assets, "Vendors": int(mr["n_vendors"]),
                    "Actual Spend": actual,
                    "At Mkt Avg": at_avg, "Opp vs Avg": opp_vs_avg, "Opp % vs Avg": opp_pct_avg,
                    "At Mkt Top": at_top, "Opp vs Top": opp_vs_top, "Opp % vs Top": opp_pct_top,
                })
        else:
            st.info("No application market benchmark data available (Benchmark.xlsx).")

    else:  # Infrastructure
        _bo_df = df[df["Layer"] == "Infrastructure"].copy()
        _bo_df["TotalAnnual"] = _bo_df["Annual_Run_Cost"] + _bo_df["Annual_Dev_Cost"]

        if infra_bench:
            _bo_rng = np.random.default_rng(42)
            _bo_premium = _bo_rng.uniform(0.20, 0.30)

            for wt in sorted(_bo_df["Worktype"].unique()):
                # Only Access has numeric KPI
                if wt not in infra_bench or wt != "Access":
                    continue
                wt_assets = _bo_df[_bo_df["Worktype"] == wt]
                actual = wt_assets["TotalAnnual"].sum()
                n_assets = len(wt_assets)
                if actual == 0:
                    continue

                acc = infra_bench[wt].copy()
                acc_cols = list(acc.columns)
                if len(acc_cols) < 3:
                    continue
                acc["_min"] = pd.to_numeric(acc[acc_cols[1]], errors="coerce")
                acc["_max"] = pd.to_numeric(acc[acc_cols[2]], errors="coerce")
                acc_num = acc.dropna(subset=["_min", "_max"]).copy()
                if len(acc_num) == 0:
                    continue
                acc_num["_mid"] = (acc_num["_min"] + acc_num["_max"]) / 2
                i_mean = acc_num["_mid"].mean()
                i_top = acc_num["_mid"].min()
                i_org = i_mean * (1 + _bo_premium)
                n_vendors = len(acc_num)

                vol = actual / i_org if i_org > 0 else 0
                at_avg = i_mean * vol
                at_top = i_top * vol
                opp_vs_avg = actual - at_avg
                opp_vs_top = actual - at_top
                opp_pct_avg = (opp_vs_avg / actual * 100) if actual > 0 else 0
                opp_pct_top = (opp_vs_top / actual * 100) if actual > 0 else 0

                _bo_rows.append({
                    "Worktype": wt, "Assets": n_assets, "Vendors": n_vendors,
                    "Actual Spend": actual,
                    "At Mkt Avg": at_avg, "Opp vs Avg": opp_vs_avg, "Opp % vs Avg": opp_pct_avg,
                    "At Mkt Top": at_top, "Opp vs Top": opp_vs_top, "Opp % vs Top": opp_pct_top,
                })
        else:
            st.info("No infrastructure market benchmark data available (Benchmark.xlsx).")

    if _bo_rows:
        bo = pd.DataFrame(_bo_rows).sort_values("Opp vs Avg", ascending=False)

        # ── Summary cards ──
        _bo_total_actual = bo["Actual Spend"].sum()
        _bo_total_opp_avg = bo["Opp vs Avg"].sum()
        _bo_total_opp_top = bo["Opp vs Top"].sum()
        _bo_pct_avg = (_bo_total_opp_avg / _bo_total_actual * 100) if _bo_total_actual > 0 else 0
        _s1, _s2, _s3, _s4 = st.columns(4)
        _s1.metric("Total Actual Spend", fmt(_bo_total_actual))
        _s2.metric("Opportunity vs Mkt Avg", fmt(_bo_total_opp_avg),
                    delta=f"{_bo_pct_avg:.1f}%")
        _s3.metric("Opportunity vs Mkt Top", fmt(_bo_total_opp_top))
        _s4.metric("Worktypes with Data", len(bo))

        # ── Opportunity $ by worktype ──
        st.markdown("### Opportunity by Worktype")
        bo_chart = bo.sort_values("Opp vs Avg", ascending=True)
        fig_bo = go.Figure()
        fig_bo.add_trace(go.Bar(
            y=bo_chart["Worktype"],
            x=bo_chart["Opp vs Avg"],
            orientation="h", name="Opportunity vs Mkt Avg",
            marker_color="#ca8a04", opacity=0.85,
            text=[fmt(v) for v in bo_chart["Opp vs Avg"]],
            textposition="auto"))
        fig_bo.add_trace(go.Bar(
            y=bo_chart["Worktype"],
            x=bo_chart["Opp vs Top"],
            orientation="h", name="Opportunity vs Mkt Top",
            marker_color="#059669", opacity=0.85,
            text=[fmt(v) for v in bo_chart["Opp vs Top"]],
            textposition="auto"))
        dark_layout(fig_bo, max(300, len(bo_chart) * 55))
        fig_bo.update_layout(barmode="group", xaxis_title="Opportunity ($)",
                              margin=dict(l=150))
        st.plotly_chart(fig_bo, use_container_width=True)
        st.caption("Yellow = opportunity vs market average · Green = opportunity vs market top performer")

        # ── Opportunity % by worktype (this varies per worktype!) ──
        st.markdown("### Opportunity % of Actual Spend")
        st.caption("Percentage of current spend that could be recaptured at market rates. "
                   "Higher % = larger repricing opportunity for that worktype.")
        bo_pct = bo.sort_values("Opp % vs Avg", ascending=True)
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Bar(
            y=bo_pct["Worktype"],
            x=bo_pct["Opp % vs Avg"],
            orientation="h", name="vs Mkt Avg",
            marker_color="#ca8a04", opacity=0.85,
            text=[f"{v:.1f}%" for v in bo_pct["Opp % vs Avg"]],
            textposition="auto"))
        fig_pct.add_trace(go.Bar(
            y=bo_pct["Worktype"],
            x=bo_pct["Opp % vs Top"],
            orientation="h", name="vs Mkt Top",
            marker_color="#059669", opacity=0.85,
            text=[f"{v:.1f}%" for v in bo_pct["Opp % vs Top"]],
            textposition="auto"))
        dark_layout(fig_pct, max(280, len(bo_pct) * 50))
        fig_pct.update_layout(barmode="group", xaxis_title="Opportunity (% of Actual Spend)",
                               margin=dict(l=150))
        st.plotly_chart(fig_pct, use_container_width=True)
        st.caption("Yellow = % opportunity vs market avg · Green = % opportunity vs market top performer")

        # ── Detail table ──
        with st.expander("Opportunity Detail", expanded=False):
            bo_disp = bo[["Worktype", "Assets", "Vendors", "Actual Spend",
                          "At Mkt Avg", "Opp vs Avg", "Opp % vs Avg",
                          "At Mkt Top", "Opp vs Top", "Opp % vs Top"]].copy()
            for c in ["Actual Spend", "At Mkt Avg", "Opp vs Avg", "At Mkt Top", "Opp vs Top"]:
                bo_disp[c] = bo_disp[c].apply(lambda x: fmt(x) if pd.notna(x) else "—")
            bo_disp["Opp % vs Avg"] = bo_disp["Opp % vs Avg"].apply(lambda x: f"{x:.1f}%")
            bo_disp["Opp % vs Top"] = bo_disp["Opp % vs Top"].apply(lambda x: f"{x:.1f}%")
            bo_disp = bo_disp.rename(columns={
                "At Mkt Avg": "At Mkt Avg", "Opp vs Avg": "Opp $ vs Avg",
                "Opp % vs Avg": "Opp % Avg", "At Mkt Top": "At Mkt Top",
                "Opp vs Top": "Opp $ vs Top", "Opp % vs Top": "Opp % Top"})
            st.dataframe(bo_disp, use_container_width=True, hide_index=True)


# ════════════════════════
# TAB 4: ASSET ROI ANALYSIS
# ════════════════════════
with tab_bizcase:
    st.subheader("Asset ROI Analysis — Per-Asset")

    # Build per-asset consolidation gaps
    bc_layer = st.radio("Layer", ["Applications", "Infrastructure"], horizontal=True, key="bc_layer")
    bc_alloc = alloc if bc_layer == "Applications" else infra_alloc

    if bc_alloc is not None and len(bc_alloc) > 0:
        asset_info_bc = df.set_index("Asset_ID")[[
            "Initial_Year", "Estimated_Lifecycle_Years", "Initial_Cost",
            "Annual_Run_Cost", "Annual_Dev_Cost", "Vendor", "Deployment_Model"
        ]].to_dict("index")
        if "Estimated_Size" in df.columns:
            for aid, info in asset_info_bc.items():
                rows = df.loc[df["Asset_ID"] == aid]
                info["Estimated_Size"] = rows["Estimated_Size"].values[0] if len(rows) > 0 else "Small"
                if "Gross_Margin_%" in df.columns:
                    info["Gross_Margin_%"] = rows["Gross_Margin_%"].values[0] if len(rows) > 0 else None

        comp_names = sorted(bc_alloc["Component"].unique())
        asset_savings = {}
        for comp in comp_names:
            for spec in [1, 2, 3]:
                grp = bc_alloc[(bc_alloc["Component"] == comp) & (bc_alloc["Spec"] == spec)]
                if len(grp) == 0:
                    continue
                n_apps = len(grp)
                if spec == 1:
                    n_keep = grp["Worktype"].nunique()
                elif spec == 2:
                    n_keep = max(1, n_apps // 2) if scenario == "Conservative" else grp["Department"].nunique()
                else:
                    n_keep = n_apps
                n_eliminate = max(0, n_apps - min(n_keep, n_apps))
                if n_eliminate == 0:
                    continue
                ranked = grp.sort_values("Total_Alloc", ascending=True)
                eliminated = ranked.head(n_eliminate)
                for _, r in eliminated.iterrows():
                    aid = r["Asset_ID"]
                    if aid not in asset_savings:
                        asset_savings[aid] = {"Asset_ID": aid, "Department": r["Department"],
                            "Worktype": r["Worktype"], "Consol_Gap": 0, "Components": [],
                            "Component_Details": [],
                            "Init_Gap": 0, "RunDev_Gap": 0}
                    asset_savings[aid]["Consol_Gap"] += r["Total_Alloc"]
                    asset_savings[aid]["Init_Gap"] += r["Init_Alloc"]
                    asset_savings[aid]["RunDev_Gap"] += r["Run_Alloc"] + r["Dev_Alloc"]
                    asset_savings[aid]["Components"].append(f"{comp} ({SPEC_LABELS[spec]})")
                    asset_savings[aid]["Component_Details"].append({
                        "name": comp, "spec": SPEC_LABELS[spec],
                        "n_apps": n_apps, "n_keep": n_keep, "n_eliminate": n_eliminate,
                        "alloc": r["Total_Alloc"]})

        # Build full asset list (including those without consolidation gaps for early pay)
        all_asset_ids = sorted(df[df["Layer"] == bc_layer].Asset_ID.unique()) if bc_layer == "Applications" else sorted(df[df["Layer"] == "Infrastructure"].Asset_ID.unique())

        # Merge asset name if available
        name_col = [c for c in df.columns if "name" in c.lower() or "asset_name" in c.lower()]
        name_map = {}
        if name_col:
            name_map = df.set_index("Asset_ID")[name_col[0]].to_dict()

        # Build combined per-asset table
        bc_rows = []
        for aid in all_asset_ids:
            info = asset_info_bc.get(aid, {})
            sav = asset_savings.get(aid, {})
            total_cost = info.get("Initial_Cost", 0) + info.get("Annual_Run_Cost", 0) + info.get("Annual_Dev_Cost", 0)

            # Early pay for this asset
            ep_gap = 0
            if "Estimated_Size" in info and info.get("Deployment_Model") in ["Vendor", "Cloud"]:
                p = EARLY_PAY_PRESETS[ep_preset]
                multiplier = 1 + ep_fine_tune / 100
                size = info["Estimated_Size"]
                disc = p["disc"].get(size, 0) * multiplier / 100
                disc = disc * margin_multiplier(info.get("Gross_Margin_%", None))
                acc = min(p["accept"].get(size, 0) * multiplier, 100) / 100
                coc_rate = max(0.5, p["coc"] / multiplier) / 100
                gross = total_cost * disc * acc
                coc = total_cost * (20 / 365) * coc_rate * acc
                ep_gap = gross - coc

            # Vendor optimization for this asset
            vi_gap = 0
            vi_neg = 0
            vi_prev = 0
            vi_tier = "None"
            vi_churn_tier = "None"
            vi_rebate_rate = 0.0
            vi_protection_rate = 0.0
            if info.get("Deployment_Model") in ["Vendor", "Cloud"] and "Gross_Margin_%" in info:
                gm = info.get("Gross_Margin_%")
                if gm is not None and not pd.isna(gm):
                    layer_df = df[(df["Layer"] == bc_layer) & df["Deployment_Model"].isin(["Vendor", "Cloud"])].dropna(subset=["Gross_Margin_%"])
                    if not layer_df.empty:
                        tsm_gross = layer_df["Gross_Margin_%"].mean()
                        headroom = gm - tsm_gross
                        abs_headroom = abs(headroom)
                        if headroom > 0:
                            if headroom <= 5: vi_tier, vi_rebate_rate = "Low", INCENTIVE_SCENARIOS[vi_scenario]["Low"]
                            elif headroom <= 15: vi_tier, vi_rebate_rate = "Medium", INCENTIVE_SCENARIOS[vi_scenario]["Medium"]
                            else: vi_tier, vi_rebate_rate = "High", INCENTIVE_SCENARIOS[vi_scenario]["High"]
                        if abs_headroom > 0:
                            if abs_headroom <= 5: vi_churn_tier, vi_protection_rate = "Low", CHURN_SCENARIOS[vi_scenario]["Low"]
                            elif abs_headroom <= 15: vi_churn_tier, vi_protection_rate = "Medium", CHURN_SCENARIOS[vi_scenario]["Medium"]
                            else: vi_churn_tier, vi_protection_rate = "High", CHURN_SCENARIOS[vi_scenario]["High"]
                        vi_neg = vi_rebate_rate * total_cost if headroom > 0 else 0
                        annual_spend = info.get("Annual_Run_Cost", 0) + info.get("Annual_Dev_Cost", 0)
                        vi_prev = vi_protection_rate * annual_spend
                        vi_gap = vi_neg + vi_prev

            # AI Work Modernization gap for this asset
            ai_gap_asset = 0
            ai_annual_gap = 0
            ai_reduction_dict = ai_reduction_app if bc_layer == "Applications" else ai_reduction_infra
            asset_alloc_rows = bc_alloc[bc_alloc["Asset_ID"] == aid]
            for _, ar in asset_alloc_rows.iterrows():
                factor = ai_reduction_dict.get(ar["Component"], 0) * ai_scale
                ai_annual_gap += (ar["Run_Alloc"] + ar["Dev_Alloc"]) * factor
                ai_gap_asset += ar["Total_Alloc"] * factor

            consol_gap = sav.get("Consol_Gap", 0) * consol_scale
            if consol_gap == 0 and ep_gap <= 0 and vi_gap <= 0 and ai_gap_asset <= 0:
                continue

            life = int(info.get("Estimated_Lifecycle_Years", 1))
            annual_run_dev = info.get("Annual_Run_Cost", 0) + info.get("Annual_Dev_Cost", 0)
            annual_consol = sav.get("RunDev_Gap", 0) * consol_scale
            total_gap = consol_gap + ep_gap + vi_gap + ai_gap_asset
            bc_rows.append({
                "Asset_ID": aid,
                "Name": name_map.get(aid, aid),
                "Department": sav.get("Department", info.get("Department", "")),
                "Worktype": sav.get("Worktype", info.get("Worktype", "")),
                "Vendor": info.get("Vendor", ""),
                "Total_Cost": total_cost,
                "Consol_Gap": consol_gap,
                "EP_Gap": ep_gap,
                "VI_Gap": vi_gap,
                "VI_Neg": vi_neg,
                "VI_Prev": vi_prev,
                "VI_Tier": vi_tier,
                "VI_Churn_Tier": vi_churn_tier,
                "VI_Rebate_Rate": vi_rebate_rate,
                "VI_Protection_Rate": vi_protection_rate,
                "AI_Gap": ai_gap_asset,
                "AI_Annual_Gap": ai_annual_gap,
                "Total_Gap": total_gap,
                "Components": ", ".join(sorted(set(sav.get("Components", [])))),
                "Lifecycle_Yrs": life,
                "Annual_RunDev": annual_run_dev,
                "Annual_Consol_Gap": annual_consol,
            })

        if bc_rows:
            bc_df = pd.DataFrame(bc_rows).sort_values("Total_Gap", ascending=False)

            # Single select
            label_col = "Name" if name_col else "Asset_ID"
            asset_options = bc_df[label_col].tolist()
            selected = st.selectbox("Select an asset", asset_options, key="bc_asset_select")
            asset_row = bc_df[bc_df[label_col] == selected].iloc[0]

            aid = asset_row["Asset_ID"]
            info = asset_info_bc.get(aid, {})
            asset_alloc_rows = bc_alloc[bc_alloc["Asset_ID"] == aid]
            life = int(info.get("Estimated_Lifecycle_Years", 1))
            start_yr = int(info.get("Initial_Year", 2020))
            end_yr = start_yr + life - 1

            # ── Metrics ──
            # Top row: Total Baseline Cost + Total Gap
            tc1, tc2 = st.columns(2)
            tc1.metric("Total Baseline Cost", fmt(asset_row["Total_Cost"]),
                f"{asset_row['Worktype']} · {asset_row['Department']}")
            tot_pct = f"{asset_row['Total_Gap']/asset_row['Total_Cost']*100:.0f}% of total cost" if asset_row["Total_Cost"] > 0 else ""
            tc2.metric("Total Opportunity", fmt(asset_row["Total_Gap"]), tot_pct)

            # Individual gap breakdown
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Asset Management", fmt(asset_row["Consol_Gap"]),
                f"{asset_row['Consol_Gap']/asset_row['Total_Cost']*100:.0f}% of cost" if asset_row["Total_Cost"] > 0 else "")
            c2.metric("Work Modernization", fmt(asset_row["AI_Gap"]),
                f"{asset_row['AI_Gap']/asset_row['Total_Cost']*100:.1f}% of cost" if asset_row["Total_Cost"] > 0 else "")
            c3.metric("Vendor Incentives", fmt(asset_row["VI_Gap"]),
                f"{asset_row['VI_Gap']/asset_row['Total_Cost']*100:.1f}% of cost" if asset_row["Total_Cost"] > 0 else "")
            c4.metric("Early Pay", fmt(asset_row["EP_Gap"]),
                f"{asset_row['EP_Gap']/asset_row['Total_Cost']*100:.1f}% of cost" if asset_row["Total_Cost"] > 0 else "")

            st.markdown("---")

            # ── Opportunity Details (sub-tabs for readability) ──
            opp_tab_labels = []
            opp_tab_keys = []
            if asset_row["Consol_Gap"] > 0:
                opp_tab_labels.append(f"Asset Management — {fmt(asset_row['Consol_Gap'])}")
                opp_tab_keys.append("consol")
            if asset_row["AI_Gap"] > 0:
                opp_tab_labels.append(f"Work Modernization — {fmt(asset_row['AI_Gap'])}")
                opp_tab_keys.append("ai")
            if asset_row["VI_Gap"] > 0:
                opp_tab_labels.append(f"Vendor Incentives — {fmt(asset_row['VI_Gap'])}")
                opp_tab_keys.append("vi")
            if asset_row["EP_Gap"] != 0:
                opp_tab_labels.append(f"Early Pay — {fmt(asset_row['EP_Gap'])}")
                opp_tab_keys.append("ep")
            if not opp_tab_labels:
                opp_tab_labels.append("No Opportunities")
                opp_tab_keys.append("none")

            opp_tabs = st.tabs(opp_tab_labels)

            for ot_idx, ot_key in enumerate(opp_tab_keys):
                with opp_tabs[ot_idx]:
                    if ot_key == "consol":
                        st.markdown("#### Asset Management")
                        st.markdown(
                            f"Components in this asset overlap with other assets in the portfolio and "
                            f"can be consolidated under the **{scenario}** scenario.\n\n"
                            f"**Lifecycle opportunity:** {fmt(asset_row['Consol_Gap'])} | "
                            f"**Annual:** {fmt(asset_row['Annual_Consol_Gap'])}/yr over "
                            f"{life} yrs ({start_yr}–{end_yr})")
                        comp_details = asset_savings.get(aid, {}).get("Component_Details", [])
                        comp_sorted = sorted(comp_details, key=lambda x: x["alloc"], reverse=True)
                        if comp_sorted:
                            ct_rows = []
                            for cd in comp_sorted:
                                ct_rows.append({
                                    "Component": cd["name"],
                                    "Specificity": cd["spec"],
                                    "Portfolio Assets": cd["n_apps"],
                                    "Keep": cd["n_keep"],
                                    "Eliminate": cd["n_eliminate"],
                                    "Opportunity": fmt(cd["alloc"] * consol_scale),
                                })
                            st.dataframe(pd.DataFrame(ct_rows), use_container_width=True, hide_index=True)

                    elif ot_key == "ep":
                        st.markdown("#### Early Pay Discount")
                        if asset_row["EP_Gap"] > 0:
                            size = info.get("Estimated_Size", "")
                            p = EARLY_PAY_PRESETS[ep_preset]
                            multiplier = 1 + ep_fine_tune / 100
                            margin_pct = info.get("Gross_Margin_%", None)
                            m_mult = margin_multiplier(margin_pct)
                            margin_tier = "Low" if (not pd.isna(margin_pct) and margin_pct > 0 and margin_pct < 15) else "Med" if (not pd.isna(margin_pct) and 15 <= margin_pct <= 30) else "High" if (not pd.isna(margin_pct) and margin_pct > 30) else "N/A"
                            disc_pct = p["disc"].get(size, 0) * multiplier * m_mult
                            acc_pct = min(p["accept"].get(size, 0) * multiplier, 100)
                            coc_pct = max(0.5, p["coc"] / multiplier)
                            margin_display = f"{margin_pct:.1f}%" if (margin_pct is not None and not pd.isna(margin_pct)) else "N/A"
                            base_disc_pct = p["disc"].get(size, 0) * multiplier

                            ep_left, ep_right = st.columns([1, 1.5])
                            with ep_left:
                                st.markdown(
                                    f"**{info.get('Vendor', 'N/A')}** is a **{size}** vendor with "
                                    f"**{margin_display}** gross margin ({margin_tier} tier).\n\n"
                                    f"Strategy: **{ep_preset}**")
                            with ep_right:
                                ep_detail = pd.DataFrame([
                                    {"Parameter": "Base discount", "Value": f"{base_disc_pct:.1f}%"},
                                    {"Parameter": "Margin multiplier", "Value": f"x{m_mult}"},
                                    {"Parameter": "Effective discount", "Value": f"{disc_pct:.1f}%"},
                                    {"Parameter": "Acceptance rate", "Value": f"{acc_pct:.0f}%"},
                                    {"Parameter": "Cost of capital", "Value": f"{coc_pct:.1f}%"},
                                    {"Parameter": "Net opportunity", "Value": fmt(asset_row["EP_Gap"])},
                                ])
                                st.dataframe(ep_detail, use_container_width=True, hide_index=True)
                        else:
                            st.markdown(f"Early payment would cost more than the discount ({fmt(asset_row['EP_Gap'])}). Not recommended.")

                    elif ot_key == "vi":
                        st.markdown("#### Vendor Management — Optimization")
                        margin_pct = info.get("Gross_Margin_%", None)
                        margin_display = f"{margin_pct:.1f}%" if (margin_pct is not None and not pd.isna(margin_pct)) else "N/A"
                        st.markdown(
                            f"**{info.get('Vendor', 'N/A')}** has **{margin_display}** gross margin "
                            f"with headroom above TSM average.")
                        vi_detail = pd.DataFrame([
                            {"Source": "Price Negotiation", "Tier": asset_row["VI_Tier"],
                             "Rate": f"{asset_row['VI_Rebate_Rate']*100:.0f}%",
                             "Opportunity": fmt(asset_row["VI_Neg"])},
                            {"Source": "Churn Protection", "Tier": asset_row["VI_Churn_Tier"],
                             "Rate": f"{asset_row['VI_Protection_Rate']*100:.0f}%",
                             "Opportunity": fmt(asset_row["VI_Prev"])},
                            {"Source": "Combined", "Tier": "", "Rate": "",
                             "Opportunity": fmt(asset_row["VI_Gap"])},
                        ])
                        st.dataframe(vi_detail, use_container_width=True, hide_index=True)

                    elif ot_key == "ai":
                        st.markdown("#### Work Modernization — AI")
                        st.markdown(
                            f"**{ai_scenario}** scenario ({ai_scale:.0%} of potential). "
                            f"**Lifecycle opportunity:** {fmt(asset_row['AI_Gap'])} | "
                            f"**Annual:** {fmt(asset_row['AI_Annual_Gap'])}/yr")
                        ai_tbc = ai_task_by_comp_app if bc_layer == "Applications" else ai_task_by_comp_infra
                        task_totals = {}
                        for _, ar in asset_alloc_rows.iterrows():
                            comp_tasks = ai_tbc.get(ar["Component"], {})
                            for task, factor in comp_tasks.items():
                                scaled_f = factor * ai_scale
                                if scaled_f > 0:
                                    task_totals[task] = task_totals.get(task, 0) + ar["Total_Alloc"] * scaled_f
                        task_items = sorted(task_totals.items(), key=lambda x: x[1], reverse=True)
                        if task_items:
                            ai_rows = [{"Task": name, "Opportunity": fmt(tg)} for name, tg in task_items]
                            st.dataframe(pd.DataFrame(ai_rows), use_container_width=True, hide_index=True)

                    elif ot_key == "none":
                        st.info("No opportunities identified for this asset.")

            st.markdown("---")

            # ── ROI Analysis ──
            st.subheader("ROI Analysis")
            migration_cost = asset_row["Annual_RunDev"] * 0.5
            annual_vi = asset_row["VI_Gap"] / max(life, 1)
            annual_ai = asset_row["AI_Annual_Gap"]
            annual_saving = asset_row["Annual_Consol_Gap"] + (asset_row["EP_Gap"] / max(life, 1)) + annual_vi + annual_ai
            remaining_life = max(1, end_yr - LAST_HIST_YEAR)

            if annual_saving > 0 and migration_cost > 0:
                payback_years = migration_cost / annual_saving
                total_net_benefit = annual_saving * remaining_life - migration_cost
                roi_pct = (total_net_benefit / migration_cost * 100) if migration_cost > 0 else 0

                st.markdown(
                    f"Migration cost: {fmt(migration_cost)} (6 months of {fmt(asset_row['Annual_RunDev'])}/yr run+dev). "
                    f"Annual savings: {fmt(annual_saving)}/yr "
                    f"({fmt(asset_row['Annual_Consol_Gap'])}/yr asset management, "
                    f"{fmt(annual_ai)}/yr work modernization, "
                    f"{fmt(annual_vi)}/yr vendor incentives, "
                    f"{fmt(asset_row['EP_Gap'] / max(life, 1))}/yr early pay). "
                    f"Payback in {payback_years:.1f} years. "
                    f"Net benefit: {fmt(total_net_benefit)} over {remaining_life} remaining years "
                    f"({end_yr - remaining_life + 1}–{end_yr}), ROI: {roi_pct:.0f}%.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Est. Migration Cost", fmt(migration_cost), "~6 months run+dev")
                c2.metric("Annual Savings", fmt(annual_saving), "AM + WM + VI + EP")
                c3.metric("Payback Period", f"{payback_years:.1f} yr", "Time to break even")
                c4.metric("ROI", f"{roi_pct:.0f}%", f"Over {remaining_life} remaining yrs")

                # ROI waterfall chart
                roi_years = list(range(2026, 2026 + min(remaining_life, proj_years if show_proj else 5)))
                cumulative = []
                running = -migration_cost
                for y in roi_years:
                    running += annual_saving
                    cumulative.append({"year": y, "Cumulative": running})
                roi_df = pd.DataFrame(cumulative)

                fig_roi = go.Figure()
                colors_roi = ["rgba(239,68,68,0.7)" if v < 0 else "rgba(52,211,153,0.7)" for v in roi_df["Cumulative"]]
                fig_roi.add_trace(go.Bar(x=roi_df["year"], y=roi_df["Cumulative"],
                    marker_color=colors_roi, text=[fmt(v) for v in roi_df["Cumulative"]], textposition="outside"))
                fig_roi.add_hline(y=0, line_dash="dash", line_color=MUTED)
                dark_layout(fig_roi, 320)
                fig_roi.update_layout(yaxis_title="Cumulative Net Benefit")
                st.plotly_chart(fig_roi, use_container_width=True)
                st.caption("Migration cost estimated at 6 months of annual run+dev spend. "
                    "Annual savings = consolidation + early pay + vendor optimization + AI work modernization opportunities spread over lifecycle. "
                    "Red = pre-payback · Green = post-payback.")
            elif annual_saving > 0:
                st.markdown(f"**Annual savings: {fmt(annual_saving)}** — no migration cost estimated (no current run+dev spend).")
            else:
                st.info("No annual savings to compute ROI.")

            # ── Gap Breakdown by Component (Sankey) ──
            st.markdown("---")
            st.subheader("Opportunity Breakdown by Component")

            # Build Sankey: Total Gap -> Categories -> Components
            sankey_labels = ["Total Opportunity"]
            sankey_colors = ["rgba(30,41,59,0.8)"]
            sankey_src = []
            sankey_tgt = []
            sankey_val = []
            sankey_link_colors = []

            cat_idx = {}  # category name -> node index
            comp_idx = {}  # component key -> node index

            # Category nodes
            categories = [
                ("Asset Management", asset_row["Consol_Gap"], "rgba(59,130,246,0.7)"),
                ("Work Modernization", asset_row["AI_Gap"], "rgba(34,211,238,0.7)"),
                ("Vendor Incentives", asset_row["VI_Gap"], "rgba(168,85,247,0.7)"),
                ("Early Pay", asset_row["EP_Gap"], "rgba(52,211,153,0.7)"),
            ]

            for cat_name, cat_val, cat_color in categories:
                if cat_val > 0:
                    idx = len(sankey_labels)
                    sankey_labels.append(cat_name)
                    sankey_colors.append(cat_color)
                    cat_idx[cat_name] = idx
                    # Link from Total Gap to category
                    sankey_src.append(0)
                    sankey_tgt.append(idx)
                    sankey_val.append(cat_val)
                    sankey_link_colors.append(cat_color.replace("0.7", "0.35"))

            # Component detail nodes for Asset Management
            if asset_row["Consol_Gap"] > 0 and "Asset Management" in cat_idx:
                comp_details = asset_savings.get(aid, {}).get("Component_Details", [])
                comp_sorted = sorted(comp_details, key=lambda x: x["alloc"], reverse=True)
                for cd in comp_sorted:
                    key = f"consol_{cd['name']}"
                    idx = len(sankey_labels)
                    sankey_labels.append(cd["name"])
                    sankey_colors.append("rgba(59,130,246,0.5)")
                    comp_idx[key] = idx
                    sankey_src.append(cat_idx["Asset Management"])
                    sankey_tgt.append(idx)
                    sankey_val.append(cd["alloc"] * consol_scale)
                    sankey_link_colors.append("rgba(59,130,246,0.25)")

            # Component detail nodes for Work Modernization
            if asset_row["AI_Gap"] > 0 and "Work Modernization" in cat_idx:
                ai_rd = ai_reduction_app if bc_layer == "Applications" else ai_reduction_infra
                ai_items = []
                for _, ar in asset_alloc_rows.iterrows():
                    scaled_f = ai_rd.get(ar["Component"], 0) * ai_scale
                    if scaled_f > 0:
                        ai_items.append((ar["Component"], ar["Total_Alloc"] * scaled_f))
                ai_items.sort(key=lambda x: x[1], reverse=True)
                for comp_name, comp_gap in ai_items:
                    key = f"ai_{comp_name}"
                    idx = len(sankey_labels)
                    sankey_labels.append(comp_name)
                    sankey_colors.append("rgba(34,211,238,0.5)")
                    comp_idx[key] = idx
                    sankey_src.append(cat_idx["Work Modernization"])
                    sankey_tgt.append(idx)
                    sankey_val.append(comp_gap)
                    sankey_link_colors.append("rgba(34,211,238,0.25)")

            if sankey_val:
                fig_sankey = go.Figure(go.Sankey(
                    node=dict(
                        pad=15, thickness=20,
                        label=[f"{l}\n{fmt(asset_row['Total_Gap'])}" if i == 0 else l for i, l in enumerate(sankey_labels)],
                        color=sankey_colors,
                        line=dict(color="rgba(30,41,59,0.2)", width=0.5),
                    ),
                    link=dict(
                        source=sankey_src, target=sankey_tgt,
                        value=sankey_val, color=sankey_link_colors,
                    ),
                    textfont=dict(color="white", size=11),
                ))
                dark_layout(fig_sankey, 400)
                fig_sankey.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_sankey, use_container_width=True)

            # ── Detail ──
            st.markdown("---")
            with st.expander("Asset Detail", expanded=False):
                detail = {
                    "Field": ["Asset ID", "Name", "Department", "Worktype", "Vendor", "Lifecycle",
                              "Total Cost", "Asset Management Opportunity", "Work Modernization Opportunity", "Vendor Incentives Opportunity", "Early Pay Opportunity", "Total Opportunity", "Components Affected"],
                    "Value": [aid, name_map.get(aid, aid), asset_row["Department"], asset_row["Worktype"],
                              asset_row["Vendor"], f"{start_yr}–{end_yr} ({life} yrs)",
                              fmt(asset_row["Total_Cost"]), fmt(asset_row["Consol_Gap"]),
                              fmt(asset_row["AI_Gap"]), fmt(asset_row["VI_Gap"]),
                              fmt(asset_row["EP_Gap"]), fmt(asset_row["Total_Gap"]), asset_row["Components"]]
                }
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        else:
            st.info(f"No opportunities found for {bc_layer.lower()} assets under the {scenario} scenario.")
    else:
        st.warning(f"No {bc_layer.lower()} component allocation data available.")


# ════════════════════════════════════════════
# TAB: BUDGET VARIANCE PERFORMANCE
# ════════════════════════════════════════════
with tab_budvar:
    if var_df.empty:
        st.warning("No budget data available. Place Digital_Budget.xlsx in the app directory.")
    else:
        bv_c1, bv_c2 = st.columns(2)
        bv_layer = bv_c1.radio("Layer", ["All", "Applications", "Infrastructure"], horizontal=True, key="bv_layer")
        bv_view = bv_c2.radio("View by", ["Department", "Worktype"], horizontal=True, key="bv_view")

        vf = var_df.copy()
        if bv_layer != "All":
            vf = vf[vf["layer"] == bv_layer]

        if len(vf) > 0:
            group_col = "department" if bv_view == "Department" else "worktype"
            color_map = DEPT_COLORS if bv_view == "Department" else WT_COLORS

            # ── Portfolio weighted score ──
            portfolio_score = np.average(vf["score"], weights=vf["budget_total"].clip(lower=1))
            net_var_dollars = (vf["actual_total"] - vf["budget_total"]).sum()

            # Per-group weighted scores for best/worst
            grp_scores = vf.groupby(group_col).apply(
                lambda g: np.average(g["score"], weights=g["budget_total"].clip(lower=1))
            ).reset_index(name="score")
            best = grp_scores.loc[grp_scores["score"].idxmax()]
            worst = grp_scores.loc[grp_scores["score"].idxmin()]

            # ── Headline metrics ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Portfolio Score", f"{portfolio_score:.1f} / 10")
            m2.metric("Best Performer", f"{best[group_col]}", f"{best['score']:.1f}")
            m3.metric("Worst Performer", f"{worst[group_col]}", f"{worst['score']:.1f}")
            m4.metric("Net Variance $", fmt(net_var_dollars))

            # ── Heatmap: Group x Year ──
            st.subheader("Score Heatmap")
            heat = vf.groupby([group_col, "year"]).apply(
                lambda g: np.average(g["score"], weights=g["budget_total"].clip(lower=1))
            ).reset_index(name="score")
            heat_pivot = heat.pivot(index=group_col, columns="year", values="score").fillna(0)
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_pivot.values,
                x=[str(y) for y in heat_pivot.columns],
                y=heat_pivot.index.tolist(),
                colorscale="RdYlGn", zmin=0, zmax=10,
                text=np.round(heat_pivot.values, 1), texttemplate="%{text}",
                colorbar=dict(title="Score", tickfont=dict(color=MUTED)),
            ))
            fig_heat.update_layout(yaxis=dict(autorange="reversed"))
            dark_layout(fig_heat, height=max(300, len(heat_pivot) * 35 + 80))
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── Ranked bar chart ──
            st.subheader("Ranked Scores")
            ranked = grp_scores.sort_values("score", ascending=True)
            bar_colors = [color_map.get(g, ACCENT) for g in ranked[group_col]]
            fig_bar = go.Figure(go.Bar(
                x=ranked["score"], y=ranked[group_col], orientation="h",
                marker_color=bar_colors,
                text=[f"{s:.1f}" for s in ranked["score"]], textposition="outside",
            ))
            fig_bar.update_xaxes(range=[0, 10.5])
            dark_layout(fig_bar, height=max(300, len(ranked) * 35 + 80))
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Trend line chart ──
            st.subheader("Score Trend")
            yearly_portfolio = vf[vf["year"] >= 2020].groupby("year").apply(
                lambda g: np.average(g["score"], weights=g["budget_total"].clip(lower=1))
            ).reset_index(name="score")
            yearly_group = vf[vf["year"] >= 2020].groupby([group_col, "year"]).apply(
                lambda g: np.average(g["score"], weights=g["budget_total"].clip(lower=1))
            ).reset_index(name="score")

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=yearly_portfolio["year"], y=yearly_portfolio["score"],
                name="Portfolio", mode="lines+markers",
                line=dict(color=TEXT, width=3),
            ))
            for grp_name in sorted(yearly_group[group_col].unique()):
                gdata = yearly_group[yearly_group[group_col] == grp_name]
                fig_trend.add_trace(go.Scatter(
                    x=gdata["year"], y=gdata["score"],
                    name=grp_name, mode="lines+markers",
                    line=dict(color=color_map.get(grp_name, ACCENT), width=1.5, dash="dot"),
                ))
            fig_trend.update_yaxes(range=[0, 10.5])
            dark_layout(fig_trend, height=400)
            st.plotly_chart(fig_trend, use_container_width=True)

            # ── Detail table ──
            with st.expander("Variance Detail Table"):
                detail_df = vf[[group_col, "year", "actual_total", "budget_total", "variance_pct", "score"]].copy()
                detail_df.columns = [bv_view, "Year", "Actual", "Budget", "Variance %", "Score"]
                detail_df["Actual"] = detail_df["Actual"].apply(fmt)
                detail_df["Budget"] = detail_df["Budget"].apply(fmt)
                detail_df["Variance %"] = detail_df["Variance %"].round(1).astype(str) + "%"
                detail_df["Score"] = detail_df["Score"].round(1)
                detail_df = detail_df.sort_values(["Year", bv_view])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("No matching data for the selected layer.")


# ════════════════════════════════════════════
# TAB: PERFORMANCE
# ════════════════════════════════════════════
with tab_perf:
    if not perf_categories:
        st.warning("No market benchmark data available. Place Market_Benchmarks.xlsx in the app directory.")
    else:
        st.markdown(
            "<h3 style='margin-bottom:0'>Performance</h3>"
            "<p style='color:#94a3b8;font-size:0.95rem;margin-top:0.25rem'>"
            "<b>Performance</b> measures your organization's current procurement posture against market participants "
            "across key operational dimensions. It is a descriptive, backward-looking assessment — scores reflect "
            "where the organization stands today relative to peers.<br><br>"
            "A score of 0 represents the worst observed performance across participants; "
            "10 represents the maximum theoretical performance.</p>",
            unsafe_allow_html=True,
        )

        # Build data structures used by chart, table, and cards
        _perf_cats = list(perf_tier_scores.keys())
        _tier_rows = [
            ("Maximum Performance",              "max"),
            (f"Alpha Performer ({_alpha_label})", "alpha"),
            ("Top Performance",                   "top"),
            ("Benchmark Performance",             "avg"),
            ("Late Adopters",                     "bot"),
            ("Observed",                          "org"),
        ]

        def _dot_color(score):
            if score is None: return "#94a3b8"
            if score >= 7: return "#059669"
            if score >= 4: return "#ca8a04"
            return "#dc2626"

        _tier_colors = {
            "Maximum Performance": "#059669",
            "Alpha Performer": "#0284c7",
            "Top Performance": "#6d28d9",
            "Benchmark Performance": "#ca8a04",
            "Late Adopters": "#dc2626",
            "Observed": "#c026d3",
        }

        # ── 0) Summary metric cards ──
        _obs_avg = round(np.mean([perf_tier_scores[c].get("org", 0) or 0 for c in _perf_cats]), 1)
        _alpha_avg = round(np.mean([perf_tier_scores[c].get("alpha", 0) or 0 for c in _perf_cats]), 1)
        _gap_to_alpha = round(_alpha_avg - _obs_avg, 1)
        _cat_gaps = {c: round((perf_tier_scores[c].get("alpha", 0) or 0) - (perf_tier_scores[c].get("org", 0) or 0), 1) for c in _perf_cats}
        _riskiest_cat = max(_cat_gaps, key=_cat_gaps.get) if _cat_gaps else ""
        _riskiest_gap = _cat_gaps.get(_riskiest_cat, 0)
        _five_yr = spend_comp[(spend_comp["year"] >= LAST_HIST_YEAR+1) & (spend_comp["year"] <= LAST_HIST_YEAR+5)]
        _five_yr_savings = (_five_yr["current_trend"].sum() - _five_yr["optimal"].sum()) if len(_five_yr) > 0 else 0

        _perf_cards_ctr = st.container(border=True)
        mc1, mc2, mc3, mc4 = _perf_cards_ctr.columns(4)
        mc1.metric("Organization Score", f"{_obs_avg} / 10", "Avg across all dimensions")
        mc2.metric("Gap to Alpha", f"{_gap_to_alpha} pts", f"Alpha avg: {_alpha_avg}")
        mc3.metric("Highest Risk Area", _riskiest_cat, f"{_riskiest_gap} pts gap to alpha")
        mc4.metric("5-Year Savings Potential", fmt(_five_yr_savings), f"Baseline − Optimal ({LAST_HIST_YEAR+1}–{LAST_HIST_YEAR+5})")

        # ── 1) Line chart: tier scores across categories ──
        _perf_chart_ctr = st.container(border=True)
        _perf_chart_ctr.markdown(f"**Observed Score: {_obs_avg} avg**")
        fig_perf = go.Figure()
        for label, tier_key in _tier_rows:
            y_vals = []
            for cat in _perf_cats:
                s = 10.0 if tier_key == "max" else perf_tier_scores.get(cat, {}).get(tier_key)
                y_vals.append(s)
            is_benchmark = tier_key == "avg"
            fig_perf.add_trace(go.Scatter(
                x=_perf_cats, y=y_vals, name=label, mode="lines+markers",
                line=dict(color=_tier_colors.get(label, ACCENT), width=2,
                          dash="dash" if is_benchmark else "solid", shape="spline"),
                marker=dict(size=7),
            ))
        fig_perf.update_yaxes(range=[0, 10.5], dtick=2)
        dark_layout(fig_perf, height=400)
        fig_perf.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=10)))
        _perf_chart_ctr.plotly_chart(fig_perf, use_container_width=True)

        # ── 2) Scorecard table (sorted by avg score descending) ──
        _hdr = "".join(f"<th style='text-align:center; color:{MUTED}; font-size:12px; "
                       f"text-transform:uppercase; padding:8px 16px; border-bottom:1px solid {BORDER};'>{c}</th>"
                       for c in _perf_cats + ["Avg"])
        def _perf_tier_avg(tier_key):
            scores = [10.0 if tier_key == "max" else perf_tier_scores.get(c, {}).get(tier_key) for c in _perf_cats]
            valid = [s for s in scores if s is not None]
            return round(np.mean(valid), 1) if valid else 0
        _tier_sorted = sorted(_tier_rows, key=lambda t: _perf_tier_avg(t[1]), reverse=True)
        _html_rows = ""
        for label, tier_key in _tier_sorted:
            scores = []
            for cat in _perf_cats:
                s = 10.0 if tier_key == "max" else perf_tier_scores.get(cat, {}).get(tier_key)
                scores.append(s)
            avg_s = round(np.mean([s for s in scores if s is not None]), 1) if any(s is not None for s in scores) else None
            scores.append(avg_s)
            _base_label = label.split(" (")[0] if " (" in label else label
            dot_c = _tier_colors.get(_base_label, _dot_color(avg_s))
            label_html = (f"<td style='padding:10px 12px; border-bottom:1px solid {BORDER}; white-space:nowrap;'>"
                          f"<span style='color:{dot_c}; font-size:16px; margin-right:8px;'>&#9632;</span>"
                          f"<span style='color:#1e293b; font-size:13px;'>{label}</span></td>")
            cells = ""
            for s in scores:
                val = f"{s:.1f}" if s is not None else "—"
                c = _dot_color(s)
                cells += (f"<td style='text-align:center; padding:10px 16px; color:{c}; "
                          f"font-weight:600; font-size:14px; border-bottom:1px solid {BORDER};'>{val}</td>")
            _html_rows += f"<tr>{label_html}{cells}</tr>\n"

        _table_html = (
            f"<table style='width:100%; border-collapse:collapse; background:rgba(0,0,0,0);'>"
            f"<thead><tr>"
            f"<th style='text-align:left; color:{MUTED}; font-size:12px; text-transform:uppercase; "
            f"padding:8px 12px; border-bottom:1px solid {BORDER};'>Participant</th>"
            f"{_hdr}"
            f"</tr></thead>"
            f"<tbody>{_html_rows}</tbody>"
            f"</table>"
        )
        _perf_chart_ctr.markdown(_table_html, unsafe_allow_html=True)

        # ── 3) Alpha Performer differentiator cards ──
        _perf_alpha_scores = [r["alpha_score"] for rows in perf_categories.values() for r in rows if r.get("alpha_score")]
        _perf_detail = {cat: [{"KPI": r["KPI"], "org_fmt": r["org_fmt"], "score": r["score"],
                              "alpha": r["alpha"]} for r in rows]
                       for cat, rows in perf_categories.items()}
        _perf_cards_html = _alpha_cards_html(
            perf_categories, _alpha_label,
            min(_perf_alpha_scores) if _perf_alpha_scores else None,
            max(_perf_alpha_scores) if _perf_alpha_scores else None,
            detail_rows=_perf_detail)
        if _perf_cards_html:
            st.markdown(_perf_cards_html, unsafe_allow_html=True)

        # ── 4) Improvement Roadmap ──
        _perf_roadmap_html = _render_roadmap(perf_categories, perf_tier_scores)
        if _perf_roadmap_html:
            st.markdown(_perf_roadmap_html, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB: PREDICTION
# ════════════════════════════════════════════
with tab_pred:
    if not pred_categories:
        st.warning("No prediction benchmark data available. Place Prediction_Benchmarks.xlsx in the app directory.")
    else:
        st.markdown(
            "<h3 style='margin-bottom:0'>Prediction</h3>"
            "<p style='color:#94a3b8;font-size:0.95rem;margin-top:0.25rem'>"
            "<b>Prediction</b> assesses the organization's analytical maturity — its readiness to forecast spend "
            "trends, evaluate returns, and quantify risk. Scores reflect how well-equipped the organization is to "
            "generate forward-looking insights, not the forecasts themselves.<br><br>"
            "A score of 0 indicates no predictive analytics capability in place; "
            "10 represents full forecasting maturity across all measured dimensions.</p>",
            unsafe_allow_html=True,
        )

        # Build data structures
        _pred_cats = list(pred_tier_scores.keys())
        _pred_cat_labels = {c: c.replace("Retrun", "Return") for c in _pred_cats}  # auto-fix known typo
        _pred_display = [_pred_cat_labels.get(c, c) for c in _pred_cats]
        _pred_tier_rows = [
            ("Maximum Performance",                      "max"),
            (f"Alpha Performer ({_pred_alpha_label})",   "alpha"),
            ("Top Performance",                           "top"),
            ("Benchmark Performance",                     "avg"),
            ("Late Adopters",                             "bot"),
            ("Observed",                                  "org"),
        ]

        def _dot_color_pred(score):
            if score is None: return "#94a3b8"
            if score >= 7: return "#059669"
            if score >= 4: return "#ca8a04"
            return "#dc2626"

        _pred_tier_colors = {
            "Maximum Performance": "#059669",
            "Alpha Performer": "#0284c7",
            "Top Performance": "#6d28d9",
            "Benchmark Performance": "#ca8a04",
            "Late Adopters": "#dc2626",
            "Observed": "#c026d3",
        }

        # ── 0) Summary metric cards ──
        _pred_obs_avg = round(np.mean([pred_tier_scores[c].get("org", 0) or 0 for c in _pred_cats]), 1)
        _pred_alpha_avg = round(np.mean([pred_tier_scores[c].get("alpha", 0) or 0 for c in _pred_cats]), 1)
        _pred_gap = round(_pred_alpha_avg - _pred_obs_avg, 1)
        _pred_cat_gaps = {_pred_cat_labels.get(c, c): round((pred_tier_scores[c].get("alpha", 0) or 0) - (pred_tier_scores[c].get("org", 0) or 0), 1) for c in _pred_cats}
        _pred_riskiest = max(_pred_cat_gaps, key=_pred_cat_gaps.get) if _pred_cat_gaps else ""
        _pred_riskiest_gap = _pred_cat_gaps.get(_pred_riskiest, 0)
        _pred_five_yr = spend_comp[(spend_comp["year"] >= LAST_HIST_YEAR+1) & (spend_comp["year"] <= LAST_HIST_YEAR+5)]
        _pred_five_yr_sav = (_pred_five_yr["current_trend"].sum() - _pred_five_yr["optimal"].sum()) if len(_pred_five_yr) > 0 else 0

        _pred_cards_ctr = st.container(border=True)
        pmc1, pmc2, pmc3, pmc4 = _pred_cards_ctr.columns(4)
        pmc1.metric("Organization Score", f"{_pred_obs_avg} / 10", "Avg across all dimensions")
        pmc2.metric("Gap to Alpha", f"{_pred_gap} pts", f"Alpha avg: {_pred_alpha_avg}")
        pmc3.metric("Highest Risk Area", _pred_riskiest, f"{_pred_riskiest_gap} pts gap to alpha")
        pmc4.metric("5-Year Savings Potential", fmt(_pred_five_yr_sav), f"Baseline − Optimal ({LAST_HIST_YEAR+1}–{LAST_HIST_YEAR+5})")

        # ── 1) Line chart ──
        _pred_chart_ctr = st.container(border=True)
        _pred_chart_ctr.markdown(f"**Observed Score: {_pred_obs_avg} avg**")
        fig_pred = go.Figure()
        for label, tier_key in _pred_tier_rows:
            y_vals = []
            for cat in _pred_cats:
                s = 10.0 if tier_key == "max" else pred_tier_scores.get(cat, {}).get(tier_key)
                y_vals.append(s)
            is_benchmark = tier_key == "avg"
            fig_pred.add_trace(go.Scatter(
                x=_pred_display, y=y_vals, name=label, mode="lines+markers",
                line=dict(color=_pred_tier_colors.get(label, ACCENT), width=2,
                          dash="dash" if is_benchmark else "solid", shape="spline"),
                marker=dict(size=7),
            ))
        fig_pred.update_yaxes(range=[0, 10.5], dtick=2)
        dark_layout(fig_pred, height=400)
        fig_pred.update_layout(margin=dict(b=100), legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10)))
        _pred_chart_ctr.plotly_chart(fig_pred, use_container_width=True)

        # ── 2) Scorecard table ──
        _pred_hdr = "".join(f"<th style='text-align:center; color:{MUTED}; font-size:12px; "
                            f"text-transform:uppercase; padding:8px 16px; border-bottom:1px solid {BORDER};'>{c}</th>"
                            for c in _pred_display + ["Avg"])
        def _pred_tier_avg(tier_key):
            scores = [10.0 if tier_key == "max" else pred_tier_scores.get(c, {}).get(tier_key) for c in _pred_cats]
            valid = [s for s in scores if s is not None]
            return round(np.mean(valid), 1) if valid else 0
        _pred_tier_sorted = sorted(_pred_tier_rows, key=lambda t: _pred_tier_avg(t[1]), reverse=True)
        _pred_html_rows = ""
        for label, tier_key in _pred_tier_sorted:
            scores = []
            for cat in _pred_cats:
                s = 10.0 if tier_key == "max" else pred_tier_scores.get(cat, {}).get(tier_key)
                scores.append(s)
            avg_s = round(np.mean([s for s in scores if s is not None]), 1) if any(s is not None for s in scores) else None
            scores.append(avg_s)
            _base_label = label.split(" (")[0] if " (" in label else label
            dot_c = _pred_tier_colors.get(_base_label, _dot_color_pred(avg_s))
            label_html = (f"<td style='padding:10px 12px; border-bottom:1px solid {BORDER}; white-space:nowrap;'>"
                          f"<span style='color:{dot_c}; font-size:16px; margin-right:8px;'>&#9632;</span>"
                          f"<span style='color:#1e293b; font-size:13px;'>{label}</span></td>")
            cells = ""
            for s in scores:
                val = f"{s:.1f}" if s is not None else "—"
                c = _dot_color_pred(s)
                cells += (f"<td style='text-align:center; padding:10px 16px; color:{c}; "
                          f"font-weight:600; font-size:14px; border-bottom:1px solid {BORDER};'>{val}</td>")
            _pred_html_rows += f"<tr>{label_html}{cells}</tr>\n"

        _pred_table_html = (
            f"<table style='width:100%; border-collapse:collapse; background:rgba(0,0,0,0);'>"
            f"<thead><tr>"
            f"<th style='text-align:left; color:{MUTED}; font-size:12px; text-transform:uppercase; "
            f"padding:8px 12px; border-bottom:1px solid {BORDER};'>Participant</th>"
            f"{_pred_hdr}"
            f"</tr></thead>"
            f"<tbody>{_pred_html_rows}</tbody>"
            f"</table>"
        )
        _pred_chart_ctr.markdown(_pred_table_html, unsafe_allow_html=True)

        # ── 3) Alpha Performer differentiator cards ──
        _pred_alpha_scores = [r["alpha_score"] for rows in pred_categories.values() for r in rows if r.get("alpha_score")]
        _pred_detail = {cat: [{"KPI": r["KPI"], "org_fmt": r["org_fmt"], "score": r["score"],
                              "alpha": r["alpha"]} for r in rows]
                       for cat, rows in pred_categories.items()}
        _pred_cards_html = _alpha_cards_html(
            pred_categories, _pred_alpha_label,
            min(_pred_alpha_scores) if _pred_alpha_scores else None,
            max(_pred_alpha_scores) if _pred_alpha_scores else None,
            cat_labels=_pred_cat_labels,
            detail_rows=_pred_detail)
        if _pred_cards_html:
            st.markdown(_pred_cards_html, unsafe_allow_html=True)

        # ── 4) Improvement Roadmap ──
        _pred_roadmap_html = _render_roadmap(pred_categories, pred_tier_scores, cat_labels=_pred_cat_labels)
        if _pred_roadmap_html:
            st.markdown(_pred_roadmap_html, unsafe_allow_html=True)


# ── FOOTER ──
st.markdown(f"<div style='text-align:center; color:{BORDER}; font-size:10px; margin-top:40px;'>"
    f"Component Cost = Weight × Total Cost · "
    f"Asset Management: Common keep 1/worktype, Shared {'keep half' if scenario == 'Conservative' else '1/dept'}, Specialized keep all · "
    f"Work Modernization: {ai_scenario} ({ai_scale:.0%}) reduction factors from AI_Reduction.xlsx · "
    f"Vendor Incentives: {vi_scenario} · "
    f"Payment Management: {pm_scenario} ({pm_rate:.1%} of spend) · "
    f"Early Pay: {ep_preset} preset ({ep_fine_tune:+d}% tune) · Net = Gross Discount × Acceptance − Cost of Cash</div>",
    unsafe_allow_html=True)
