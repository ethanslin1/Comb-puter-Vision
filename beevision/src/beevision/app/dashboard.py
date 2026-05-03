"""BeeVision dashboard — Streamlit entry point.

Run with::

    streamlit run -m beevision.app.dashboard
    # or via the project Makefile:
    make app

Reads HealthReport JSON files produced by the pipeline and renders an
interactive dashboard. With no real reports available, switch to
``Demo`` mode in the sidebar to view a synthetic ``HealthReport`` built
from in-memory cell/bee predictions — the math is real, only the upstream
inputs are fake.

Layout
------
- **Sidebar**: source selector (Demo / file / directory), demo profile
  switcher, environment summary.
- **Header**: large composite score with color-coded badge, frame thumb
  if a sibling ``frame.jpg`` exists.
- **Tabs**: Overview / Brood / Composition / Mites / Notes & raw JSON.

The Streamlit dependency is loaded only here; ``app.data`` and
``app.components`` are streamlit-free so unit tests don't need it.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import streamlit as st  # type: ignore[import-not-found]

from beevision.app.components import (
    class_breakdown_chart_data,
    food_breakdown_chart_data,
    format_count,
    format_pct,
    format_ratio,
    score_bucket,
    score_components_chart_data,
    severity_for_note,
)
from beevision.app.data import (
    find_frame_image,
    generate_demo_report,
    list_reports,
    load_report_json,
    report_to_json,
)


# ---------- Page config ----------------------------------------------------


st.set_page_config(
    page_title="BeeVision — Colony Health Dashboard",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Source selection ---------------------------------------------


def _default_reports_root() -> Path:
    """Default place to look for real pipeline reports."""
    root = os.environ.get("BEEVISION_ROOT", f"/oscar/scratch/{os.environ.get('USER', '')}/beevision")
    return Path(root) / "processed" / "reports"


def _sidebar_source() -> dict:
    """Render the sidebar; return a dict describing the chosen source."""
    st.sidebar.header("Data source")
    mode = st.sidebar.radio(
        "Mode",
        options=["Demo (synthetic)", "From disk"],
        index=0,
        help="Demo builds a synthetic HealthReport so the dashboard is usable "
             "without a real pipeline run.",
    )

    if mode == "Demo (synthetic)":
        quality = st.sidebar.selectbox(
            "Demo profile",
            options=["healthy", "struggling", "failing"],
            index=0,
        )
        seed = int(st.sidebar.number_input("Seed", min_value=0, value=1337, step=1))
        return {"mode": "demo", "quality": quality, "seed": seed}

    # From disk: pick a directory or single JSON.
    default_root = str(_default_reports_root())
    root = st.sidebar.text_input("Reports root", value=default_root)
    available = list_reports(root) if Path(root).exists() else []
    if available:
        names = [p.name for p in available]
        choice = st.sidebar.selectbox("Report", options=names)
        chosen = next(p for p in available if p.name == choice)
        return {"mode": "disk", "report_dir": str(chosen)}
    st.sidebar.warning(f"No reports under {root!r}. Switch to Demo mode or run the pipeline.")
    return {"mode": "disk", "report_dir": ""}


# ---------- Loading -------------------------------------------------------


def _load_report(source: dict) -> tuple[dict, Path | None]:
    if source["mode"] == "demo":
        report = generate_demo_report(quality=source["quality"], seed=source["seed"])
        return report_to_json(report), None
    if not source.get("report_dir"):
        st.stop()
    rd = Path(source["report_dir"])
    return load_report_json(rd), find_frame_image(rd)


# ---------- Header --------------------------------------------------------


def _render_header(report: dict, frame_path: Path | None) -> None:
    score = float(report["composite_score"])
    bucket = score_bucket(score)

    cols = st.columns([1, 2])
    with cols[0]:
        if frame_path is not None:
            st.image(str(frame_path), caption=f"Frame: {frame_path.name}", use_column_width=True)
        else:
            st.markdown(
                "<div style='border:1px dashed #ccc; height:120px; "
                "display:flex; align-items:center; justify-content:center; "
                "color:#888;'>no frame.jpg</div>",
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.markdown(
            f"<div style='display:flex; align-items:baseline; gap:12px;'>"
            f"<span style='font-size:64px; font-weight:700; color:{bucket.color};'>{score:.1f}</span>"
            f"<span style='font-size:14px; color:#666;'>/ 10</span>"
            f"<span style='padding:4px 10px; border-radius:6px; "
            f"background:{bucket.color}; color:white; font-weight:600; "
            f"text-transform:uppercase; letter-spacing:0.05em;'>{bucket.label}</span>"
            f"</div>"
            f"<div style='color:#444; margin-top:6px;'>"
            f"BeeVision colony health composite (1 = failing, 10 = thriving).</div>",
            unsafe_allow_html=True,
        )

        comp = report.get("score_components", {})
        weights = report.get("weights", {})
        sub = st.columns(3)
        sub[0].metric(
            "Brood health",
            format_pct(comp.get("brood_health"), decimals=0),
            help=f"weight {weights.get('brood', 0):.2f}",
        )
        sub[1].metric(
            "Food balance",
            format_pct(comp.get("food_balance"), decimals=0),
            help=f"weight {weights.get('food', 0):.2f}",
        )
        sub[2].metric(
            "Mite safety",
            format_pct(comp.get("mite_safety"), decimals=0),
            help=f"weight {weights.get('mites', 0):.2f}",
        )


# ---------- Tabs ---------------------------------------------------------


def _tab_overview(report: dict) -> None:
    st.subheader("Cell composition")
    comp_rows = class_breakdown_chart_data(report)
    st.bar_chart(
        {row["category"]: row["count"] for row in comp_rows},
        height=200,
    )

    st.subheader("Score axes")
    rows = score_components_chart_data(report)
    st.bar_chart(
        {row["axis"]: row["weighted"] for row in rows},
        height=200,
    )
    st.caption(
        "Bars are weighted contributions to the [0, 1] raw score "
        "(score = 1 + 9 × Σ weighted)."
    )


def _tab_brood(report: dict) -> None:
    b = report["brood"]
    cols = st.columns(4)
    cols[0].metric("Brood cells", format_count(b["n_brood_cells"]))
    cols[1].metric("Brood fraction", format_pct(b["brood_fraction"]))
    cols[2].metric("Capped fraction", format_pct(b["capped_brood_fraction"]))
    cols[3].metric("Regularity", format_pct(b["regularity"]))

    cols = st.columns(2)
    cols[0].metric("Fill ratio", format_pct(b["fill_ratio"]))
    cols[1].metric("Spacing CV", f"{b['nearest_neighbor_cv']:.2f}",
                   help="coefficient of variation of nearest-neighbor distances; "
                        "low = uniform spacing, high = clustered")

    if b["n_brood_cells"] == 0:
        st.warning("No brood cells detected on this frame.")


def _tab_composition(report: dict) -> None:
    c = report["composition"]
    cols = st.columns(4)
    cols[0].metric("Food cells", format_count(c["n_food_cells"]))
    cols[1].metric("Food fraction", format_pct(c["food_fraction"]))
    cols[2].metric("Honey:pollen", format_ratio(c["honey_pollen_ratio"]))
    cols[3].metric("Food balance", format_pct(c["food_balance"]),
                   help="peaked at 70% honey-fraction (winter-prep target)")

    rows = food_breakdown_chart_data(report)
    st.bar_chart(
        {row["food_class"]: row["count"] for row in rows},
        height=200,
    )


def _tab_mites(report: dict) -> None:
    m = report["mites"]
    cols = st.columns(4)
    cols[0].metric("Bees", format_count(m["n_bees"]))
    cols[1].metric("Mite hits", format_count(m["n_mite"]))
    cols[2].metric("Hard load", format_pct(m["mite_load"]))
    cols[3].metric("Soft load", format_pct(m["mite_load_weighted"]),
                   help="probability-weighted; more robust than the argmax count")
    if m["n_bees"] == 0:
        st.warning("No bees detected on this frame; mite load is uninformative.")


def _tab_notes(report: dict) -> None:
    notes = report.get("notes", [])
    if not notes:
        st.success("No notes from the metrics module.")
    else:
        for n in notes:
            sev = severity_for_note(n)
            (st.warning if sev == "warning" else st.info)(n)

    st.subheader("Raw report JSON")
    st.code(json.dumps(report, indent=2), language="json")


# ---------- Main ---------------------------------------------------------


def main() -> None:  # pragma: no cover - thin Streamlit driver
    st.title("BeeVision — Colony Health Dashboard")

    source = _sidebar_source()
    report, frame_path = _load_report(source)

    _render_header(report, frame_path)
    st.divider()

    overview, brood, comp, mites, notes = st.tabs(
        ["Overview", "Brood", "Composition", "Mites", "Notes & JSON"]
    )
    with overview:
        _tab_overview(report)
    with brood:
        _tab_brood(report)
    with comp:
        _tab_composition(report)
    with mites:
        _tab_mites(report)
    with notes:
        _tab_notes(report)

    st.caption("BeeVision · CSCI 1430 final project")


# Streamlit runs this script top-to-bottom on every rerun, so call main()
# at module level. Tests should NOT import this module — they exercise
# ``app.data`` and ``app.components`` instead, both of which are
# streamlit-free.
main()  # pragma: no cover
