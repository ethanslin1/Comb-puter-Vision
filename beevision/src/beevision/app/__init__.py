"""Streamlit dashboard for BeeVision colony health reports.

Public, streamlit-free helpers (safe to import from anywhere)::

    from beevision.app import (
        load_report_json,
        generate_demo_report,
        score_bucket,
    )

The dashboard page itself lives at ``beevision.app.dashboard`` and
imports ``streamlit``; only import that module when you intend to launch
the UI (``streamlit run``).
"""
from beevision.app.components import (
    SCORE_BUCKETS,
    ScoreBucket,
    class_breakdown_chart_data,
    confusion_matrix_text,
    food_breakdown_chart_data,
    format_count,
    format_pct,
    format_ratio,
    score_bucket,
    score_components_chart_data,
    severity_for_note,
)
from beevision.app.data import (
    FRAME_CANDIDATES,
    REPORT_FILENAME,
    find_frame_image,
    generate_demo_report,
    list_reports,
    load_report_json,
    make_demo_health_inputs,
    report_to_json,
    resolve_report_path,
    save_demo_report,
)

__all__ = [
    # data
    "load_report_json",
    "save_demo_report",
    "generate_demo_report",
    "make_demo_health_inputs",
    "report_to_json",
    "resolve_report_path",
    "find_frame_image",
    "list_reports",
    "REPORT_FILENAME",
    "FRAME_CANDIDATES",
    # components
    "score_bucket",
    "SCORE_BUCKETS",
    "ScoreBucket",
    "format_count",
    "format_pct",
    "format_ratio",
    "class_breakdown_chart_data",
    "food_breakdown_chart_data",
    "score_components_chart_data",
    "confusion_matrix_text",
    "severity_for_note",
]
