from __future__ import annotations

import base64
import hashlib
import io
import json
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .schemas import AnalyzeRequest, Endpoint, Participant

ENGINE_VERSION = "engine_v1.0.0"
MIN_PER_PROTOCOL_COMPLIANCE = 75.0


def analyze_request(req: AnalyzeRequest) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    endpoint_results = [_endpoint_result(endpoint, req.participants) for endpoint in req.endpoints]
    raw_wilcoxon_p_values = [r["wilcoxon_p"] for r in endpoint_results if r["wilcoxon_p"] is not None]
    if raw_wilcoxon_p_values:
        adjusted = iter(multipletests(raw_wilcoxon_p_values, method="bonferroni")[1].tolist())
        for result in endpoint_results:
            result["bonferroni_p"] = float(next(adjusted)) if result["wilcoxon_p"] is not None else None
    else:
        for result in endpoint_results:
            result["bonferroni_p"] = None
    raw_paired_t_p_values = [r["paired_t_p"] for r in endpoint_results if r["paired_t_p"] is not None]
    if raw_paired_t_p_values:
        adjusted = iter(multipletests(raw_paired_t_p_values, method="fdr_bh")[1].tolist())
        for result in endpoint_results:
            result["bh_fdr_p"] = float(next(adjusted)) if result["paired_t_p"] is not None else None
    else:
        for result in endpoint_results:
            result["bh_fdr_p"] = None

    charts = {
        "waterfall": [_waterfall_chart(endpoint, req.participants) for endpoint in req.endpoints],
        "time_series": [_time_series_chart(endpoint, req.participants) for endpoint in req.endpoints],
    }
    report_without_pdf = {
        "engine_version": ENGINE_VERSION,
        "generated_at": generated_at,
        "trial": req.trial.model_dump(),
        "endpoint_results": endpoint_results,
        "dropout_sensitivity": _dropout_sensitivity(req.endpoints, req.participants),
        "subgroup_analysis": _subgroup_analysis(req.endpoints, req.participants),
        "charts": charts,
        "methodology": _methodology(req),
    }
    report_hash = hashlib.sha256(
        json.dumps(report_without_pdf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **report_without_pdf,
        "report_hash": report_hash,
        "pdf_base64": base64.b64encode(_render_pdf(report_without_pdf, report_hash)).decode("ascii"),
    }


def _endpoint_result(endpoint: Endpoint, participants: Sequence[Participant]) -> Dict[str, Any]:
    rows = _participant_deltas(endpoint, participants, per_protocol=False)
    pp_rows = _participant_deltas(endpoint, participants, per_protocol=True)
    deltas = np.array([row["oriented_delta"] for row in rows], dtype=float)
    pp_deltas = np.array([row["oriented_delta"] for row in pp_rows], dtype=float)
    baseline_values = np.array([row["baseline_mean"] for row in rows], dtype=float)
    intervention_values = np.array([row["intervention_mean"] for row in rows], dtype=float)
    ci_low, ci_high = _ci95(deltas)
    wilcoxon_p = _wilcoxon_p(deltas)
    threshold = 0.0 if endpoint.responder_threshold is None else float(endpoint.responder_threshold)
    responders = int(np.sum(deltas > threshold)) if deltas.size else 0
    return {
        "endpoint_key": endpoint.key,
        "label": endpoint.label,
        "role": endpoint.role,
        "direction": endpoint.direction,
        "n_itt": int(deltas.size),
        "n_per_protocol": int(pp_deltas.size),
        "baseline_mean": _mean_or_none(baseline_values),
        "baseline_sd": _sd_or_none(baseline_values),
        "intervention_mean": _mean_or_none(intervention_values),
        "intervention_sd": _sd_or_none(intervention_values),
        "itt_mean_delta": _round_or_none(float(np.mean(deltas))) if deltas.size else None,
        "itt_delta_sd": _sd_or_none(deltas),
        "per_protocol_mean_delta": _round_or_none(float(np.mean(pp_deltas))) if pp_deltas.size else None,
        "ci95": {"low": ci_low, "high": ci_high},
        "cohen_d": _cohen_dz(deltas),
        "paired_t_p": _paired_t_p(baseline_values, intervention_values),
        "bh_fdr_p": None,
        "wilcoxon_p": wilcoxon_p,
        "bonferroni_p": None,
        "responder_rate": _round_or_none(responders / deltas.size) if deltas.size else None,
        "responders": responders,
        "waterfall_points": [
            {"participant_id": row["participant_id"], "delta": _round_or_none(row["oriented_delta"])}
            for row in sorted(rows, key=lambda item: item["oriented_delta"])
        ],
    }


def _participant_deltas(endpoint: Endpoint, participants: Sequence[Participant], per_protocol: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for participant in participants:
        if per_protocol and not _is_per_protocol(participant):
            continue
        data = participant.endpoints.get(endpoint.key)
        if data is None or not data.baseline_values or not data.intervention_values:
            continue
        baseline = _safe_mean(data.baseline_values)
        intervention = _safe_mean(data.intervention_values)
        if baseline is None or intervention is None:
            continue
        out.append(
            {
                "participant_id": participant.participant_id,
                "baseline_mean": baseline,
                "intervention_mean": intervention,
                "oriented_delta": _oriented_delta(endpoint, baseline, intervention),
                "subgroups": participant.subgroups,
            }
        )
    return out


def _is_per_protocol(participant: Participant) -> bool:
    if participant.status not in {"completed", "complete"}:
        return False
    pct = participant.verified_compliance_pct
    return pct is None or float(pct) >= MIN_PER_PROTOCOL_COMPLIANCE


def _oriented_delta(endpoint: Endpoint, baseline: float, intervention: float) -> float:
    if endpoint.direction == "lower_better":
        return baseline - intervention
    if endpoint.direction == "optimal_range":
        return _distance_from_range(baseline, endpoint.optimal_min, endpoint.optimal_max) - _distance_from_range(
            intervention, endpoint.optimal_min, endpoint.optimal_max
        )
    return intervention - baseline


def _distance_from_range(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is None or high is None or low > high:
        return 0.0
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(numeric)) if numeric else None


def _ci95(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if values.size == 0:
        return None, None
    if values.size == 1:
        v = _round_or_none(float(values[0]))
        return v, v
    sem = stats.sem(values)
    margin = stats.t.ppf(0.975, values.size - 1) * sem
    mean = float(np.mean(values))
    return _round_or_none(mean - margin), _round_or_none(mean + margin)


def _wilcoxon_p(values: np.ndarray) -> Optional[float]:
    if values.size < 2 or np.allclose(values, 0):
        return None
    try:
        return _round_or_none(float(stats.wilcoxon(values, zero_method="wilcox").pvalue), digits=6)
    except ValueError:
        return None


def _paired_t_p(baseline: np.ndarray, intervention: np.ndarray) -> Optional[float]:
    if baseline.size < 2 or intervention.size < 2:
        return None
    try:
        pvalue = float(stats.ttest_rel(intervention, baseline, nan_policy="omit").pvalue)
    except ValueError:
        return None
    return _round_or_none(pvalue, digits=6)


def _cohen_dz(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    sd = float(np.std(values, ddof=1))
    return None if sd == 0.0 else _round_or_none(float(np.mean(values)) / sd)


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    return _round_or_none(float(np.mean(values))) if values.size else None


def _sd_or_none(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    return _round_or_none(float(np.std(values, ddof=1)))


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _waterfall_chart(endpoint: Endpoint, participants: Sequence[Participant]) -> Dict[str, Any]:
    rows = sorted(_participant_deltas(endpoint, participants, per_protocol=False), key=lambda item: item["oriented_delta"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    values = [row["oriented_delta"] for row in rows]
    ax.bar(range(len(values)), values, color=["#b94a48" if value < 0 else "#2f7d59" for value in values])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{endpoint.label} responder waterfall")
    ax.set_ylabel("Preferred-direction delta")
    fig.tight_layout()
    return {"endpoint_key": endpoint.key, "image_base64": _fig_to_base64(fig)}


def _time_series_chart(endpoint: Endpoint, participants: Sequence[Participant]) -> Dict[str, Any]:
    series_rows: List[Dict[str, Any]] = []
    for participant in participants:
        data = participant.endpoints.get(endpoint.key)
        if data is None:
            continue
        for idx, value in enumerate(data.daily_values):
            if value is not None:
                series_rows.append({"day": idx + 1, "value": float(value)})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if series_rows:
        frame = pd.DataFrame(series_rows)
        summary = frame.groupby("day")["value"].agg(["mean", "count", "std"]).reset_index()
        summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"].clip(lower=1))
        summary["ci"] = 1.96 * summary["sem"]
        ax.plot(summary["day"], summary["mean"], color="#24515f", linewidth=2)
        ax.fill_between(summary["day"], summary["mean"] - summary["ci"], summary["mean"] + summary["ci"], color="#8fb6c1", alpha=0.35)
    ax.set_title(f"{endpoint.label} cohort time series")
    ax.set_xlabel("Trial day")
    fig.tight_layout()
    return {"endpoint_key": endpoint.key, "image_base64": _fig_to_base64(fig)}


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _subgroup_analysis(endpoints: Sequence[Endpoint], participants: Sequence[Participant]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        for row in _participant_deltas(endpoint, participants, per_protocol=False):
            for name, value in row["subgroups"].items():
                rows.append({"endpoint_key": endpoint.key, "subgroup": name, "value": value, "delta": row["oriented_delta"]})
    if not rows:
        return {"available": False, "groups": []}
    frame = pd.DataFrame(rows)
    groups = []
    for (endpoint_key, subgroup, value), group in frame.groupby(["endpoint_key", "subgroup", "value"]):
        if len(group) >= 2:
            groups.append(
                {
                    "endpoint_key": endpoint_key,
                    "subgroup": subgroup,
                    "value": value,
                    "n": int(len(group)),
                    "mean_delta": _round_or_none(float(group["delta"].mean())),
                    "ci95": dict(zip(["low", "high"], _ci95(group["delta"].to_numpy(dtype=float)), strict=True)),
                }
            )
    return {"available": bool(groups), "groups": groups}


def _dropout_sensitivity(endpoints: Sequence[Endpoint], participants: Sequence[Participant]) -> Dict[str, Any]:
    results = []
    dropout_count = sum(1 for participant in participants if participant.status not in {"completed", "complete"})
    for endpoint in endpoints:
        itt = np.array([row["oriented_delta"] for row in _participant_deltas(endpoint, participants, False)], dtype=float)
        pp = np.array([row["oriented_delta"] for row in _participant_deltas(endpoint, participants, True)], dtype=float)
        results.append(
            {
                "endpoint_key": endpoint.key,
                "dropout_count": int(dropout_count),
                "itt_mean_delta": _round_or_none(float(np.mean(itt))) if itt.size else None,
                "per_protocol_mean_delta": _round_or_none(float(np.mean(pp))) if pp.size else None,
                "delta_gap": _round_or_none(float(np.mean(pp) - np.mean(itt))) if itt.size and pp.size else None,
                "worst_case_dropout_delta": 0.0 if dropout_count else None,
            }
        )
    return {"items": results}


def _methodology(req: AnalyzeRequest) -> Dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "analysis_sets": {
            "itt": "All enrolled participants with baseline and intervention data for the endpoint.",
            "per_protocol": f"Completed participants with verified compliance at or above {MIN_PER_PROTOCOL_COMPLIANCE:.0f} percent.",
        },
        "statistics": [
            "Participant-level baseline and intervention means are compared as paired deltas.",
            "Cohen's dz is computed from paired deltas.",
            "Paired t-test p values are adjusted with Benjamini-Hochberg false-discovery-rate correction.",
            "Wilcoxon signed-rank p values are Bonferroni-adjusted across endpoints.",
            "Responder rate is the share of participants above the endpoint responder threshold.",
            "Time-series confidence bands use normal 95 percent intervals around daily cohort means.",
        ],
        "concurrent_interventions_count": len(req.concurrent_interventions),
        "adverse_events_count": len(req.adverse_events),
    }


def _render_pdf(report: Dict[str, Any], report_hash: str) -> bytes:
    buf = io.BytesIO()
    sns.set_theme(style="whitegrid")
    with PdfPages(buf) as pdf:
        trial = report["trial"]
        _pdf_text_page(
            pdf,
            "Outliyr Sponsored Trial Cohort Report",
            [
                f"Trial: {trial.get('title', '')}",
                f"Sponsor: {trial.get('sponsor_name', '')}",
                f"Engine: {report['engine_version']}",
                f"Generated: {report['generated_at']}",
                f"Protocol Hash: {trial.get('protocol_hash') or 'Not Recorded'}",
                f"Report Hash: {report_hash}",
            ],
            font_size=12,
        )
        _pdf_text_page(pdf, "Executive Summary", _executive_summary_lines(report))
        _pdf_text_page(pdf, "Endpoint Results", _endpoint_result_lines(report))
        _pdf_text_page(pdf, "Methodology", _methodology_lines(report))
        _pdf_text_page(pdf, "Safety And Compliance Notes", _safety_lines(report))
        _pdf_text_page(pdf, "Limitations And Citation", _limitations_lines(report))
        for chart_group in ("waterfall", "time_series"):
            for chart in report["charts"][chart_group]:
                fig, ax = plt.subplots(figsize=(8.5, 5))
                ax.axis("off")
                ax.imshow(plt.imread(io.BytesIO(base64.b64decode(chart["image_base64"])), format="png"))
                pdf.savefig(fig)
                plt.close(fig)
    return buf.getvalue()


def _pdf_text_page(pdf: PdfPages, title: str, lines: Sequence[str], font_size: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    wrapped = [title, ""]
    for line in lines:
        if line == "":
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=92) or [""])
    ax.text(0.06, 0.95, "\n".join(wrapped), va="top", ha="left", fontsize=font_size, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _executive_summary_lines(report: Dict[str, Any]) -> List[str]:
    endpoints = report["endpoint_results"]
    primary = next((item for item in endpoints if item.get("role") == "primary"), endpoints[0] if endpoints else None)
    if not primary:
        return ["No eligible endpoint data was available for this report."]
    return [
        f"Primary Endpoint: {primary['label']}",
        f"Cohort Size: {primary['n_itt']} ITT participants; {primary['n_per_protocol']} per-protocol participants.",
        f"Mean Preferred-Direction Delta: {primary['itt_mean_delta']} with 95% CI {primary['ci95']['low']} to {primary['ci95']['high']}.",
        f"Effect Size: Cohen's dz {primary['cohen_d']}.",
        f"Paired T-Test P Value: {primary['paired_t_p']}; BH-FDR P Value: {primary['bh_fdr_p']}.",
        f"Responder Count: {primary['responders']} participants; responder rate {primary['responder_rate']}.",
        "Interpretation: Positive deltas indicate movement in the endpoint's preferred direction.",
    ]


def _endpoint_result_lines(report: Dict[str, Any]) -> List[str]:
    if not report["endpoint_results"]:
        return ["No endpoint rows were available."]
    lines = []
    for item in report["endpoint_results"]:
        lines.extend(
            [
                f"{item['label']} ({item['role'].title()})",
                f"Baseline Mean: {item['baseline_mean']}; Intervention Mean: {item['intervention_mean']}.",
                f"ITT Mean Delta: {item['itt_mean_delta']}; Delta SD: {item['itt_delta_sd']}.",
                f"95% CI: {item['ci95']['low']} to {item['ci95']['high']}.",
                f"Cohen's dz: {item['cohen_d']}; Paired T-Test P: {item['paired_t_p']}; BH-FDR P: {item['bh_fdr_p']}.",
                f"Wilcoxon P: {item['wilcoxon_p']}; Bonferroni P: {item['bonferroni_p']}.",
                "",
            ]
        )
    return lines


def _methodology_lines(report: Dict[str, Any]) -> List[str]:
    methodology = report["methodology"]
    analysis_sets = methodology.get("analysis_sets", {})
    lines = [
        f"ITT Set: {analysis_sets.get('itt', '')}",
        f"Per-Protocol Set: {analysis_sets.get('per_protocol', '')}",
        "",
        "Statistical Methods:",
    ]
    lines.extend([f"- {item}" for item in methodology.get("statistics", [])])
    return lines


def _safety_lines(report: Dict[str, Any]) -> List[str]:
    methodology = report["methodology"]
    dropout_items = report.get("dropout_sensitivity", {}).get("items", [])
    lines = [
        f"Concurrent Interventions Count: {methodology.get('concurrent_interventions_count', 0)}.",
        f"Adverse Events Count: {methodology.get('adverse_events_count', 0)}.",
        "",
        "Dropout Sensitivity:",
    ]
    if not dropout_items:
        lines.append("No dropout sensitivity rows were available.")
        return lines
    for item in dropout_items:
        lines.append(
            f"{item['endpoint_key']}: Dropouts {item['dropout_count']}; ITT Delta {item['itt_mean_delta']}; "
            f"Per-Protocol Delta {item['per_protocol_mean_delta']}; Gap {item['delta_gap']}."
        )
    return lines


def _limitations_lines(report: Dict[str, Any]) -> List[str]:
    trial = report["trial"]
    return [
        "This report summarizes an open-label sponsored product trial. It is not a randomized, blinded, placebo-controlled clinical trial.",
        "Results may be affected by selection bias, adherence differences, concurrent interventions, device measurement error, "
        "and regression to the mean.",
        "Sponsor claims should describe this as Outliyr cohort evidence unless a separate regulatory or IRB-reviewed study supports "
        "stronger language.",
        "",
        "Suggested Citation:",
        f"Outliyr. {trial.get('title', 'Sponsored Trial')} Cohort Report. Engine {report['engine_version']}. "
        f"Generated {report['generated_at']}.",
    ]
