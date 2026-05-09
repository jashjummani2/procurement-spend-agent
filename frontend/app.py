"""Procurement Spend Analysis Agent — Streamlit frontend."""

import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(
    page_title="Procurement Spend Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.metric-card {
    background: #f8f9fa; border-radius: 8px; padding: 1rem;
    border-left: 4px solid #0066cc;
}
.risk-high { color: #dc3545; font-weight: bold; }
.risk-medium { color: #fd7e14; font-weight: bold; }
.risk-low { color: #28a745; font-weight: bold; }
.finding-card {
    background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
    padding: 1rem; margin-bottom: 0.75rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("session_id", None),
    ("upload_summary", None),
    ("result", None),
    ("status", "idle"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Spend Analysis Agent")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        """
1. Upload your spend transaction CSV (required)
2. Optionally add Supplier Master & Category Taxonomy
3. Add company context
4. Run the 4-step AI analysis
5. Review findings, anomalies & savings plan
"""
    )
    st.markdown("---")
    st.markdown("**Sample data** is in `sample_data/`")
    st.markdown("**Backend:** FastAPI + Claude Sonnet")

    if st.session_state.session_id:
        st.markdown("---")
        st.markdown(f"**Session:** `{st.session_state.session_id[:8]}…`")
        if st.button("🔄 Start New Analysis"):
            for k in ["session_id", "upload_summary", "result", "status"]:
                st.session_state[k] = None if k != "status" else "idle"
            st.rerun()


# ── Helper functions ──────────────────────────────────────────────────────────
def fmt_eur(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"€{value/1_000:.0f}K"
    return f"€{value:.0f}"


def severity_badge(sev: str) -> str:
    colours = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    return colours.get(sev, "⚪")


def effort_badge(effort: str) -> str:
    return {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}.get(effort, effort)


# ── Page: Upload ──────────────────────────────────────────────────────────────
def page_upload() -> None:
    st.header("1 · Upload Files")

    col1, col2 = st.columns([2, 1])
    with col1:
        tx_file = st.file_uploader(
            "Spend Transactions CSV **(required)**",
            type=["csv"],
            help="Must contain at minimum: date, supplier_name, amount, category columns.",
        )
        sup_file = st.file_uploader(
            "Supplier Master CSV _(optional)_",
            type=["csv"],
            help="Adds preferred-supplier and contract-status analysis.",
        )
        cat_file = st.file_uploader(
            "Category Taxonomy CSV _(optional)_",
            type=["csv"],
            help="Adds benchmark savings-potential percentages per category.",
        )

        company_ctx = st.text_area(
            "Company Context",
            placeholder="E.g. Mid-market industrial manufacturer, €150M revenue, preparing Q1 board review. Focus on indirect procurement.",
            height=80,
        )

    with col2:
        st.info(
            "**Required columns**\n\n"
            "- `date` (YYYY-MM-DD)\n"
            "- `supplier_name`\n"
            "- `amount` (EUR)\n"
            "- `category`\n\n"
            "**Recommended**\n\n"
            "- `invoice_number`\n"
            "- `department`\n"
            "- `payment_terms`\n"
            "- `sub_category`"
        )

    if tx_file and st.button("⬆️ Upload & Preview", type="primary"):
        with st.spinner("Uploading…"):
            files: dict[str, Any] = {"transactions": (tx_file.name, tx_file.getvalue(), "text/csv")}
            if sup_file:
                files["supplier_master"] = (sup_file.name, sup_file.getvalue(), "text/csv")
            if cat_file:
                files["category_taxonomy"] = (cat_file.name, cat_file.getvalue(), "text/csv")

            resp = requests.post(f"{API}/upload", files=files)
            if resp.status_code == 200:
                summary = resp.json()
                st.session_state.session_id = summary["session_id"]
                st.session_state.upload_summary = summary
                st.session_state.company_ctx = company_ctx
                st.session_state.status = "uploaded"
                st.rerun()
            else:
                st.error(f"Upload failed: {resp.text}")

    if st.session_state.upload_summary:
        _render_upload_preview()


def _render_upload_preview() -> None:
    s = st.session_state.upload_summary
    st.success("Files uploaded successfully!")
    st.markdown("### Data Preview")

    cols = st.columns(4)
    cols[0].metric("Transactions", f"{s['transactions_count']:,}")
    if "total_spend" in s:
        cols[1].metric("Total Spend", fmt_eur(s["total_spend"]))
    if "supplier_count" in s:
        cols[2].metric("Suppliers", s["supplier_count"])
    if "date_range" in s:
        cols[3].metric("Period", f"{s['date_range']['start'][:7]} → {s['date_range']['end'][:7]}")

    st.markdown(f"**Columns detected:** `{'` · `'.join(s['columns'])}`")

    company_ctx = st.session_state.get("company_ctx", "")
    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.spinner("Starting analysis…"):
            resp = requests.post(
                f"{API}/analyze/{st.session_state.session_id}",
                data={"company_context": company_ctx},
            )
            if resp.status_code == 200:
                st.session_state.status = "analyzing"
                st.rerun()
            else:
                st.error(f"Failed to start analysis: {resp.text}")


# ── Page: Analysis in progress ────────────────────────────────────────────────
def page_analyzing() -> None:
    st.header("2 · Analysis Running…")
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    for _ in range(180):  # poll for up to 3 minutes
        resp = requests.get(f"{API}/sessions/{st.session_state.session_id}")
        data = resp.json()

        with progress_placeholder.container():
            steps = data.get("progress", [])
            for i, msg in enumerate(steps):
                st.success(f"✅ {msg}")
            if len(steps) < 4:
                st.info(f"⏳ {['Initialising…', 'Computing statistics…', 'Running pattern analysis…', 'Detecting anomalies…'][len(steps)]}")

        if data["status"] == "completed":
            st.session_state.result = data["result"]
            st.session_state.status = "completed"
            st.rerun()
        elif data["status"] == "error":
            st.session_state.status = "error"
            st.error(f"Analysis error: {data.get('error', 'Unknown error')}")
            return

        time.sleep(3)

    st.error("Timed out waiting for analysis. Check the backend logs.")


# ── Page: Results ─────────────────────────────────────────────────────────────
def page_results() -> None:
    result = st.session_state.result
    if not result:
        st.warning("No result available.")
        return

    report: dict = result.get("report", {})
    stats: dict = result.get("statistics", {})
    anomalies: dict = result.get("anomalies", {})
    categorisation: dict = result.get("categorisation", {})

    st.header("📋 Spend Analysis Report")

    # ── KPI bar ──
    cols = st.columns(5)
    cols[0].metric("Total Spend", fmt_eur(stats.get("total_spend", 0)))
    cols[1].metric("Transactions", f"{stats.get('transaction_count', 0):,}")
    cols[2].metric("Suppliers", stats.get("supplier_count", "—"))
    cols[3].metric(
        "Addressable Savings",
        fmt_eur(report.get("total_addressable_savings_eur", 0)),
        delta=f"{report.get('total_savings_pct_of_spend', 0):.1f}% of spend",
    )
    risk = anomalies.get("overall_risk_score", "—")
    cols[4].metric("Risk Score", f"{risk}/10" if isinstance(risk, int) else "—")

    st.markdown("---")
    tabs = st.tabs(["📄 Executive Summary", "📊 Spend Breakdown", "⚠️ Anomalies", "💰 Savings Plan", "🗓️ Action Plan"])

    with tabs[0]:
        _tab_summary(report, categorisation)
    with tabs[1]:
        _tab_spend(stats)
    with tabs[2]:
        _tab_anomalies(anomalies)
    with tabs[3]:
        _tab_savings(report)
    with tabs[4]:
        _tab_actions(report, anomalies)


def _tab_summary(report: dict, categorisation: dict) -> None:
    st.subheader("Executive Summary")
    summary_text = report.get("executive_summary", "No summary available.")
    for para in summary_text.split("\n\n"):
        if para.strip():
            st.markdown(para.strip())

    st.markdown("---")
    st.subheader("Key Findings")
    findings = report.get("key_findings", [])
    for f in findings:
        sev = f.get("severity", "Medium")
        with st.expander(f"{severity_badge(sev)} {f.get('title', '')} — {fmt_eur(f.get('financial_impact_eur', 0))}"):
            st.markdown(f.get("description", ""))

    st.markdown("---")
    st.subheader("Quick Wins")
    qw = categorisation.get("quick_wins", [])
    for w in qw:
        st.markdown(
            f"**{w.get('title', '')}** · _{w.get('timeline', '')}_  \n"
            f"{w.get('description', '')}  \n"
            f"Estimated impact: **{fmt_eur(w.get('estimated_impact_eur', 0))}**"
        )
        st.markdown("---")


def _tab_spend(stats: dict) -> None:
    col1, col2 = st.columns(2)

    # Category donut
    cat_data = stats.get("by_category", {})
    if cat_data:
        df_cat = pd.DataFrame(
            [(k, v["sum"], v.get("pct_of_total", 0)) for k, v in cat_data.items()],
            columns=["Category", "Spend (€)", "% of Total"],
        ).sort_values("Spend (€)", ascending=False)

        with col1:
            st.subheader("Spend by Category")
            fig = px.pie(
                df_cat,
                names="Category",
                values="Spend (€)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                df_cat.style.format({"Spend (€)": "€{:,.0f}", "% of Total": "{:.1f}%"}),
                hide_index=True,
                use_container_width=True,
            )

    # Supplier bar
    sup_data = stats.get("by_supplier", {})
    if sup_data:
        df_sup = pd.DataFrame(
            [(k, v["sum"], v.get("pct_of_total", 0), v["count"]) for k, v in sup_data.items()],
            columns=["Supplier", "Spend (€)", "% of Total", "# Invoices"],
        ).sort_values("Spend (€)", ascending=False).head(10)

        with col2:
            st.subheader("Top 10 Suppliers")
            fig = px.bar(
                df_sup,
                x="Spend (€)",
                y="Supplier",
                orientation="h",
                color="% of Total",
                color_continuous_scale="Blues",
                text="Spend (€)",
            )
            fig.update_traces(texttemplate="€%{x:,.0f}", textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Department
    dept_data = stats.get("by_department", {})
    if dept_data:
        st.subheader("Spend by Department")
        df_dept = pd.DataFrame(
            list(dept_data.items()), columns=["Department", "Spend (€)"]
        ).sort_values("Spend (€)", ascending=False)
        fig = px.bar(
            df_dept,
            x="Department",
            y="Spend (€)",
            color="Spend (€)",
            color_continuous_scale="Teal",
            text="Spend (€)",
        )
        fig.update_traces(texttemplate="€%{y:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Payment terms
    pt_data = stats.get("payment_terms_distribution", {})
    if pt_data:
        st.subheader("Payment Terms Distribution")
        df_pt = pd.DataFrame(list(pt_data.items()), columns=["Terms", "Count"])
        fig = px.pie(df_pt, names="Terms", values="Count", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def _tab_anomalies(anomalies: dict) -> None:
    risk_score = anomalies.get("overall_risk_score", 0)
    colour = "🔴" if risk_score >= 7 else "🟡" if risk_score >= 4 else "🟢"
    st.subheader(f"Procurement Risk Score: {colour} {risk_score}/10")
    st.markdown(anomalies.get("overall_risk_score_rationale", ""))

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    def _risk_card(col, title: str, data: dict) -> None:
        with col:
            sev = data.get("severity", "Low")
            col.metric(
                title,
                fmt_eur(data.get("exposure_eur", data.get("total_value_eur", 0))),
                delta=sev,
                delta_color="inverse" if sev == "High" else "off",
            )
            st.markdown(data.get("description", ""))
            if "recommended_action" in data:
                st.info(f"**Action:** {data['recommended_action']}")

    _risk_card(col1, "Duplicate Invoices", anomalies.get("duplicate_invoice_risk", {}))
    _risk_card(col2, "Invoice Splitting", anomalies.get("split_invoice_risk", {}))
    _risk_card(col3, "Inactive Suppliers", anomalies.get("inactive_supplier_risk", {}))

    st.markdown("---")
    st.subheader("Immediate Actions Required")
    for i, action in enumerate(anomalies.get("immediate_actions", []), 1):
        st.markdown(f"**{i}.** {action}")

    outliers = anomalies.get("high_value_outliers", [])
    if outliers:
        st.markdown("---")
        st.subheader("High-Value Transaction Flags")
        for o in outliers:
            st.warning(str(o))


def _tab_savings(report: dict) -> None:
    total_savings = report.get("total_addressable_savings_eur", 0)
    pct = report.get("total_savings_pct_of_spend", 0)

    st.subheader(f"Total Addressable Savings: {fmt_eur(total_savings)} ({pct:.1f}% of spend)")

    opps = report.get("savings_opportunities", [])
    if not opps:
        st.info("No savings opportunities in result.")
        return

    # Waterfall chart
    fig = go.Figure(
        go.Waterfall(
            name="Savings",
            orientation="v",
            measure=["relative"] * len(opps) + ["total"],
            x=[o.get("initiative", f"Initiative {i+1}")[:25] for i, o in enumerate(opps)] + ["Total"],
            y=[o.get("estimated_savings_eur", 0) for o in opps] + [0],
            totals={"marker": {"color": "#0066cc"}},
            connector={"line": {"color": "#adb5bd"}},
        )
    )
    fig.update_layout(
        title="Savings Waterfall (€)",
        showlegend=False,
        margin=dict(t=40, b=40),
        yaxis_tickformat="€,.0f",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail cards
    st.markdown("---")
    for opp in sorted(opps, key=lambda o: o.get("estimated_savings_eur", 0), reverse=True):
        with st.expander(
            f"💰 {opp.get('initiative', '')} — {fmt_eur(opp.get('estimated_savings_eur', 0))} "
            f"({opp.get('savings_pct', 0):.1f}%)"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category", opp.get("category", "—"))
            c2.metric("Savings", fmt_eur(opp.get("estimated_savings_eur", 0)))
            c3.metric("Effort", effort_badge(opp.get("effort", "Medium")))
            c4.metric("Timeline", opp.get("timeline", "—"))

            actions = opp.get("actions", [])
            if actions:
                st.markdown("**Next steps:**")
                for a in actions:
                    st.markdown(f"- {a}")


def _tab_actions(report: dict, anomalies: dict) -> None:
    st.subheader("90-Day Action Plan")
    plan = report.get("next_90_day_plan", [])
    if plan:
        df_plan = pd.DataFrame(plan)
        st.dataframe(df_plan, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Risk Mitigation Controls")
    controls = report.get("risk_mitigation_plan", [])
    if controls:
        for ctrl in sorted(controls, key=lambda c: c.get("priority", 99)):
            prio = ctrl.get("priority", "—")
            st.markdown(f"**{prio}. {ctrl.get('control', '')}**  \n{ctrl.get('description', '')}")
            st.markdown("---")


# ── Router ────────────────────────────────────────────────────────────────────
status = st.session_state.status

if status == "idle" or status == "uploaded":
    page_upload()
elif status == "analyzing":
    page_analyzing()
elif status == "completed":
    page_results()
    with st.expander("🔁 Upload new files"):
        page_upload()
elif status == "error":
    st.error("Analysis failed. Please try again.")
    if st.button("Reset"):
        for k in ["session_id", "upload_summary", "result"]:
            st.session_state[k] = None
        st.session_state.status = "idle"
        st.rerun()
