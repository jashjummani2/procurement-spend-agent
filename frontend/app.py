"""Procurement Spend Analysis Agent — Streamlit app (self-contained, no FastAPI)."""

import io
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent import SpendAnalysisAgent

st.set_page_config(
    page_title="Procurement Spend Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.metric-card { background:#f8f9fa; border-radius:8px; padding:1rem; border-left:4px solid #0066cc; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("data", None), ("result", None), ("upload_summary", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Spend Analysis Agent")
    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Upload your spend CSV\n"
        "2. Optionally add Supplier Master & Taxonomy\n"
        "3. Add company context\n"
        "4. Run the 4-step AI analysis\n"
        "5. Review findings & savings plan"
    )
    st.markdown("---")
    st.markdown("**Sample data** is in `sample_data/`")
    if st.session_state.result and st.button("🔄 Start New Analysis"):
        st.session_state.data = None
        st.session_state.result = None
        st.session_state.upload_summary = None
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_eur(v) -> str:
    try:
        v = float(str(v).replace("€", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return str(v)
    if abs(v) >= 1_000_000:
        return f"€{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"€{v/1_000:.0f}K"
    return f"€{v:.0f}"


def sev_icon(s: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(s, "⚪")


def effort_label(e: str) -> str:
    return {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}.get(e, e)


def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(file.read().decode("utf-8")))


# ── Upload section ────────────────────────────────────────────────────────────
if not st.session_state.result:
    st.header("1 · Upload Files")
    col1, col2 = st.columns([2, 1])

    with col1:
        tx_file = st.file_uploader("Spend Transactions CSV **(required)**", type=["csv"])
        sup_file = st.file_uploader("Supplier Master CSV _(optional)_", type=["csv"])
        cat_file = st.file_uploader("Category Taxonomy CSV _(optional)_", type=["csv"])
        company_ctx = st.text_area(
            "Company Context",
            placeholder="E.g. Mid-market industrial manufacturer, €150M revenue, preparing Q1 board review.",
            height=80,
        )

    with col2:
        st.info(
            "**Required columns**\n\n"
            "- `date`\n- `supplier_name`\n- `amount`\n- `category`\n\n"
            "**Recommended**\n\n"
            "- `invoice_number`\n- `department`\n- `payment_terms`"
        )

    if tx_file:
        data: dict = {"transactions": read_csv(tx_file)}
        if sup_file:
            data["supplier_master"] = read_csv(sup_file)
        if cat_file:
            data["category_taxonomy"] = read_csv(cat_file)

        df_tx = data["transactions"]
        st.success(f"Loaded **{len(df_tx):,}** transactions · **{df_tx['supplier_name'].nunique() if 'supplier_name' in df_tx.columns else '—'}** suppliers · **{fmt_eur(float(df_tx['amount'].sum())) if 'amount' in df_tx.columns else '—'}** total spend")

        if st.button("🚀 Run AI Analysis", type="primary"):
            progress_msgs = []

            def on_progress(msg: str) -> None:
                progress_msgs.append(msg)

            with st.status("Running 4-step analysis…", expanded=True) as status_box:
                agent = SpendAnalysisAgent(progress_callback=on_progress)

                original_log = agent._log

                def live_log(msg: str) -> None:
                    st.write(msg)
                    original_log(msg)

                agent._log = live_log

                try:
                    result = agent.analyze(data, company_ctx)
                    status_box.update(label="Analysis complete!", state="complete")
                    st.session_state.result = result
                    st.session_state.data = data
                    st.rerun()
                except Exception as e:
                    status_box.update(label="Analysis failed", state="error")
                    st.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    report: dict = result.get("report", {})
    stats: dict = result.get("statistics", {})
    anomalies: dict = result.get("anomalies", {})
    categorisation: dict = result.get("categorisation", {})

    st.header("📋 Spend Analysis Report")

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

    # ── Tab 1: Executive Summary ──
    with tabs[0]:
        st.subheader("Executive Summary")
        for para in report.get("executive_summary", "").split("\n\n"):
            if para.strip():
                st.markdown(para.strip())

        st.markdown("---")
        st.subheader("Key Findings")
        for f in report.get("key_findings", []):
            sev = f.get("severity", "Medium")
            with st.expander(f"{sev_icon(sev)} {f.get('title', '')} — {fmt_eur(f.get('financial_impact_eur', 0))}"):
                st.markdown(f.get("description", ""))

        st.markdown("---")
        st.subheader("Quick Wins")
        for w in categorisation.get("quick_wins", []):
            st.markdown(
                f"**{w.get('title', '')}** · _{w.get('timeline', '')}_  \n"
                f"{w.get('description', '')}  \n"
                f"Estimated impact: **{fmt_eur(w.get('estimated_impact_eur', 0))}**"
            )
            st.markdown("---")

    # ── Tab 2: Spend Breakdown ──
    with tabs[1]:
        col1, col2 = st.columns(2)

        cat_data = stats.get("by_category", {})
        if cat_data:
            df_cat = pd.DataFrame(
                [(k, v["sum"], v.get("pct_of_total", 0)) for k, v in cat_data.items()],
                columns=["Category", "Spend (€)", "% of Total"],
            ).sort_values("Spend (€)", ascending=False)
            with col1:
                st.subheader("Spend by Category")
                fig = px.pie(df_cat, names="Category", values="Spend (€)", hole=0.45,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_cat.style.format({"Spend (€)": "€{:,.0f}", "% of Total": "{:.1f}%"}),
                             hide_index=True, use_container_width=True)

        sup_data = stats.get("by_supplier", {})
        if sup_data:
            df_sup = pd.DataFrame(
                [(k, v["sum"], v.get("pct_of_total", 0)) for k, v in sup_data.items()],
                columns=["Supplier", "Spend (€)", "% of Total"],
            ).sort_values("Spend (€)", ascending=False).head(10)
            with col2:
                st.subheader("Top 10 Suppliers")
                fig = px.bar(df_sup, x="Spend (€)", y="Supplier", orientation="h",
                             color="% of Total", color_continuous_scale="Blues", text="Spend (€)")
                fig.update_traces(texttemplate="€%{x:,.0f}", textposition="outside")
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        dept_data = stats.get("by_department", {})
        if dept_data:
            st.subheader("Spend by Department")
            df_dept = pd.DataFrame(list(dept_data.items()), columns=["Department", "Spend (€)"]).sort_values("Spend (€)", ascending=False)
            fig = px.bar(df_dept, x="Department", y="Spend (€)", color="Spend (€)",
                         color_continuous_scale="Teal", text="Spend (€)")
            fig.update_traces(texttemplate="€%{y:,.0f}", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        pt_data = stats.get("payment_terms_distribution", {})
        if pt_data:
            st.subheader("Payment Terms Distribution")
            df_pt = pd.DataFrame(list(pt_data.items()), columns=["Terms", "Count"])
            fig = px.pie(df_pt, names="Terms", values="Count", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Anomalies ──
    with tabs[2]:
        risk_score = anomalies.get("overall_risk_score", 0)
        icon = "🔴" if risk_score >= 7 else "🟡" if risk_score >= 4 else "🟢"
        st.subheader(f"Procurement Risk Score: {icon} {risk_score}/10")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        def risk_card(col, title: str, d: dict) -> None:
            sev = d.get("severity", "Low")
            col.metric(title, fmt_eur(d.get("exposure_eur", d.get("total_value_eur", 0))),
                       delta=sev, delta_color="inverse" if sev == "High" else "off")
            col.markdown(d.get("description", ""))
            if "recommended_action" in d:
                col.info(f"**Action:** {d['recommended_action']}")

        risk_card(c1, "Duplicate Invoices", anomalies.get("duplicate_invoice_risk", {}))
        risk_card(c2, "Invoice Splitting", anomalies.get("split_invoice_risk", {}))
        risk_card(c3, "Inactive Suppliers", anomalies.get("inactive_supplier_risk", {}))

        st.markdown("---")
        st.subheader("Immediate Actions Required")
        for i, action in enumerate(anomalies.get("immediate_actions", []), 1):
            st.markdown(f"**{i}.** {action}")

        for o in anomalies.get("high_value_outliers", []):
            st.warning(str(o))

    # ── Tab 4: Savings Plan ──
    with tabs[3]:
        total_sav = report.get("total_addressable_savings_eur", 0)
        pct = report.get("total_savings_pct_of_spend", 0)
        st.subheader(f"Total Addressable Savings: {fmt_eur(total_sav)} ({pct:.1f}% of spend)")

        opps = report.get("savings_opportunities", [])
        if opps:
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(opps) + ["total"],
                x=[o.get("initiative", "")[:25] for o in opps] + ["Total"],
                y=[o.get("estimated_savings_eur", 0) for o in opps] + [0],
                totals={"marker": {"color": "#0066cc"}},
                connector={"line": {"color": "#adb5bd"}},
            ))
            fig.update_layout(title="Savings Waterfall (€)", showlegend=False,
                              margin=dict(t=40, b=40), yaxis_tickformat="€,.0f")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            for opp in sorted(opps, key=lambda o: o.get("estimated_savings_eur", 0), reverse=True):
                with st.expander(f"💰 {opp.get('initiative', '')} — {fmt_eur(opp.get('estimated_savings_eur', 0))} ({opp.get('savings_pct', 0):.1f}%)"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Category", opp.get("category", "—"))
                    c2.metric("Savings", fmt_eur(opp.get("estimated_savings_eur", 0)))
                    c3.metric("Effort", effort_label(opp.get("effort", "Medium")))
                    c4.metric("Timeline", opp.get("timeline", "—"))
                    for a in opp.get("actions", []):
                        st.markdown(f"- {a}")

    # ── Tab 5: Action Plan ──
    with tabs[4]:
        st.subheader("90-Day Action Plan")
        plan = report.get("next_90_day_plan", [])
        if plan:
            st.dataframe(pd.DataFrame(plan), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Risk Mitigation Controls")
        for ctrl in sorted(report.get("risk_mitigation_plan", []), key=lambda c: c.get("priority", 99)):
            st.markdown(f"**{ctrl.get('priority', '')}. {ctrl.get('control', '')}**  \n{ctrl.get('description', '')}")
            st.markdown("---")
