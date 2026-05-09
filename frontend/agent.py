import json
import os
from typing import Any, Callable, Optional

import pandas as pd
from huggingface_hub import InferenceClient

_SYSTEM = """You are a senior procurement consultant with 15 years of experience in
spend analysis, strategic sourcing, and cost optimisation. You produce board-ready
findings: specific numbers, realistic savings estimates (5-15% of addressable spend),
and prioritised actions. Never invent data; only use figures from what you are given."""


def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to environment variable."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


def _strip_fences(text: str) -> str:
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


class SpendAnalysisAgent:
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        self._log = progress_callback or (lambda _: None)
        self._client = InferenceClient(
            model=_get_secret("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            token=_get_secret("HF_API_TOKEN"),
        )

    def analyze(self, data: dict, company_context: str) -> dict:
        df_tx: pd.DataFrame = data["transactions"]
        df_sup: Optional[pd.DataFrame] = data.get("supplier_master")
        df_cat: Optional[pd.DataFrame] = data.get("category_taxonomy")

        self._log("Step 1/4: Computing spend statistics…")
        stats = self._compute_statistics(df_tx, df_sup, df_cat)

        self._log("Step 2/4: Analysing spend patterns and categorisation…")
        categorisation = self._call_llm(self._categorisation_prompt(stats, company_context))

        self._log("Step 3/4: Detecting anomalies and control weaknesses…")
        anomalies = self._call_llm(self._anomaly_prompt(df_tx, stats, categorisation))

        self._log("Step 4/4: Generating executive report and savings plan…")
        report = self._call_llm(self._report_prompt(stats, categorisation, anomalies, company_context))

        return {
            "statistics": stats,
            "categorisation": categorisation,
            "anomalies": anomalies,
            "report": report,
        }

    # ------------------------------------------------------------------
    # Step 1 — pure Python statistics
    # ------------------------------------------------------------------
    def _compute_statistics(
        self,
        df_tx: pd.DataFrame,
        df_sup: Optional[pd.DataFrame],
        df_cat: Optional[pd.DataFrame],
    ) -> dict:
        stats: dict[str, Any] = {}

        stats["total_spend"] = float(df_tx["amount"].sum())
        stats["transaction_count"] = int(len(df_tx))
        stats["avg_transaction"] = float(df_tx["amount"].mean())

        if "date" in df_tx.columns:
            stats["date_range"] = {
                "start": str(df_tx["date"].min()),
                "end": str(df_tx["date"].max()),
            }

        if "category" in df_tx.columns:
            cat = (
                df_tx.groupby("category")["amount"]
                .agg(["sum", "count", "mean"])
                .round(2)
                .sort_values("sum", ascending=False)
            )
            cat["pct_of_total"] = (cat["sum"] / stats["total_spend"] * 100).round(2)
            stats["by_category"] = cat.to_dict(orient="index")

        if "supplier_name" in df_tx.columns:
            sup = (
                df_tx.groupby("supplier_name")["amount"]
                .agg(["sum", "count"])
                .round(2)
                .sort_values("sum", ascending=False)
            )
            sup["pct_of_total"] = (sup["sum"] / stats["total_spend"] * 100).round(2)
            stats["by_supplier"] = sup.head(20).to_dict(orient="index")
            stats["supplier_count"] = int(df_tx["supplier_name"].nunique())
            stats["top5_supplier_concentration_pct"] = round(
                sup.head(5)["sum"].sum() / stats["total_spend"] * 100, 1
            )

        if "department" in df_tx.columns:
            stats["by_department"] = (
                df_tx.groupby("department")["amount"].sum().round(2).sort_values(ascending=False).to_dict()
            )

        if "payment_terms" in df_tx.columns:
            stats["payment_terms_distribution"] = df_tx["payment_terms"].value_counts().to_dict()

        if "invoice_number" in df_tx.columns and "supplier_name" in df_tx.columns:
            dup_mask = df_tx.duplicated(subset=["invoice_number", "supplier_name"], keep=False)
            dups = df_tx[dup_mask]
            stats["duplicate_invoice_count"] = int(len(dups))
            stats["duplicate_invoice_exposure_eur"] = float(dups["amount"].sum() / 2)
            stats["duplicate_invoices_sample"] = (
                dups[["invoice_number", "supplier_name", "amount", "department"]]
                .head(10)
                .to_dict(orient="records")
            )

        split_candidates = []
        if "date" in df_tx.columns and "supplier_name" in df_tx.columns:
            sub = df_tx[df_tx["amount"] < 5_000]
            for (supplier, date), grp in sub.groupby(["supplier_name", "date"]):
                if len(grp) >= 3 and grp["amount"].sum() > 5_000:
                    split_candidates.append({
                        "supplier": supplier,
                        "date": str(date),
                        "num_invoices": int(len(grp)),
                        "total_amount": float(grp["amount"].sum()),
                        "individual_amounts": [round(a, 2) for a in grp["amount"].tolist()],
                    })
        stats["split_invoice_candidates"] = split_candidates[:5]

        if (
            df_sup is not None
            and "contract_status" in df_sup.columns
            and "supplier_name" in df_sup.columns
            and "supplier_name" in df_tx.columns
        ):
            inactive = set(df_sup[df_sup["contract_status"] == "Inactive"]["supplier_name"].tolist())
            stats["inactive_supplier_spend_eur"] = float(
                df_tx[df_tx["supplier_name"].isin(inactive)]["amount"].sum()
            )
            stats["inactive_suppliers"] = sorted(inactive)

        if df_cat is not None and "category" in df_cat.columns and "savings_potential_pct" in df_cat.columns:
            stats["category_benchmark_savings"] = df_cat.set_index("category")["savings_potential_pct"].to_dict()

        return stats

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def _categorisation_prompt(self, stats: dict, company_context: str) -> str:
        return f"""You are reviewing procurement spend data for a client.

COMPANY CONTEXT: {company_context or "Mid-market industrial manufacturing company, ~€150M revenue"}

SPEND SUMMARY:
- Total spend analysed: €{stats['total_spend']:,.0f}
- Transactions: {stats['transaction_count']}
- Suppliers: {stats.get('supplier_count', 'N/A')}
- Date range: {stats.get('date_range', {}).get('start', 'N/A')} → {stats.get('date_range', {}).get('end', 'N/A')}
- Top-5 supplier concentration: {stats.get('top5_supplier_concentration_pct', 'N/A')}%

SPEND BY CATEGORY:
{json.dumps(stats.get('by_category', {}), indent=2)}

TOP 20 SUPPLIERS:
{json.dumps(stats.get('by_supplier', {}), indent=2)}

SPEND BY DEPARTMENT:
{json.dumps(stats.get('by_department', {}), indent=2)}

PAYMENT TERMS:
{json.dumps(stats.get('payment_terms_distribution', {}), indent=2)}

Return ONLY a JSON object with these keys:
- "category_insights": array — one object per category with keys: category, spend_eur, pct_of_total, key_observation
- "supplier_insights": array — top-5 observations about supplier base
- "department_insights": array — observations per department
- "spend_concentration_risk": string — concise risk assessment
- "payment_terms_insight": string — are terms optimised?
- "quick_wins": array of 3 objects with keys: title, description, estimated_impact_eur, timeline"""

    def _anomaly_prompt(self, df_tx: pd.DataFrame, stats: dict, categorisation: dict) -> str:
        high_value_cols = [
            c for c in ["invoice_number", "transaction_id", "supplier_name", "amount", "category", "date", "department"]
            if c in df_tx.columns
        ]
        high_value = df_tx.nlargest(10, "amount")[high_value_cols].to_dict(orient="records")

        return f"""You are a forensic procurement analyst reviewing control weaknesses.

DUPLICATE INVOICE DATA:
- Count: {stats.get('duplicate_invoice_count', 0)}
- Exposure: €{stats.get('duplicate_invoice_exposure_eur', 0):,.0f}
- Sample: {json.dumps(stats.get('duplicate_invoices_sample', []), indent=2, default=str)}

SPLIT-INVOICE CANDIDATES:
{json.dumps(stats.get('split_invoice_candidates', []), indent=2, default=str)}

INACTIVE SUPPLIER SPEND:
- Total: €{stats.get('inactive_supplier_spend_eur', 0):,.0f}
- Suppliers: {stats.get('inactive_suppliers', [])}

TOP-10 HIGH-VALUE TRANSACTIONS:
{json.dumps(high_value, indent=2, default=str)}

Return ONLY a JSON object with these keys:
- "duplicate_invoice_risk": object with keys: severity (High/Medium/Low), exposure_eur, description, recommended_action
- "split_invoice_risk": object with keys: severity, num_candidates, total_value_eur, description, recommended_action
- "inactive_supplier_risk": object with keys: severity, exposure_eur, description, recommended_action
- "high_value_outliers": array of up to 3 noteworthy transactions with explanation
- "overall_risk_score": integer 1–10 with rationale
- "immediate_actions": array of 3 strings"""

    def _report_prompt(self, stats: dict, categorisation: dict, anomalies: dict, company_context: str) -> str:
        return f"""You are writing a procurement spend analysis report for the CFO and board.

COMPANY: {company_context or "Mid-market industrial manufacturing company, ~€150M revenue"}

SPEND DATA SUMMARY:
- Total spend: €{stats['total_spend']:,.0f}
- Transactions: {stats['transaction_count']}
- Suppliers: {stats.get('supplier_count', 'N/A')}
- Top-5 concentration: {stats.get('top5_supplier_concentration_pct', 'N/A')}%
- Duplicate invoice exposure: €{stats.get('duplicate_invoice_exposure_eur', 0):,.0f}
- Inactive supplier spend: €{stats.get('inactive_supplier_spend_eur', 0):,.0f}

PATTERN ANALYSIS:
{json.dumps(categorisation, indent=2)}

ANOMALY FINDINGS:
{json.dumps(anomalies, indent=2)}

Return ONLY a JSON object with these keys:
"executive_summary": string — 3-paragraph board-ready summary
"key_findings": array of 6 objects with: title, description, financial_impact_eur, severity (High/Medium/Low)
"savings_opportunities": array of 6 objects with: initiative, category, estimated_savings_eur, savings_pct, effort (Low/Medium/High), timeline ("0-30 days"|"30-90 days"|"90-180 days"), actions (array of 2-3 steps)
"total_addressable_savings_eur": number
"total_savings_pct_of_spend": number
"risk_mitigation_plan": array of 4 objects with: control, description, priority (1-4)
"next_90_day_plan": array of objects with: week_range, milestone, owner_role

Use specific numbers from the data. Savings estimates must be 5-15% of addressable spend."""

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> dict:
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,
        )
        raw = response.choices[0].message.content or ""
        return json.loads(_strip_fences(raw))
