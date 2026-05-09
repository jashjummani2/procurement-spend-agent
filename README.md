# Procurement Spend Analysis Agent

An AI-powered consulting tool that analyses procurement spend data and generates board-ready insights in minutes.

## Consulting Scenario

**Situation:** A mid-market industrial manufacturing company (~€150M revenue) is preparing for a board review. The CFO wants to understand indirect procurement spend patterns, identify cost-saving opportunities, and surface control weaknesses — work that would normally take a consultant 2-5 days.

**What the agent does (4 steps):**
1. Computes spend statistics (Python/pandas)
2. Analyses categorisation and patterns (Claude)
3. Detects anomalies — duplicate invoices, split invoices, inactive supplier spend (Claude)
4. Generates executive report with quantified savings opportunities (Claude)

**Evaluation criteria:**
- Savings estimates realistic (5–15% of addressable spend)
- Anomalies detected with specific evidence (€ values, invoice numbers)
- Findings structured for board presentation (executive summary, prioritised actions)
- Numbers in the report cross-check against the raw data

## Quick Start

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
cp ../.env.example .env   # fill in ANTHROPIC_API_KEY
uvicorn main:app --reload --port 8000
```

### 2. Frontend (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

### 3. Sample Data

Upload from `sample_data/`:
- `spend_transactions.csv` — 240 transactions, full year 2024 (required)
- `supplier_master.csv` — 19 suppliers with contract status (optional, recommended)
- `category_taxonomy.csv` — 40 categories with benchmark savings % (optional)

**Embedded anomalies in the sample data:**
- TXN-024: Duplicate invoice `INV-TP-2401` (same as TXN-001, paid twice — €12,500 exposure)
- TXN-119–121: SoftEdge AG split-invoice pattern on 2024-07-01 (3 × <€5k on same day)
- TXN-184–186: Second split-invoice cluster on 2024-10-01
- SUP-005 (CyberShield Ltd) & SUP-009 (GreenSpace FM): Inactive supplier spend
- SUP-019 (ExecutiveFlight Ltd): Non-preferred travel supplier at premium rates

## Project Structure

```
English-quest/
├── backend/
│   ├── main.py              # FastAPI: upload, analyse, session endpoints
│   ├── agent.py             # 4-step Claude agent (statistics → categorisation → anomalies → report)
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit: upload → progress → 5-tab results
│   └── requirements.txt
├── sample_data/
│   ├── spend_transactions.csv
│   ├── supplier_master.csv
│   └── category_taxonomy.csv
└── .env.example
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload CSV files, get session_id |
| POST | `/analyze/{session_id}` | Start background analysis |
| GET | `/sessions/{session_id}` | Poll for status and results |

## Result Tabs

| Tab | Content |
|-----|---------|
| Executive Summary | Board-ready 3-paragraph summary + key findings + quick wins |
| Spend Breakdown | Category donut, supplier bar chart, department chart, payment terms |
| Anomalies | Risk score, duplicate/split/inactive supplier findings, immediate actions |
| Savings Plan | Waterfall chart + detailed opportunity cards with effort/timeline |
| Action Plan | 90-day milestone table + risk mitigation controls |
