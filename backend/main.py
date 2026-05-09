import io
import uuid
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).parent.parent / ".env")

from agent import SpendAnalysisAgent

app = FastAPI(title="Procurement Spend Analysis Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict = {}
uploaded_data: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_files(
    transactions: Annotated[UploadFile, File(description="Spend transactions CSV")],
    supplier_master: Annotated[Optional[UploadFile], File()] = None,
    category_taxonomy: Annotated[Optional[UploadFile], File()] = None,
):
    session_id = str(uuid.uuid4())
    data: dict = {}

    content = await transactions.read()
    df_tx = pd.read_csv(io.StringIO(content.decode("utf-8")))
    data["transactions"] = df_tx

    if supplier_master and supplier_master.filename:
        content = await supplier_master.read()
        data["supplier_master"] = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if category_taxonomy and category_taxonomy.filename:
        content = await category_taxonomy.read()
        data["category_taxonomy"] = pd.read_csv(io.StringIO(content.decode("utf-8")))

    uploaded_data[session_id] = data
    sessions[session_id] = {"status": "uploaded", "result": None, "progress": []}

    summary = {
        "session_id": session_id,
        "transactions_count": len(df_tx),
        "columns": list(df_tx.columns),
    }
    if "amount" in df_tx.columns:
        summary["total_spend"] = float(df_tx["amount"].sum())
    if "date" in df_tx.columns:
        summary["date_range"] = {
            "start": str(df_tx["date"].min()),
            "end": str(df_tx["date"].max()),
        }
    if "supplier_name" in df_tx.columns:
        summary["supplier_count"] = int(df_tx["supplier_name"].nunique())

    return summary


@app.post("/analyze/{session_id}")
async def start_analysis(
    session_id: str,
    background_tasks: BackgroundTasks,
    company_context: str = Form(default=""),
):
    if session_id not in uploaded_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if sessions[session_id]["status"] == "analyzing":
        raise HTTPException(status_code=409, detail="Analysis already in progress")

    sessions[session_id]["status"] = "analyzing"
    sessions[session_id]["progress"] = []
    background_tasks.add_task(_run_analysis, session_id, company_context)

    return {"session_id": session_id, "status": "analyzing"}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


async def _run_analysis(session_id: str, company_context: str) -> None:
    def on_progress(msg: str) -> None:
        sessions[session_id]["progress"].append(msg)

    try:
        agent = SpendAnalysisAgent(progress_callback=on_progress)
        result = await agent.analyze(uploaded_data[session_id], company_context)
        sessions[session_id]["status"] = "completed"
        sessions[session_id]["result"] = result
    except Exception as exc:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(exc)
