# api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mcts_engine import MCTSTextToSQL
from db_handler import SQLiteHandler
from llm_handler import LLMHandler
import time

app = FastAPI(title="MCTS Text-to-SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Load handlers globally
db_path = "california_schools.sqlite"  # Change path accordingly
db_handler = SQLiteHandler(db_path)
llm_handler = LLMHandler()
mcts_converter = MCTSTextToSQL(llm_handler, db_handler)

class QueryRequest(BaseModel):
    question: str
    use_mcts: bool = True
    mcts_iterations: int = 10

class QueryResponse(BaseModel):
    sql_query: str
    execution_success: bool
    results: dict
    metrics: dict
    error_message: str = ""

@app.post("/query", response_model=QueryResponse)
def convert_question_to_sql(request: QueryRequest):
    if request.use_mcts:
        sql_query, metrics = mcts_converter.mcts_search(request.question, request.mcts_iterations)
        metrics["method"] = "mcts"
    else:
        sql_query, metrics = mcts_converter.simple_baseline(request.question)

    success, results, error = db_handler.execute_query(sql_query)
    return QueryResponse(
        sql_query=sql_query,
        execution_success=success,
        results=results if success else {},
        metrics=metrics,
        error_message=error
    )

@app.get("/schema")
def get_schema():
    return {"schema": db_handler.schema}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}
