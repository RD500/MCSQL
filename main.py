# main.py

from mcts_engine import MCTSTextToSQL
from llm_handler import LLMHandler
from db_handler import SQLiteHandler
from metrics import MetricsTracker

DB_PATH = "california_schools.sqlite"

def main():
    print("🔧 Initializing...")
    db = SQLiteHandler(DB_PATH)
    llm = LLMHandler()
    mcts = MCTSTextToSQL(llm, db)
    tracker = MetricsTracker()

    demo_questions = [
        "How many students are there in total?",
        "What are the names of all schools?",
        "Which district has the most schools?"
    ]

    for q in demo_questions:
        print(f"\n❓ Question: {q}")

        # Baseline
        print("Running baseline...")
        base_query, base_metrics = mcts.simple_baseline(q)
        print(f"Baseline SQL: {base_query}")
        print(f"Score: {base_metrics['score']:.2f}")
        tracker.add_result(q, "baseline", base_metrics, base_query)

        # MCTS
        print("Running MCTS...")
        best_query, mcts_metrics = mcts.mcts_search(q, num_iterations=10)
        print(f"MCTS SQL: {best_query}")
        print(f"Best Score: {mcts_metrics['best_score']:.2f}")
        tracker.add_result(q, "mcts", mcts_metrics, best_query)

    tracker.generate_comparison_report()

if __name__ == "__main__":
    main()
