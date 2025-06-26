# # # main.py

# from mcts_engine import MCTSTextToSQL
# from llm_handler import LLMHandler
# from db_handler import SQLiteHandler
# from metrics import MetricsTracker

# DB_PATH = "california_schools.sqlite"

# def main():
#     print("🔧 Initializing...")
#     db = SQLiteHandler(DB_PATH)
#     llm = LLMHandler()
#     mcts = MCTSTextToSQL(llm, db)
#     tracker = MetricsTracker()

#     demo_questions = [
#         "How many students are there in total?",
#         "What are the names of all schools?",
#         "Which district has the most schools?"
#     ]

#     for q in demo_questions:
#         print(f"\n❓ Question: {q}")

#         # Baseline
#         print("Running baseline...")
#         base_query, base_metrics = mcts.simple_baseline(q)
#         print(f"Baseline SQL: {base_query}")
#         print(f"Score: {base_metrics['score']:.2f}")
#         tracker.add_result(q, "baseline", base_metrics, base_query)

#         # MCTS
#         print("Running MCTS...")
#         best_query, mcts_metrics = mcts.mcts_search(q, num_iterations=10)
#         print(f"MCTS SQL: {best_query}")
#         print(f"Best Score: {mcts_metrics['best_score']:.2f}")
#         tracker.add_result(q, "mcts", mcts_metrics, best_query)

#     tracker.generate_comparison_report()

# if __name__ == "__main__":
#     main()

# main.py
from tabulate import tabulate
from mcts_engine import MCTSTextToSQL
from llm_handler import LLMHandler
from db_handler import SQLiteHandler
from metrics import MetricsTracker

DB_PATH = "california_schools.sqlite"
LLM_MODELS = ["llama3.1:8b"]
demo_questions = [
    "What are the names of all schools?",
    "Show me schools with enrollment greater than 1000"
]

def write_to_file(file, content):
    file.write(content + "\n")
    print(content)

def main():
    print("🔧 Initializing...")
    db = SQLiteHandler(DB_PATH)

    for model_name in LLM_MODELS:
        output_file_path = f"results_{model_name.replace(':', '_')}.txt"
        with open(output_file_path, "w") as f:
            write_to_file(f, f"\nRunning tests with LLM: {model_name}")
            llm = LLMHandler(model=model_name)
            mcts = MCTSTextToSQL(llm, db)
            tracker = MetricsTracker()

            for q in demo_questions:
                write_to_file(f, f"\n❓ Question: {q}")

                # --- Baseline ---
                write_to_file(f, "Running baseline...")
                base_query, base_metrics = mcts.simple_baseline(q)
                write_to_file(f, f"Baseline SQL: {base_query}")
                write_to_file(f, f"Score: {base_metrics['score']:.2f}")
                tracker.add_result(q, f"{model_name}-baseline", base_metrics, base_query)

                success, output, error = db.execute_query(base_query)
                write_to_file(f, " Final Baseline Query Output:")
                if success and output["data"]:
                    table_str = tabulate(output["data"], headers=output["columns"], tablefmt="grid")
                    write_to_file(f, table_str)
                elif not success:
                    write_to_file(f, f"Error executing baseline query: {error}")
                else:
                    write_to_file(f, "Baseline query executed, but returned no results.")

                # --- MCTS ---
                write_to_file(f, "Running MCTS...")
                best_query, mcts_metrics = mcts.mcts_search(q, num_iterations=10)
                write_to_file(f, f" MCTS SQL: {best_query}")
                write_to_file(f, f" Best Score: {mcts_metrics['best_score']:.2f}")
                tracker.add_result(q, f"{model_name}-mcts", mcts_metrics, best_query)

                success, output, error = db.execute_query(best_query)
                write_to_file(f, " Final MCTS Query Output:")
                if success and output["data"]:
                    table_str = tabulate(output["data"], headers=output["columns"], tablefmt="grid")
                    write_to_file(f, table_str)
                elif not success:
                    write_to_file(f, f" Error executing MCTS query: {error}")
                else:
                    write_to_file(f, "MCTS query executed, but returned no results.")

            # Save Report
            tracker.generate_comparison_report(f"comparison_report_{model_name}")

if __name__ == "__main__":
    main()
