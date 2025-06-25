# metrics.py

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self):
        self.results = []

    def add_result(self, question: str, method: str, metrics: dict, sql_query: str = ""):
        result = {
            "timestamp": time.time(),
            "question": question,
            "method": method,
            "sql_query": sql_query,
            **metrics
        }
        self.results.append(result)

    def generate_comparison_report(self):
        if not self.results:
            print(" No results to analyze.")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Score Distribution
        if 'best_score' in df.columns and 'score' in df.columns:
            mcts_scores = df[df['method'] == 'mcts']['best_score'].tolist()
            baseline_scores = df[df['method'] == 'baseline']['score'].tolist()

            axes[0, 0].hist([baseline_scores, mcts_scores], label=['Baseline', 'MCTS'], alpha=0.7)
            axes[0, 0].set_title('Score Distribution')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()

        # 2. Execution Time
        axes[0, 1].boxplot([
            df[df['method'] == 'baseline']['execution_time'].tolist(),
            df[df['method'] == 'mcts']['execution_time'].tolist()
        ], labels=['Baseline', 'MCTS'])
        axes[0, 1].set_title('Execution Time (s)')

        # 3. MCTS Iteration Score Progression
        mcts_data = df[df['method'] == 'mcts']
        avg_scores = []
        if not mcts_data.empty and 'iteration_scores' in mcts_data.columns:
            max_iters = max(len(row) for row in mcts_data['iteration_scores'] if isinstance(row, list))
            for i in range(max_iters):
                iter_scores = [row[i] for row in mcts_data['iteration_scores'] if isinstance(row, list) and len(row) > i]
                if iter_scores:
                    avg_scores.append(np.mean(iter_scores))

            axes[1, 0].plot(range(1, len(avg_scores) + 1), avg_scores, marker='o')
            axes[1, 0].set_title('MCTS Avg Score per Iteration')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Avg Score')

        # 4. Success Rate
        success_baseline = len(df[(df['method'] == 'baseline') & (df.get('score', 0) > 0.5)])
        success_mcts = len(df[(df['method'] == 'mcts') & (df.get('best_score', 0) > 0.5)])
        total_baseline = len(df[df['method'] == 'baseline'])
        total_mcts = len(df[df['method'] == 'mcts'])

        axes[1, 1].bar(['Baseline', 'MCTS'], [
            success_baseline / max(total_baseline, 1),
            success_mcts / max(total_mcts, 1)
        ], color=['skyblue', 'lightcoral'])

        axes[1, 1].set_title('Success Rate (Score > 0.5)')
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        # Always save the figure since SSH likely lacks GUI backend
        output_file = "comparison_report.png"
        plt.savefig(output_file)
        print(f"Report saved as {output_file}")


