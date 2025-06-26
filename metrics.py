# # metrics.py

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

    def generate_comparison_report(self, output_prefix="comparison_report"):
        if not self.results:
            print("No results to analyze.")
            return

        df = pd.DataFrame(self.results)

        # Extract unique LLM names from method strings like 'mistral-baseline'
        llm_names = sorted(set(method.split("-")[0] for method in df["method"]))

        for llm in llm_names:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            llm_df = df[df["method"].str.startswith(llm)]

            print(f"\n Generating report for: {llm}")

            # Score Distribution
            mcts_scores = llm_df[llm_df['method'].str.endswith('mcts')]['best_score'].tolist()
            baseline_scores = llm_df[llm_df['method'].str.endswith('baseline')]['score'].tolist()

            axes[0, 0].hist([baseline_scores, mcts_scores], label=['Baseline', 'MCTS'], alpha=0.7)
            axes[0, 0].set_title(f'{llm.capitalize()} - Score Distribution')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()

            # Execution Time
            axes[0, 1].boxplot([
                llm_df[llm_df['method'].str.endswith('baseline')]['execution_time'].tolist(),
                llm_df[llm_df['method'].str.endswith('mcts')]['execution_time'].tolist()
            ], labels=['Baseline', 'MCTS'])
            axes[0, 1].set_title(f'{llm.capitalize()} - Execution Time (s)')

            # MCTS Iteration Score Progression
            mcts_data = llm_df[llm_df['method'].str.endswith('mcts')]
            avg_scores = []
            if not mcts_data.empty and 'iteration_scores' in mcts_data.columns:
                max_iters = max(len(row) for row in mcts_data['iteration_scores'] if isinstance(row, list))
                for i in range(max_iters):
                    iter_scores = [row[i] for row in mcts_data['iteration_scores']
                                   if isinstance(row, list) and len(row) > i]
                    if iter_scores:
                        avg_scores.append(np.mean(iter_scores))

                axes[1, 0].plot(range(1, len(avg_scores) + 1), avg_scores, marker='o')
                axes[1, 0].set_title(f'{llm.capitalize()} - MCTS Avg Score per Iteration')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Avg Score')

            # Success Rate
            success_baseline = len(llm_df[(llm_df['method'].str.endswith('baseline')) & (llm_df.get('score', 0) > 0.5)])
            success_mcts = len(llm_df[(llm_df['method'].str.endswith('mcts')) & (llm_df.get('best_score', 0) > 0.5)])
            total_baseline = len(llm_df[llm_df['method'].str.endswith('baseline')])
            total_mcts = len(llm_df[llm_df['method'].str.endswith('mcts')])

            axes[1, 1].bar(['Baseline', 'MCTS'], [
                success_baseline / max(total_baseline, 1),
                success_mcts / max(total_mcts, 1)
            ], color=['skyblue', 'lightcoral'])

            axes[1, 1].set_title(f'{llm.capitalize()} - Success Rate (Score > 0.5)')
            axes[1, 1].set_ylim(0, 1)

            plt.tight_layout()
            filename = f"{output_prefix}_{llm}.png"
            plt.savefig(filename)
            print(f" Saved: {filename}")
            plt.close()
