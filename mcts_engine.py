from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import time
import math
import numpy as np

class SQLEvaluator:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def evaluate_query(self, query: str, question: str = "") -> float:
        score = 0.0

        valid, result, error = self.db_handler.execute_query(query)
        if not valid:
            return 0.1

        score += 0.4

        if result and result.get("data"):
            score += 0.2
            row_count = len(result["data"])
            score += 0.2 if 1 <= row_count <= 1000 else 0.1

        query_upper = query.upper()
        for keyword in ['JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'WHERE']:
            if keyword in query_upper:
                score += 0.05

        return min(score, 1.0)

@dataclass
class MCTSNode:
    sql_query: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    reward_sum: float = 0.0
    is_terminal: bool = False

    def __post_init__(self):
        self.children = self.children or []

    def ucb_score(self, exploration_weight: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        avg_reward = self.reward_sum / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return avg_reward + exploration

    def is_fully_expanded(self, max_children: int = 5) -> bool:
        return len(self.children) >= max_children

    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda child: child.ucb_score())

    def add_child(self, sql_query: str) -> 'MCTSNode':
        child = MCTSNode(sql_query=sql_query, parent=self)
        self.children.append(child)
        return child

class MCTSTextToSQL:
    def __init__(self, llm_handler, db_handler):
        self.llm = llm_handler
        self.db_handler = db_handler
        self.evaluator = SQLEvaluator(db_handler)

    def create_base_prompt(self, question: str) -> str:
        schema_lines = []
        for table, meta in self.db_handler.schema.items():
            schema_lines.append(f"Table `{table}` has the following columns:")
            for col, typ in zip(meta['columns'], meta['types']):
            # Use quoting for safe SQL parsing
                quoted_col = f'"{col}"' if " " in col or "(" in col else col
                schema_lines.append(f" - {quoted_col} ({typ})")

        schema_text = "\n".join(schema_lines)

        return f"""
You are an expert in generating valid SQLite SQL queries.

Instructions:
- Use **only** the tables and columns provided in the schema.
- Use **double quotes** around column names with spaces or special characters.
- End your query with a semicolon.
- Only use valid SQL syntax that works in SQLite.

Schema:
{schema_text}

Question:
\"{question}\"

SQL Query:
"""


    def generate_prompt_variations(self, base_prompt: str, index: int) -> str:
        hints = [
            "",
            "\nHint: use GROUP BY if aggregation is required.",
            "\nHint: use JOIN to combine multiple tables.",
            "\nHint: consider adding WHERE clause.",
            "\nHint: use ORDER BY to sort results."
        ]
        return base_prompt + hints[index % len(hints)]

    def mcts_search(self, question: str, num_iterations: int = 10, max_time: float = 30.0) -> Tuple[str, dict]:
        base_prompt = self.create_base_prompt(question)
        root_query = self.llm.generate_sql(base_prompt)
        root = MCTSNode(sql_query=root_query)
        best_query = root_query
        best_score = 0.0
        iteration_scores = []

        start_time = time.time()
        for i in range(num_iterations):
            if time.time() - start_time > max_time:
                break

            node = root
            path = [node]
            while node.children and not node.is_terminal:
                node = node.best_child()
                path.append(node)

            if not node.is_fully_expanded():
                new_prompt = self.generate_prompt_variations(base_prompt, i)
                new_query = self.llm.generate_sql(new_prompt, temperature=0.7 + 0.05 * i)

                if new_query not in [child.sql_query for child in node.children]:
                    node = node.add_child(new_query)
                    path.append(node)

            score = self.evaluator.evaluate_query(node.sql_query, question)
            iteration_scores.append(score)

            if score > best_score:
                best_score = score
                best_query = node.sql_query

            for n in path:
                n.visits += 1
                n.reward_sum += score

            print(f"[Iter {i+1}] Score={score:.3f}, Query={node.sql_query[:80]}...")

        metrics = {
            "best_score": best_score,
            "average_score": np.mean(iteration_scores),
            "score_improvement": iteration_scores[-1] - iteration_scores[0] if len(iteration_scores) > 1 else 0,
            "execution_time": time.time() - start_time,
            "iteration_scores": iteration_scores,
            "total_nodes_explored": sum(1 for _ in self._traverse_tree(root))
        }

        return best_query, metrics

    def _traverse_tree(self, node: MCTSNode):
        yield node
        for child in node.children:
            yield from self._traverse_tree(child)

    def simple_baseline(self, question: str) -> Tuple[str, dict]:
        start = time.time()
        prompt = self.create_base_prompt(question)
        query = self.llm.generate_sql(prompt)
        score = self.evaluator.evaluate_query(query, question)

        return query, {
            "score": score,
            "execution_time": time.time() - start,
            "method": "baseline"
        }
