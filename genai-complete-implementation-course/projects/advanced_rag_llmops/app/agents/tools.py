import ast
import operator
from typing import Callable, Dict

from app.rag.pipeline import RAGPipeline


ALLOWED_OPERATORS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def safe_calculate(expression: str) -> float:
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](eval_node(node.left), eval_node(node.right))
        raise ValueError("Only simple numeric expressions are allowed")

    parsed = ast.parse(expression, mode="eval")
    return float(eval_node(parsed))


def search_knowledge_base(pipeline: RAGPipeline, question: str) -> str:
    return pipeline.ask(question).answer

