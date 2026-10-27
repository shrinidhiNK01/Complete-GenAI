from dataclasses import dataclass

from app.agents.tools import safe_calculate, search_knowledge_base
from app.rag.pipeline import RAGPipeline


@dataclass
class AgentState:
    user_input: str
    route: str = ""
    observation: str = ""
    final_answer: str = ""


def route(state: AgentState) -> AgentState:
    lowered = state.user_input.lower()
    if any(word in lowered for word in ["calculate", "sum", "multiply", "+", "-", "*", "/"]):
        state.route = "calculator"
    elif any(word in lowered for word in ["policy", "refund", "shipping", "support", "rag", "deployment"]):
        state.route = "knowledge_base"
    else:
        state.route = "direct"
    return state


def act(state: AgentState, pipeline: RAGPipeline) -> AgentState:
    if state.route == "calculator":
        expression = state.user_input.replace("calculate", "").strip()
        state.observation = str(safe_calculate(expression))
    elif state.route == "knowledge_base":
        state.observation = search_knowledge_base(pipeline, state.user_input)
    else:
        state.observation = "No tool was needed. Give a short general answer."
    return state


def respond(state: AgentState) -> AgentState:
    if state.route == "calculator":
        state.final_answer = f"The calculated result is {state.observation}."
    elif state.route == "knowledge_base":
        state.final_answer = state.observation
    else:
        state.final_answer = "I can answer this directly, but for production use connect this route to an LLM."
    return state


def run_agent(user_input: str, pipeline: RAGPipeline) -> AgentState:
    state = AgentState(user_input=user_input)
    state = route(state)
    state = act(state, pipeline)
    state = respond(state)
    return state

