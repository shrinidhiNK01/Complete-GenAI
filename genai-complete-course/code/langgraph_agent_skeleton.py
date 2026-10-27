from typing import Literal, TypedDict


class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    route: str


def route_question(state: AgentState) -> AgentState:
    question = state["question"].lower()
    if "document" in question or "policy" in question:
        state["route"] = "retrieve"
    else:
        state["route"] = "answer_directly"
    return state


def retrieve_context(state: AgentState) -> AgentState:
    state["context"] = "Refunds are allowed within 30 days with proof of purchase."
    return state


def answer(state: AgentState) -> AgentState:
    if state.get("context"):
        state["answer"] = f"Using retrieved context: {state['context']}"
    else:
        state["answer"] = "This question can be answered directly by the model."
    return state


def choose_next(state: AgentState) -> Literal["retrieve", "answer"]:
    if state["route"] == "retrieve":
        return "retrieve"
    return "answer"


if __name__ == "__main__":
    state: AgentState = {
        "question": "What does the policy say about refunds?",
        "context": "",
        "answer": "",
        "route": "",
    }
    state = route_question(state)
    if choose_next(state) == "retrieve":
        state = retrieve_context(state)
    state = answer(state)
    print(state["answer"])

