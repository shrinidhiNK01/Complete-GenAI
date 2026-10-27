"""
Multi-Agent Supervisor System in LangGraph.
This script demonstrates:
1. Orchestrator-worker pattern: A supervisor routes tasks to specialized workers.
2. Worker nodes: Research Agent and Writing Agent.
3. State routing: Passing control back to supervisor until task is complete.

Run: python Complete_Course/LANGRAPH/implementations/multi_agent_system.py
"""

from typing import TypedDict, Annotated, List
import operator

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

if LANGGRAPH_AVAILABLE:
    class MultiAgentState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next_agent: str  # Tracks who should run next: "supervisor", "researcher", "writer", or "end"
else:
    MultiAgentState = dict


# Node 1: Researcher Agent
def researcher_node(state):
    print("\n[Researcher Agent] Analyzing documents and extracting facts...")
    # Simulate extracting research details
    fact = AIMessage(content="[Research Fact]: The population of Mars in 2050 is projected to be 10,000 colonists.")
    return {
        "messages": [fact],
        "next_agent": "supervisor"  # Return control to supervisor
    }


# Node 2: Writer Agent
def writer_node(state):
    print("\n[Writer Agent] Converting facts into a polished summary paragraph...")
    # Simulate writing process
    summary = AIMessage(content="[Written Summary]: Humanity will establish its first major foothold on Mars, reaching 10,000 residents by 2050.")
    return {
        "messages": [summary],
        "next_agent": "supervisor"  # Return control to supervisor
    }


# Node 3: Supervisor Router Agent
def supervisor_node(state):
    messages = state["messages"]
    last_msg = messages[-1].content
    print(f"\n[Supervisor Agent] Reviewing current status: '{last_msg}'")

    # Routing decision logic
    if "[Research Fact]" not in last_msg and "[Written Summary]" not in last_msg:
        print("[Supervisor Agent] Decision: Let's start with research.")
        next_step = "researcher"
    elif "[Research Fact]" in last_msg:
        print("[Supervisor Agent] Decision: Research gathered. Sending to writer.")
        next_step = "writer"
    else:
        print("[Supervisor Agent] Decision: Writer finished. Terminating workflow.")
        next_step = "end"

    return {"next_agent": next_step}


# Conditional Edge Router
def route_next(state):
    return state["next_agent"]


def main():
    if not LANGGRAPH_AVAILABLE:
        print("[WARNING] 'langgraph' or 'langchain' is not installed.")
        print("Showing simulation of the Multi-Agent Supervisor workflow:")
        print("  1. User asks: 'Write a brief report about Mars population.'")
        print("  2. Supervisor Node runs -> decides we need background research. Returns next_agent='researcher'.")
        print("  3. Researcher Node runs -> finds Mars fact. Returns control to supervisor.")
        print("  4. Supervisor Node runs -> sees research data, decides we need writing. Returns next_agent='writer'.")
        print("  5. Writer Node runs -> drafts summary. Returns control to supervisor.")
        print("  6. Supervisor Node runs -> checks draft, determines completion. Returns next_agent='end' -> END.")
        return

    # Create State Graph
    builder = StateGraph(MultiAgentState)

    # Add agent nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)

    # Set Entry Point
    builder.set_entry_point("supervisor")

    # Add worker routes back to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")

    # Add conditional router edge from supervisor to workers/END
    builder.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "researcher": "researcher",
            "writer": "writer",
            "end": END
        }
    )

    app = builder.compile()
    print("Multi-Agent Graph compiled successfully.")

    # Run query
    query = HumanMessage(content="Create a short summary about the Mars population projections.")
    print("\n--- Starting Multi-Agent Session ---")
    state_output = app.invoke({"messages": [query], "next_agent": "supervisor"})
    
    print("\n--- Final Chat History ---")
    for msg in state_output["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")

if __name__ == '__main__':
    main()
