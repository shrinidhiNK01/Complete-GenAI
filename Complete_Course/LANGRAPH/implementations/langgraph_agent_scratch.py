"""
LangGraph Agent from Scratch.
This script demonstrates:
1. State definition (global memory).
2. Nodes: Agent and Tool Execution functions.
3. Edges: Conditional routing based on tool calls.
4. Compilation with Memory Checkpoint Saver.

Run: python Complete_Course/LANGRAPH/implementations/langgraph_agent_scratch.py
"""

import os
from typing import TypedDict, Annotated, Sequence
import operator

# Try importing LangGraph components
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# 1. Define Graph State Schema
# Annotated[Sequence[BaseMessage], operator.add] tells LangGraph to append new messages
# to the list rather than overwriting it when nodes return messages.
if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
else:
    AgentState = dict


# 2. Define Tools
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression string."""
    try:
        # Dangerous in production, but okay for a closed sandboxed demo
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


# 3. Define the Node Functions
def call_agent(state):
    """The Agent node: decides what to do next based on conversation history."""
    messages = state["messages"]
    last_message = messages[-1]
    
    print(f"\n[Agent Node] Reviewing last message: '{last_message.content}'")
    
    # Simulating LLM logic: if the prompt asks for math, return a Tool Call proposal
    if "calculate" in last_message.content.lower() or "+" in last_message.content or "*" in last_message.content:
        # Simulate LLM outputting a structured tool call
        print("[Agent Node] Decision: I need to use the calculator tool.")
        simulated_response = AIMessage(
            content="I will calculate that for you.",
            tool_calls=[{
                "name": "calculator",
                "args": {"expression": "2 * 8 + 4"},
                "id": "call_123"
            }]
        )
    else:
        print("[Agent Node] Decision: No tools needed. I can answer directly.")
        simulated_response = AIMessage(content="Hello! I am an assistant. How can I help you today?")
        
    return {"messages": [simulated_response]}


def execute_tools(state):
    """The Tools execution node: fires proposed tools and returns observations."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_outputs = []
    # If the last message contains tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"[Tools Node] Executing '{tool_call['name']}' with args: {tool_call['args']}")
            
            if tool_call["name"] == "calculator":
                result = calculator_tool(tool_call["args"]["expression"])
            else:
                result = f"Tool '{tool_call['name']}' not found."
                
            # Create a ToolMessage to report the observation back to the agent
            tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
            tool_outputs.append(tool_msg)
            
    return {"messages": tool_outputs}


# 4. Define Conditional Routing Edge
def should_continue(state):
    """Decides if the graph should route to the tools executor node or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message contains tool calls, route to 'tools'
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, stop
    return "end"


def main():
    if not LANGGRAPH_AVAILABLE:
        print("[WARNING] 'langgraph' or 'langchain' is not installed.")
        print("Run: pip install langgraph langchain-core")
        print("\nShowing simulation of the Compiled State Graph Workflow:")
        print("  1. State initialized: {'messages': [HumanMessage('Calculate 2 * 8 + 4')]}" )
        print("  2. Route to Node: agent")
        print("  3. Agent output: AIMessage(tool_calls=[calculator('2 * 8 + 4')])")
        print("  4. Conditional Edge splits -> routes to: tools")
        print("  5. Tools output: ToolMessage('20')")
        print("  6. Route back to Node: agent")
        print("  7. Agent outputs final answer: AIMessage('The result is 20.')")
        print("  8. Conditional Edge splits -> routes to: END")
        return

    # Initialize the Graph Builder
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", execute_tools)

    # Set Entry Point
    workflow.set_entry_point("agent")

    # Add Normal Edges
    # After tools are executed, always return to the agent to interpret the observations
    workflow.add_edge("tools", "agent")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Add Memory Checkpointer (to store conversational state across threads)
    memory = MemorySaver()

    # Compile the Graph
    app = workflow.compile(checkpointer=memory)
    print("State Graph compiled successfully.")

    # Run query
    config = {"configurable": {"thread_id": "session_1"}}
    query = HumanMessage(content="Hello assistant, please calculate 2 * 8 + 4 for me.")
    
    print("\n--- Starting Graph Run ---")
    events = app.stream({"messages": [query]}, config)
    for event in events:
        for node_name, state_update in event.items():
            print(f"Node '{node_name}' finished. Current messages count: {len(state_update.get('messages', []))}")

    # Read final state
    final_state = app.get_state(config)
    print("\n--- Final Message History ---")
    for msg in final_state.values["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")

if __name__ == '__main__':
    main()
