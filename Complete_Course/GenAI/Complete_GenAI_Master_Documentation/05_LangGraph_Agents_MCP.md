# LangGraph, Agents, and Model Context Protocol (MCP)

This document provides a comprehensive, beginner-to-advanced walkthrough of Agentic AI, state management in LangGraph, Human-in-the-Loop design patterns, and the Model Context Protocol (MCP) specification.

---

## 1. What is Agentic AI?

Traditional pipelines feed inputs through fixed chains. **Agentic AI** uses the LLM as a brain to make dynamic decisions:

*   **ReAct Pattern:** The agent loops through **Reasoning** ("Thought") $\to$ **Acting** ("Action" calling a tool) $\to$ **Observing** ("Observation" of tool output) until it determines it has solved the query.
*   **Cyclic workflows:** Unlike chains, agents must be able to loop back to previous steps (e.g. if a tool returns an error, the agent corrects its parameters and tries again).

---

## 2. LangGraph Fundamentals

LangGraph is a framework for building stateful, multi-actor applications with LLMs, modeling workflows as graphs.

```text
               [ State (TypedDict + Reducers) ]
               
         Node A (Node) ====> Normal Edge ====> Node B (Node)
                                                 │
                                           Conditional Edge
                                           /            \
                                       (Tools)         (Finish)
```

### Core Primitives
1.  **State:** The shared database of the graph. It is defined as a class. Nodes write updates to the state.
    *   *Reducers:* Define how new data is merged. E.g., `messages: Annotated[list, add]` tells the state to append new messages to the list rather than overwriting it.
2.  **Nodes:** Python functions that execute work. They take the current state as input, perform actions, and return a dictionary of updates to write back to the state.
3.  **Edges:** Define the connection lines between nodes.
    *   *Normal Edges:* Connect Node A directly to Node B.
    *   *Conditional Edges:* Run a router function to inspect state and select which node to execute next.

---

## 3. Advanced LangGraph Patterns

### 1. State Persistence & Checkpointing
By adding a checkpointer (like `MemorySaver` or `SqliteSaver`) when compiling the graph, LangGraph automatically writes a snapshot of the state database after every node execution.
*   **Conversational Memory:** The agent can remember past conversation turns across separate runs simply by using a consistent `thread_id`.
*   **State Rewinding (Time Travel):** You can query the history of a thread, inspect the state at step 2, and fork (rewind) the conversation timeline starting from that past checkpoint.

### 2. Human-in-the-Loop (HITL)
LangGraph allows you to place **breakpoints** that interrupt execution *before* a node runs. This is critical for security:

```text
[ Agent Proposes Action ] ===> Interrupt & Pause ===> [ Human Review ]
                                                            │
                                                  (Approve / Edit / Deny)
                                                            │
                                                            ▼
[ Execute Node ] <=========================================/
```

1.  **Approval Loops:** Pause before running tool nodes that perform destructive actions (e.g., executing SQL commands or sending emails).
2.  **State Modification:** A human can inspect the proposed tool parameters, rewrite them, and resume the graph execution.

---

## 4. Model Context Protocol (MCP)

Introduced by Anthropic, the **Model Context Protocol (MCP)** is an open-source standard that unifies how LLMs connect to data sources and tools.

### The Problem
Previously, if you wanted an LLM to read a Postgres database, a Slack channel, and a Github repository, you had to write custom API connectors for each model, IDE, and platform.

### The Solution: MCP
MCP standardizes this connection with a client-server architecture:

```text
                  [ MCP Client ]  (e.g., Claude Desktop, Cursor, IDE)
                   /     |      \
                  /      |       \  (Standard JSON-RPC protocol)
                 v       v        v
          Postgres     Slack    GitHub   [ MCP Servers ]
```

*   **MCP Host (Client):** The application using the LLM (like Claude Desktop, Cursor, or your custom agent framework).
*   **MCP Server:** A lightweight service exposing resources, prompts, and tools.
*   **Transport layer:** Clients and servers communicate using Standard Input/Output (stdio) or HTTP with Server-Sent Events (SSE).

### Three Core Capabilities of MCP
1.  **Resources:** Read-only data sources that the model can query (e.g., file contents, database schemas, log files).
2.  **Prompts:** Pre-defined templates for instructions or conversation structures.
3.  **Tools:** Executable functions that the model can request to run (e.g., execute code, run shell commands, write to database).
