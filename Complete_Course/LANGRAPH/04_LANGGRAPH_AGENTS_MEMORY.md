# LangGraph, Agents, Workflows, and Memory

## 1. What Is LangGraph?

LangGraph is a framework for building stateful, long-running, reliable agent workflows.

It is useful when a simple chain is not enough.

Use LangGraph for:

- Multi-step workflows
- Agents with tools
- Conditional branching
- Loops
- Persistence
- Human approval
- Memory
- Streaming
- Time travel debugging
- Subgraphs
- Fault-tolerant execution

## 2. LangGraph Mental Model

```text
State = data carried through workflow
Node = function that reads/writes state
Edge = connection between nodes
Conditional edge = route based on state
Graph = whole workflow
Checkpointer = saves state over time
Thread = one conversation/session
```

## 3. State

State is the shared data structure passed through the graph.

Example:

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
```

Every node can read state and return updates.

## 4. Nodes

A node is a function.

Example:

```python
def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(d.page_content for d in docs)}
```

## 5. Edges

Edges decide flow.

Example:

```text
START -> retrieve -> generate -> END
```

## 6. Conditional Edges

Conditional edges choose the next node based on logic.

Example:

```text
If retrieval quality is good -> generate
If retrieval quality is bad -> rewrite query
```

## 7. Loops

Loops allow repeated attempts.

Example:

```text
retrieve -> grade -> rewrite -> retrieve
```

Be careful with loops. Always add a stopping condition.

## 8. Sequential Workflow

Steps happen one after another.

```text
Input -> classify -> retrieve -> answer
```

Use when the process is fixed.

## 9. Parallel Workflow

Multiple branches happen at the same time.

Example:

```text
Question
  -> search policy docs
  -> search FAQ
  -> search tickets
Merge results -> answer
```

Use when independent work can happen together.

## 10. Conditional Workflow

Route based on state.

Example:

```text
If question is about policy -> RAG
If question is about calculation -> calculator
If question is small talk -> direct answer
```

## 11. Iterative Workflow

Repeat until quality is acceptable.

Example:

```text
draft -> critique -> revise -> final
```

## 12. Agents in LangGraph

An agent can be represented as a graph:

```text
User -> model node
Model chooses:
  -> final answer
  -> tool node
Tool result -> model node
```

This loop continues until the model returns final answer or max steps are reached.

## 13. ReAct Pattern

ReAct means Reason + Act.

Flow:

```text
Thought -> Action -> Observation -> Thought -> Action -> Observation -> Final
```

Modern tool calling may hide the raw "Thought" text, but the same idea remains: the model decides actions based on observations.

## 14. Tools in LangGraph

Tools can be normal functions.

Examples:

- Search docs
- Query SQL
- Call weather API
- Send email
- Execute code
- Open ticket

Dangerous tools should require approval.

## 15. Persistence

Persistence saves graph state as checkpoints.

Why it matters:

- Resume interrupted runs
- Debug previous states
- Support memory
- Support human-in-the-loop
- Recover from crashes

In LangGraph, checkpoints are organized by thread.

Thread example:

```text
thread_id = "user-123-conversation-456"
```

## 16. Time Travel

Time travel means going back to a previous checkpoint.

Use cases:

- Debug why the agent made a mistake.
- Resume from an earlier step.
- Try alternate path.
- Inspect state at each node.

## 17. Human in the Loop

Human-in-the-loop means the graph pauses and waits.

Example:

```text
Agent drafts email
Pause for human review
Human edits or approves
Agent sends email
```

Use it for risky actions:

- Payments
- Deleting data
- Sending messages
- Legal output
- Medical advice
- Production database writes

## 18. Streaming

LangGraph can stream:

- LLM tokens
- Node updates
- Tool calls
- Intermediate state
- Final answer

Streaming improves user experience and helps debug long workflows.

## 19. Memory Concepts

Memory is not the same as RAG.

RAG retrieves external knowledge.

Memory stores information about the conversation, user, or task.

### Short-Term Memory

Short-term memory is scoped to a thread.

Example:

```text
Conversation history in current chat
```

In LangGraph, short-term memory is usually part of graph state and persisted with a checkpointer.

### Long-Term Memory

Long-term memory works across threads.

Example:

```text
User prefers Python examples.
User is preparing for GenAI interviews.
```

Long-term memories are usually stored in a store with namespaces and keys.

## 20. Memory Design

Good memory should be:

- Useful
- Accurate
- User-approved when sensitive
- Editable
- Deletable
- Scoped
- Not too noisy

Bad memory:

```text
Remember every sentence the user says forever.
```

Good memory:

```text
Remember stable preferences and facts that improve future help.
```

## 21. Conversation Summary Memory

Instead of storing all messages forever, summarize older messages.

Why:

- Saves tokens
- Keeps important context
- Avoids huge prompts

Risk:

- Summary can lose details
- Summary can contain errors

## 22. Subgraphs

Subgraph means a graph inside a graph.

Use subgraphs for reusable workflows.

Example:

```text
Main support agent
  -> refund subgraph
  -> troubleshooting subgraph
  -> escalation subgraph
```

## 23. Building a Chatbot with UI

Architecture:

```text
Streamlit/React UI
  -> backend endpoint
  -> LangGraph app
  -> checkpointer for memory
  -> stream response to UI
```

Important UI features:

- Chat history
- Loading state
- Streaming response
- Error message
- Reset thread
- Source citations for RAG

## 24. SQLite with LangGraph

SQLite can be used for local persistence.

Good for:

- Learning
- Local demos
- Small prototypes

For production, use durable DB options such as PostgreSQL where appropriate.

## 25. Observability

Track:

- State at each node
- Node latency
- Tool calls
- Retrieval results
- Model output
- Errors
- User feedback

Without observability, agent debugging becomes guesswork.

## 26. LangGraph Interview Answer

Say this:

```text
LangGraph is used to build reliable stateful agents and workflows. It models the application as a graph of nodes and edges. State flows through the graph, nodes update state, and edges decide the next step. It supports conditional routing, loops, persistence, human-in-the-loop, streaming, time travel, and memory. I would use LangGraph when I need more control than a simple LangChain chain or when the agent must run over multiple steps reliably.
```
