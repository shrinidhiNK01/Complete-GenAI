# Model Context Protocol Complete Guide

## 1. What Is MCP?

MCP means Model Context Protocol.

It is a standard protocol for connecting AI applications to external tools and data sources.

Simple analogy:

```text
USB-C standardizes hardware connections.
MCP standardizes AI tool/data connections.
```

## 2. Why MCP Exists

Without MCP:

- Every AI app builds custom integrations.
- Every tool needs different connection logic.
- Tool discovery is inconsistent.
- Security and permissions are harder to standardize.

With MCP:

- Servers expose tools, resources, and prompts in a standard way.
- AI apps can discover and call them.
- Developers can build one MCP server and use it across multiple AI clients.

## 3. MCP Architecture

Main parts:

```text
MCP Host -> MCP Client -> MCP Server
```

### MCP Host

The AI application.

Examples:

- Claude Desktop
- Claude Code
- VS Code AI extension
- Custom AI assistant

The host coordinates everything.

### MCP Client

The client is created by the host to connect to one MCP server.

If a host connects to three MCP servers, it usually has three MCP clients.

### MCP Server

The server exposes capabilities.

Examples:

- Filesystem server
- Database server
- GitHub server
- Slack server
- Custom company API server

## 4. MCP Layers

MCP has two conceptual layers.

### Data Layer

Defines JSON-RPC messages and primitives:

- Tools
- Resources
- Prompts
- Lifecycle
- Notifications
- Sampling

### Transport Layer

Defines how messages move.

Common transports:

- STDIO for local servers
- Streamable HTTP for remote servers

## 5. JSON-RPC

MCP uses JSON-RPC style communication.

Basic idea:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

The server responds with result or error.

## 6. MCP Primitives

### Tools

Tools are actions the model can call.

Examples:

- `search_files`
- `query_database`
- `create_ticket`
- `send_message`

Tools usually have:

- Name
- Description
- Input schema
- Output

### Resources

Resources are data the model can read.

Examples:

- File contents
- Database schema
- API documentation
- Project config

Resources are not necessarily actions. They are context.

### Prompts

Prompts are reusable templates exposed by the server.

Example:

```text
Generate a bug report from this stack trace.
```

### Notifications

Notifications are one-way messages for updates.

Example:

```text
tool list changed
resource updated
progress update
```

### Sampling

Sampling lets an MCP server ask the host's model for a completion.

This is useful when a server needs LLM help but does not want to directly depend on a model provider.

## 7. MCP Lifecycle

Typical lifecycle:

1. Client connects to server.
2. Client and server initialize.
3. Capabilities are negotiated.
4. Client discovers tools/resources/prompts.
5. Host makes them available to the model.
6. Model requests tool call when needed.
7. Client sends tool call to server.
8. Server executes and returns result.
9. Connection eventually closes.

## 8. Local vs Remote MCP Servers

### Local MCP Server

Usually runs on your machine using STDIO.

Good for:

- Filesystem access
- Local scripts
- Developer tools
- Personal automation

### Remote MCP Server

Runs over HTTP.

Good for:

- SaaS tools
- Company APIs
- Shared services
- Multi-user systems

## 9. MCP Tool Example

Imagine a database MCP server exposing:

```text
Tool: query_database
Input: SQL query
Output: rows
```

The AI app discovers this tool. When the user asks:

```text
How many orders failed yesterday?
```

The model can choose to call `query_database`.

## 10. MCP Resource Example

The same server can expose:

```text
Resource: database_schema
```

The model reads the schema before writing SQL.

This reduces wrong table or column names.

## 11. MCP Prompt Example

The server can expose:

```text
Prompt: sql_analysis_prompt
```

This prompt can guide the assistant to:

- Inspect schema
- Write safe SQL
- Explain result
- Avoid destructive queries

## 12. MCP Security

MCP security is extremely important.

Risks:

- Tool can access sensitive data.
- Tool can write/delete data.
- Prompt injection can trick model into unsafe calls.
- Remote server can be malicious.

Best practices:

- Use least privilege.
- Require human approval for dangerous actions.
- Validate tool inputs.
- Log tool calls.
- Use authentication for remote servers.
- Do not expose secrets in tool results.
- Add permission boundaries.
- Treat retrieved content as untrusted.

## 13. MCP vs LangChain Tools

LangChain tools are tool abstractions inside your LangChain app.

MCP is a protocol for exposing tools/resources/prompts across AI applications.

Comparison:

```text
LangChain tool = local framework-level callable
MCP tool = standardized external capability discoverable by MCP clients
```

They can work together. A LangChain or LangGraph app can use MCP tools as external tools.

## 14. MCP vs API

An API is a general software interface.

MCP is an AI-focused protocol that makes tools and context discoverable in a model-friendly way.

MCP does not replace all APIs. It wraps or exposes APIs for AI applications.

## 15. When to Build an MCP Server

Build MCP server when:

- You want multiple AI apps to use the same tool.
- You need standardized discovery.
- You expose company data/tools to AI assistants.
- You want clean separation between AI host and tool logic.

Do not build MCP server when:

- A simple local function is enough.
- Only one small app uses it.
- You do not need discovery or cross-client support.

## 16. MCP Interview Answer

Say this:

```text
MCP, or Model Context Protocol, is a standard way for AI applications to connect with tools and data sources. The architecture has an MCP host, MCP clients, and MCP servers. Servers expose tools, resources, and prompts. The host discovers these capabilities through the client and makes them available to the model. MCP uses a JSON-RPC based data layer and transports such as STDIO for local servers and HTTP for remote servers. It is useful because it avoids custom integration work for every AI app and tool.
```
