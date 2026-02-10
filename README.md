# SimpleCoder

A ReAct-style CLI coding agent that solves programming tasks through iterative reasoning, deterministic file editing, and optional semantic code search.

**Demo:** [Watch on YouTube](https://www.youtube.com/watch?v=dQ4qWciEOt8)

**Quick Setup:**

```bash
git clone <repo-url> && cd simplecoder-agent
pip install -r requirements.txt
echo 'GEMINI_API_KEY="your-key"' > simplecoder/.env
python -m simplecoder.main
```

See [Installation](#installation) for detailed setup.

---

## How It Works

```
User input
    │
    ▼
┌─ Planning (optional) ─────────────────────┐
│  Heuristic check → LLM decomposition      │
│  Complex tasks split into subtasks         │
└───────────────────────────────────────────┘
    │
    ▼
┌─ ReAct Loop ──────────────────────────────┐
│  1. Reason  – LLM decides what to do      │
│  2. Act     – Execute a tool              │
│  3. Observe – Result added to context     │
│  4. Repeat  – Until done or max iters     │
└───────────────────────────────────────────┘
    │
    ▼
Final response displayed to user
```

The agent treats the LLM as a reasoning engine inside an execution loop—not a single-shot code generator.

---

## Design Approaches

### Agents

The agent implements a **ReAct (Reason + Act) loop** where the LLM iteratively decides actions based on accumulated observations. Built on **LiteLLM** for provider abstraction, each iteration streams the LLM response in real-time, showing the user which tool the agent is deciding to use before execution completes. The loop is bounded by a configurable `max_iterations` (default 10) to prevent runaway execution. When planning mode is enabled, complex tasks are decomposed into subtasks that share a persistent message history—allowing the agent to build on prior work while trimming verbose tool traces into concise summaries between subtasks.

### Tools

The agent exposes five deterministic tools: `read_file`, `write_file`, `replace_lines`, `list_files`, and `search_files`. The critical design decision is **line-number-based editing rather than text matching**—the most common failure mode in LLM editors is whitespace hallucination where the model misremembers indentation and corrupts files. By enforcing a strict protocol (read file → identify line numbers → replace by position → verify), edits become deterministic. All file writes use **atomic operations**: content goes to a temp file first, then an atomic rename, so a crash mid-write never corrupts the original. Result sets are capped (20 files max, 500-line read limits) to prevent context window explosion.

### RAG

The RAG (Retrieval Augmented Generation) system uses **AST-based chunking** rather than naive text splitting. Python files are parsed with the `ast` module to extract semantic units—functions, async functions, and classes—each preserving its name, docstring, line numbers, and source code. These chunks are embedded using Gemini's embedding model and cached to disk (pickle format) for reuse. Search computes **cosine similarity** between the query embedding and all chunk embeddings, returning top-k results. If embedding fails for any reason, the system **gracefully falls back to keyword matching**, scoring chunks by query word frequency in their name, code, and docstring.

### Context

Context management addresses the fundamental constraint that LLM context windows are finite. The system uses **conservative token estimation** (~1 token per 2.5 characters) to track usage. When history exceeds ~4000 tokens, older messages undergo **extractive summarization**: code blocks are preserved verbatim while explanatory text is condensed. The last 8 messages always remain intact since the agent needs recent context to act coherently. A critical implementation detail handles **Gemini's strict message ordering requirement**—tool result messages must immediately follow the assistant message that invoked them, so the compaction algorithm scans backwards to avoid breaking these pairs.

### Planner

The planner decides whether a task needs decomposition and how to break it up. A **heuristic check** first identifies simple tasks (under 50 characters, no complex keywords like "and", "implement", "refactor") which execute directly. Complex tasks are sent to the LLM with a prompt requesting 2-4 numbered subtasks. If LLM parsing fails, a **fallback heuristic** splits on conjunctions ("and", "then"). The key architectural choice is **persistent message history across subtasks**—each subtask sees accumulated context from prior work. After each subtask completes, verbose tool traces are trimmed and replaced with a 500-character summary, preserving _what happened_ without the full transcript bloating context.

### Permissions

The permission system provides defense-in-depth for file operations. **Session-based tracking** (keyed by working directory hash) persists permissions to `~/.simplecoder/`. System **deny patterns** block access to `.git/`, `.env`, `.venv/`, `__pycache__/`, `node_modules/`, `.ssh/`, and critically the agent's own source code (`simplecoder/`). Reads are allowed by default; writes require the path to be within the current working directory (auto-allowed) or explicitly permitted. **Path traversal prevention** resolves symlinks before checking that the final path stays within project bounds—a malicious path like `../../../etc/passwd` is caught and rejected.

---

## Responsive Interaction & User Feedback

The interface prioritizes keeping users informed of agent progress. During LLM calls, a **streaming spinner** updates in real-time to show which tool the agent is deciding to use _before_ execution begins. Each tool invocation displays a **narrative message** ("Reading file app.py...", "Editing file utils.py...") so users see exactly what's happening. In planning mode, **subtask progress** is shown with numbered headers and visual separators. All output uses **Rich console formatting** with colored panels, markdown rendering, and styled text to make responses scannable. Verbose mode adds thought panels showing the agent's reasoning before tool calls.

---

## Edge Cases Handled

- **Cycle detection**: If the agent calls the same tool with identical arguments 3 times consecutively, execution halts with an error message
- **Stuck-on-file recovery**: After reading the same file 3 times without making edits, a recovery prompt is injected forcing the agent to stop and plan
- **Large file gating**: Files over 500 lines require explicit line ranges to prevent dumping entire codebases into context
- **Atomic writes**: Temp file + rename pattern ensures crashes never leave corrupted files
- **Context overflow**: Extractive summarization kicks in before hitting token limits
- **Gemini tool pairing**: Compaction algorithm preserves assistant→tool message ordering
- **Path traversal**: Symlink resolution prevents directory escape attacks
- **RAG graceful fallback**: Embedding failures trigger keyword search instead of crashing

---

## Installation

```bash
git clone <repo-url>
cd simplecoder-agent

python -m venv .venv
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Set your API key in `simplecoder/.env`:

```
GEMINI_API_KEY="your-key-here"
```

---

## Usage

**Interactive mode:**

```bash
python -m simplecoder.main
```

**Single task:**

```bash
python -m simplecoder.main "create a fibonacci function in fib.py"
python -m simplecoder.main --use-rag "where is the login handler?"
python -m simplecoder.main --use-planning "build a calculator with tests"
```

Type `\` in interactive mode to open the settings menu.

---

## CLI Options

| Flag                  | Default                         | Description                     |
| --------------------- | ------------------------------- | ------------------------------- |
| `--model`             | `gemini/gemini-3-flash-preview` | LLM to use                      |
| `--max-iterations`    | `10`                            | Max ReAct loop iterations       |
| `--verbose`           | off                             | Show reasoning and tool details |
| `--interactive`       | on                              | Interactive loop vs single-shot |
| `--use-planning`      | auto/off                        | Decompose complex tasks         |
| `--use-rag`           | auto/off                        | Enable semantic code search     |
| `--rag-embedder`      | `gemini/gemini-embedding-001`   | Embedding model                 |
| `--rag-index-pattern` | `**/*.py`                       | Files to index                  |

---

## Modules

| File             | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `main.py`        | CLI, interactive loop, settings menu                 |
| `agent.py`       | ReAct loop, streaming, cycle detection               |
| `tools.py`       | File operations (read, write, replace, list, search) |
| `context.py`     | Token estimation, history summarization              |
| `planner.py`     | Task decomposition (heuristic + LLM)                 |
| `rag.py`         | AST chunking, embedding search, keyword fallback     |
| `permissions.py` | File access control, path validation                 |

---

## Known Gaps

- No rollback/undo for edits
- No confirmation prompts before writes
- No cost/token-usage display

---

## License

MIT
