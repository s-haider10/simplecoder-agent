# SimpleCoder

A ReAct-style CLI coding agent that solves programming tasks through iterative reasoning, deterministic file editing, and optional semantic code search. Built on litellm for LLM access, with a focus on reliable edits and manageable context.

---

## How It Works at a Glance

```
User input
    │
    ▼
┌─ Planning (optional) ─────────────────────┐
│  Heuristic check → LLM decomposition      │
│  Complex tasks split into subtasks         │
│  Context persists across subtasks          │
└───────────────────────────────────────────┘
    │
    ▼
┌─ ReAct Loop ──────────────────────────────┐
│  1. Reason  – LLM decides what to do      │
│  2. Act     – Execute a tool              │
│  3. Observe – Result added to context     │
│  4. Repeat  – Until done or max iters     │
│                                           │
│  Guardrails:                              │
│    • Cycle detection (3 identical calls)  │
│    • Max 10 iterations                    │
│    • Permission checks on every file op   │
│    • Context summarization when needed    │
└───────────────────────────────────────────┘
    │
    ▼
Final response displayed to user
```

The agent treats the LLM as a reasoning engine inside an execution loop—not a single-shot code generator. Each iteration the LLM sees the results of what it just did before deciding the next step.

---

## Design Principles

### Deterministic Edits, Not Text Matching

The single most common failure mode in LLM-based editing is whitespace hallucination: the model tries to match a text string it remembers, gets the indentation wrong, and the edit silently fails or corrupts the file. SimpleCoder avoids this entirely.

**How edits actually work:**

```
1. read_file("app.py")          → Agent sees exact content with line numbers
2. Identify lines to change     → e.g., lines 12–15
3. replace_lines("app.py", 12, 15, new_code)  → Replaces by position, not by text match
4. read_file("app.py") again    → Verify the result
```

Line numbers are stable identifiers. The agent is instructed (via system prompt) to always read before editing, and the tool itself enforces line-number-based replacement. There is no fuzzy string matching anywhere in the edit path.

### Atomic File Writes

Every write (create or edit) goes through a temp file first, then an atomic rename. If the process crashes mid-write, the original file is untouched. No partial or corrupted states.

### Keeping Context Manageable

LLM context windows are finite. A coding agent that keeps appending full file contents and tool outputs will eventually overflow or slow down. SimpleCoder addresses this at multiple layers:

- **Large files are gated:** `read_file` on a file > 500 lines requires specifying a line range. This prevents accidentally dumping an entire codebase into context.
- **Result limits:** `list_files` and `search_files` cap output at 20 results by default.
- **RAG previews are short:** Semantic search returns ~300 characters of code per result, not full source.
- **History summarization:** When conversation history exceeds ~4000 tokens, older messages are extractively summarized. The last 8 messages stay intact (the agent needs recent context to act). Code snippets in older messages are preserved; explanatory text is condensed.
- **Subtask summaries:** In planning mode, each completed subtask's full trace is replaced with a 500-character summary before the next subtask runs. The agent remembers _what happened_, not every line of output.

This is an application of the **recognition over recall** principle: the agent doesn't need to hold everything in memory simultaneously. It needs enough to recognize what it already did and act on it.

### Progressive Disclosure in the CLI

Not every user needs every feature on every invocation. The interactive mode defaults to a minimal prompt. RAG and planning are off by default. Verbose output is off. The settings menu is only shown if you explicitly request it (type `\` after your input, or end your input with `\`). Features that aren't activated are never initialized—no embedding indexing runs unless you ask for it.

---

## Installation

```bash
git clone <repo-url>
cd simplecoder-agent

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

Or with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set your API key in `simplecoder/.env`:

```
GEMINI_API_KEY="your-key-here"
```

---

## Usage

### Interactive Mode

```bash
python -m simplecoder.main
```

The agent runs in a loop. Type a task, get a response, repeat. Context carries over between turns within the session.

Special inputs in interactive mode:

| Input                             | What it does                                |
| --------------------------------- | ------------------------------------------- |
| `help`                            | Shows available commands                    |
| `\` (alone)                       | Opens the settings menu                     |
| `your task\` (trailing backslash) | Opens settings menu before running the task |
| `exit` / `quit` / `q`             | Exits                                       |

The settings menu shows three toggles—RAG, planning, verbose—each with three states: ON, auto, OFF. In auto mode, the agent uses keyword heuristics to decide whether the feature is useful for a given input. You can lock them on or off if you prefer.

### Single Task Mode

```bash
python -m simplecoder.main "create a fibonacci function in fib.py"
```

Runs once and exits. Flags can be passed explicitly:

```bash
# Enable semantic code search
python -m simplecoder.main --use-rag "where is the login handler?"

# Enable task decomposition
python -m simplecoder.main --use-planning "create a calculator module and write tests for it"

# Both, plus verbose output
python -m simplecoder.main --use-rag --use-planning --verbose "refactor the auth layer"
```

### All CLI Options

| Flag                               | Default                         | Description                           |
| ---------------------------------- | ------------------------------- | ------------------------------------- |
| `--model`                          | `gemini/gemini-3-flash-preview` | Which LLM to use                      |
| `--max-iterations`                 | `10`                            | Max ReAct loop iterations per task    |
| `--verbose`                        | off                             | Show agent reasoning and tool details |
| `--interactive / --no-interactive` | on                              | Interactive loop vs single-shot       |
| `--use-planning`                   | off                             | Decompose complex tasks into subtasks |
| `--use-rag`                        | off                             | Enable semantic code search           |
| `--rag-embedder`                   | `gemini/gemini-embedding-001`   | Embedding model                       |
| `--rag-index-pattern`              | `**/*.py`                       | Which files to index for search       |

---

## The Modules

```
simplecoder/
├── main.py          # CLI, interactive loop, intent heuristics, settings menu
├── agent.py         # ReAct loop, streaming, cycle detection, subtask orchestration
├── tools.py         # File operations (read, write, replace_lines, list, search)
├── context.py       # Token estimation, history summarization
├── planner.py       # Task decomposition (heuristic + LLM)
├── rag.py           # AST chunking, embedding search, keyword fallback
└── permissions.py   # File access control, path validation
```

### Agent (`agent.py`)

The core loop. Each iteration:

1. Passes current message history to the LLM (via litellm, `temperature=0.1`)
2. Streams the response—a spinner updates in real time showing what tool the agent is about to use
3. If the LLM chose a tool, executes it, adds the result to history, loops back
4. If no tool call, returns the response as the final answer

Cycle detection: if the agent calls the exact same tool with the exact same arguments three times in a row, it stops and reports the loop. Max iterations (default 10) is a hard ceiling regardless.

When planning is enabled, subtasks share a single message history. After each subtask completes, the detailed tool traces are trimmed and replaced with a short summary. This keeps the context coherent without growing unboundedly.

### Tools (`tools.py`)

| Tool            | Notes                                                                  |
| --------------- | ---------------------------------------------------------------------- |
| `read_file`     | Returns content with line numbers. Files > 500 lines require a range.  |
| `write_file`    | Atomic: writes to temp, then renames. Creates parent dirs if needed.   |
| `replace_lines` | Line-number based. Returns a unified diff of the change. Atomic write. |
| `list_files`    | Glob pattern, sorted by modification time, capped at 20.               |
| `search_files`  | Case-insensitive substring search across files.                        |
| `search_code`   | Semantic search via RAG (only available when `--use-rag` is on).       |

All paths are normalized (repeated `output/` prefixes are stripped) and validated against the project root via symlink-resolved path checking.

### RAG (`rag.py`)

Semantic code search, built around AST parsing rather than naive text splitting.

**Indexing:** Python files are parsed with `ast`. Functions, async functions, and classes are extracted as chunks—each with its name, source, line numbers, and docstring if present. These chunks are embedded using a Gemini embedding model and cached to `.simplecoder/embeddings.pkl`. On subsequent runs, the cache is reused (note: the cache does not currently check file modification times, so clear it manually if your codebase changes significantly).

**Searching:** Your query is embedded, then compared to all chunk embeddings via cosine similarity. Top results are returned with relevance scores.

**Fallback:** If embedding fails for any reason, the system falls back to keyword matching: it scores chunks by how many query words appear in their name, code, and docstring.

### Context Manager (`context.py`)

Tracks estimated token usage (~1 token per 2.5 characters, which is conservative relative to the naive length/4 estimate). When total context exceeds ~4000 tokens and there are enough messages to summarize, it compacts history:

- System message and the first user message are always kept
- The most recent 8 messages are kept intact
- Everything in between is extractively summarized: code blocks are preserved verbatim, explanations are condensed
- Message ordering is carefully managed to avoid breaking Gemini's requirement that tool-result messages immediately follow the assistant message that produced them

### Planner (`planner.py`)

Decides whether a task needs decomposition, and if so, how to break it up.

**Simple task heuristic:** If the input is under 50 characters and contains none of the keywords `and`, `with`, `then`, `also`, `multiple`, `several`, `create`, `build`, `implement`, `refactor`, `update`, `fix`—it's treated as simple and executed directly.

**Complex tasks:** The LLM is asked to produce 2–4 numbered subtasks. If parsing that output fails, a fallback splits on `and` or `then`. The planner uses `temperature=0.5` (higher than the agent's 0.1) to allow some variation in how it structures the breakdown.

### Permissions (`permissions.py`)

Controls what the agent can read and write.

**Always blocked:** `.git/`, `.env`, `.venv/`, `__pycache__/`, `node_modules/`, `.ssh/`, `*.pyc`, and the agent's own source (`simplecoder/`).

**Reads** are allowed by default unless explicitly denied. **Writes** require the target to be within the current working directory (auto-allowed) or explicitly permitted. Permissions persist for the session via a file in `~/.simplecoder/`.

Path traversal is prevented by resolving symlinks before checking that the final path is within the project root.

---

## How Features Get Activated (Auto Mode)

When a toggle is set to "auto" in the settings menu, the agent uses simple keyword matching to decide whether to enable it for a given input:

- **RAG triggers on:** find, search, where, locate, look for, what is, how does, understand, explain, structure, show me
- **Planning triggers on:** build, create app, implement, develop, make a, set up, scaffold, generate, design

This is heuristic-based, not AI-driven. It will occasionally misfire (e.g., "find me a tutorial" would trigger RAG). If that bothers you, lock the toggles to ON or OFF instead of auto.

---

## Troubleshooting

**Agent seems stuck in a loop**
Cycle detection should catch this (3 identical tool calls → halt). If it doesn't trigger, max iterations (default 10) will. Try rephrasing or breaking the task into smaller pieces.

**Edits are producing wrong results**
The agent should always read the file first to get line numbers, then use `replace_lines`. If it's skipping the read step, try adding "read the file first" to your instruction, or use `--verbose` to see what it's actually doing.

**RAG isn't finding relevant code**
Clear the cache: `rm .simplecoder/embeddings.pkl`. Re-run with `--use-rag`. If your files aren't `.py`, adjust `--rag-index-pattern`.

**Context feels like it's being lost between steps**
In planning mode, subtask traces are intentionally summarized to 500 characters each. The agent keeps a summary of what happened, not the full transcript. If you need full continuity, consider running subtasks manually as separate interactive inputs.

---

## What's Not Here (Known Gaps)

- No file modification tracking in the RAG cache (stale embeddings if files change)
- No rollback or undo for edits
- No confirmation prompts before writes (the agent acts directly)
- No cost or token-usage display
- The `--output-dir` flag is accepted but currently has no effect

---

## License

MIT
