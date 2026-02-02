# SimpleCoder

A ReAct-style CLI coding agent that helps with programming tasks using LLM-powered reasoning, tool use, and semantic code search.

## Features

### Core Capabilities

- **ReAct Agent Loop**: Iterative reasoning and acting cycle that thinks through problems step-by-step
- **File Operations**: Read, write, edit, list, and search files in your codebase
- **Semantic Code Search (RAG)**: AST-based code chunking with embedding search to find relevant code
- **Task Planning**: Automatic decomposition of complex tasks into manageable subtasks
- **Context Management**: Smart conversation history summarization to stay within token limits
- **Permission System**: Session-persistent file access controls with sensible defaults

### Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a file |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace specific text in a file |
| `list_files` | List files matching a glob pattern |
| `search_files` | Search for text patterns in files |
| `search_code` | Semantic code search using embeddings (requires `--use-rag`) |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd simplecoder-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Alternative: Using uv (faster)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the `simplecoder/` directory with your Gemini API key:

```
GEMINI_API_KEY="your-api-key-here"
```

## Usage

### Basic Commands

```bash
# Single task (non-interactive)
python -m simplecoder.main "create a hello.py file that prints hello world"

# Interactive mode (default)
python -m simplecoder.main

# With verbose output
python -m simplecoder.main --verbose "what files are in this project?"
```

### Advanced Options

```bash
# Enable task planning (decomposes complex tasks)
python -m simplecoder.main --use-planning "create a web server with home and about routes"

# Enable semantic code search
python -m simplecoder.main --use-rag "what functions handle user authentication?"

# Combine features
python -m simplecoder.main --use-rag --use-planning --verbose "refactor the login system"

# Use a different model
python -m simplecoder.main --model "gemini/gemini-2.0-flash" "explain this codebase"
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gemini/gemini-3-flash-preview` | LLM model to use |
| `--max-iterations` | `10` | Maximum ReAct loop iterations |
| `--verbose` | `False` | Enable detailed output |
| `--interactive/--no-interactive` | `True` | Interactive chat mode |
| `--use-planning` | `False` | Enable task decomposition |
| `--use-rag` | `False` | Enable semantic code search |
| `--rag-embedder` | `gemini/gemini-embedding-001` | Embedding model for RAG |
| `--rag-index-pattern` | `**/*.py` | File pattern for RAG indexing |

## Architecture

```
simplecoder/
├── main.py          # CLI entry point (Click-based)
├── agent.py         # ReAct agent loop and orchestration
├── tools.py         # File operation tools and schemas
├── context.py       # Token tracking and history summarization
├── planner.py       # Task decomposition logic
├── rag.py           # AST-based code chunking and embedding search
├── permissions.py   # File access permission management
└── .env             # API key configuration
```

### High-Level Logic

1. **User Input**: Task received via CLI argument or interactive prompt
2. **Planning** (optional): Complex tasks are decomposed into subtasks
3. **ReAct Loop**:
   - Agent receives task and available tools
   - LLM reasons about the task and decides which tool to use
   - Tool is executed with permission checking
   - Result is added to context
   - Loop continues until task is complete or max iterations reached
4. **Context Management**: When conversation gets long, older messages are summarized
5. **Output**: Final response displayed to user

### Key Components

#### Agent (agent.py)
The core ReAct loop that:
- Calls the LLM with the current context and available tools
- Parses tool calls from the response
- Executes tools through the permission system
- Detects stuck loops (same tool called 3x consecutively)
- Returns final response when no more tool calls needed

#### RAG System (rag.py)
Semantic code search using:
- **AST Chunking**: Parses Python files and extracts functions, classes, and async functions as semantic units
- **Embeddings**: Uses Gemini embeddings for vector similarity search
- **Caching**: Stores embeddings to disk (`.simplecoder/embeddings.pkl`) to avoid recomputation
- **Fallback**: Keyword search when embeddings fail

#### Context Manager (context.py)
Handles token limits by:
- Estimating tokens (~1 token per 4 characters)
- Triggering summarization when exceeding 6000 tokens
- Keeping last 8 messages intact
- Creating extractive summaries of older messages (preserves code snippets)

#### Task Planner (planner.py)
Decomposes complex tasks by:
- Heuristic check: Short tasks (<50 chars) without complex keywords are kept as-is
- LLM decomposition: Asks the model to break task into 2-4 subtasks
- Fallback: Splits on "and" or "then" conjunctions

#### Permission Manager (permissions.py)
Controls file access with:
- **Default deny patterns**: `.git`, `.env`, `__pycache__`, `node_modules`, etc.
- **Auto-allow**: Writes within current working directory
- **Session persistence**: Permissions saved to `~/.simplecoder/permissions_<session>.json`

## Examples

### Creating Files
```
You: create a fibonacci function in fib.py
Agent: I'll create a file with a fibonacci function...
[Uses write_file tool]
Successfully wrote fib.py with the fibonacci function.
```

### Exploring Codebase
```
You: what Python files exist in this project?
Agent: Let me list the Python files...
[Uses list_files tool]
Found 6 Python files: main.py, agent.py, tools.py, context.py, planner.py, rag.py
```

### Semantic Search (with --use-rag)
```
You: how does the agent handle tool calls?
Agent: Let me search for relevant code...
[Uses search_code tool]
Found the ReAct loop in agent.py that processes tool_calls from the LLM response...
```

### Multi-Step Tasks (with --use-planning)
```
You: create a calculator module with add, subtract, and multiply functions, then test it
Agent: I'll break this into subtasks...
[Subtask 1: Create calculator.py with functions]
[Subtask 2: Create test file]
[Subtask 3: Verify implementation]
All subtasks completed successfully.
```

## Troubleshooting

### API Key Issues
- Ensure `GEMINI_API_KEY` is set in `simplecoder/.env`
- Check the key is valid at Google AI Studio

### RAG Not Finding Code
- Run with `--verbose` to see indexing progress
- Clear cache: delete `.simplecoder/embeddings.pkl`
- Check `--rag-index-pattern` matches your files

### Agent Stuck in Loop
- The agent auto-detects when it calls the same tool 3 times and stops
- Try rephrasing your request or breaking it into smaller tasks

## License

MIT
