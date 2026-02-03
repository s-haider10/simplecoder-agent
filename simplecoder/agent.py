# simplecoder/agent.py
# Deterministic ReAct-style coding agent with streaming feedback.

import os
import sys
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

import logging
from litellm import completion

# litellm prints "Give Feedback" / "LiteLLM.Info" lines to its own logger on errors — silence them
logging.getLogger("litellm").setLevel(logging.CRITICAL)

from .tools import TOOLS, execute_tool
from .permissions import PermissionManager
from .context import ContextManager


# =============================================================================
# SYSTEM PROMPT WITH EDIT PROTOCOL
# =============================================================================

SYSTEM_PROMPT = """You are SimpleCoder, a deterministic coding assistant.

# TOOLS AVAILABLE

- read_file(path): Read file with line numbers. ALWAYS call this before editing.
- write_file(path, content): Create or overwrite a file (atomic write).
- replace_lines(path, start_line, end_line, new_content): Edit file by line numbers.
- list_files(pattern): List files matching glob pattern.
- search_files(pattern): Search for text in files.

# EDIT PROTOCOL (MANDATORY)

When editing files, you MUST follow this exact workflow:

## Step 1: Read the file first
- Call: read_file(path)
- Output shows line numbers like: "   1 | def hello():"
- Purpose: See exact line numbers for editing

## Step 2: Identify lines to change
- Note the start_line and end_line (1-indexed, inclusive)
- Example: To change line 5, use start_line=5, end_line=5

## Step 3: Replace those lines
- Call: replace_lines(path, start_line, end_line, new_content)
- new_content replaces lines start_line through end_line
- Output shows unified diff of changes

## Step 4: Verify the edit
- Call: read_file(path) again to confirm changes

# CRITICAL RULES

1. NEVER guess line numbers. Always read_file first.
2. NEVER use text matching to find content. Use line numbers only.
3. NEVER rewrite entire files unless creating new ones.
4. Paths are relative to project root. Never use "output/" prefix.
5. After every edit, verify by reading the file again.

# EXAMPLES

GOOD workflow for editing:
1. read_file("app.py")           # See: "   5 |     print('old')"
2. replace_lines("app.py", 5, 5, "    print('new')")
3. read_file("app.py")           # Verify change

GOOD workflow for creating:
1. write_file("hello.py", "print('hello world')")
2. read_file("hello.py")         # Verify creation

BAD (will fail):
- replace_lines without reading first
- Guessing line numbers
- Using "output/file.py" instead of "file.py"

# BEHAVIOR

- Be concise. Use tools, don't describe what you would do.
- After completing a task, summarize what you did concisely and in a user friendly way.
- If a tool returns an error, read it carefully and adjust.
"""


class Agent:
    """Deterministic ReAct-style coding agent with streaming."""

    def __init__(
        self,
        model: str = "gemini/gemini-3-flash-preview",
        max_iterations: int = 10,
        verbose: bool = False,
        use_planning: bool = False,
        use_rag: bool = False,
        rag_embedder: str = "gemini/gemini-embedding-001",
        rag_index_pattern: str = "**/*.py",
        output_dir: str = "output",  # Kept for backward compatibility, ignored
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_planning = use_planning
        self.use_rag = use_rag
        self.rag_embedder = rag_embedder
        self.rag_index_pattern = rag_index_pattern
        self.console = Console()
        # Initialize core components
        self.permission_manager = PermissionManager()
        self.context_manager = ContextManager(verbose=verbose)

        # Initialize RAG if needed
        self.rag = None
        if self.use_rag:
            from .rag import RAG
            self.rag = RAG(embedder=rag_embedder, index_pattern=rag_index_pattern, verbose=verbose)

        # Initialize planner if needed
        self.planner = None
        if self.use_planning:
            from .planner import TaskPlanner
            self.planner = TaskPlanner(model="gemini/gemini-3-pro-preview", verbose=verbose)

        if self.verbose:
            # Clean initialization log
            model_name = model.split("/")[-1]
            self.console.print(f"[bold blue]Agent:[/bold blue] Initialized (Model: [cyan]{model_name}[/cyan])")
            if self.use_rag:
                self.console.print("[bold blue]Agent:[/bold blue] RAG enabled")
            if self.use_planning:
                self.console.print("[bold blue]Agent:[/bold blue] Planning enabled")

    def run(self, task: str) -> str:
        """Execute a task using the ReAct loop."""
        if self.verbose:
            self.console.print(f"[bold blue]Agent:[/bold blue] Task: [bold]{task}[/bold]")
            self.console.rule()

        # If planning is enabled, decompose the task first
        if self.use_planning and self.planner:
            return self._run_with_plan(task)
        else:
            return self._execute_task(task)

    def _run_with_plan(self, task: str) -> str:
        """Run task with persistent context across subtasks."""
        subtasks = self.planner.decompose(task)

        if len(subtasks) <= 1:
            # Single task, no decomposition needed
            return self._execute_task(task)

        if self.verbose:
            print(f"[PLANNER] Decomposed into {len(subtasks)} subtasks")

        # Persistent message history (NOT reset between subtasks)
        persistent_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Main task: {task}"}
        ]

        results = []

        for i, subtask in enumerate(subtasks, 1):
            if self.verbose:
                self.console.rule(f"[bold blue]Subtask {i}/{len(subtasks)}[/bold blue]")
                self.console.print(f"[bold]{subtask}[/bold]")

            # Snapshot length so we can trim the tool-call noise afterwards
            history_len_before = len(persistent_messages)

            persistent_messages.append({
                "role": "user",
                "content": f"[Subtask {i}/{len(subtasks)}] {subtask}"
            })

            # Execute with full history (context persists!)
            result = self._execute_task_with_messages(subtask, persistent_messages)

            if result.startswith("✗"):
                results.append(f"**Subtask {i}: {subtask}** — skipped")
            else:
                results.append(f"**Subtask {i}: {subtask}**\n{result}")

            # _execute_task_with_messages appended assistant+tool_calls and tool messages
            # in-place.  Gemini requires strict assistant[tool_calls]→tool pairing — if
            # compaction or the next subtask's user message breaks that ordering it errors.
            # Trim everything back to the snapshot and replace with a clean summary pair.
            del persistent_messages[history_len_before:]

            result_summary = result[:500] + "..." if len(result) > 500 else result
            persistent_messages.append({
                "role": "user",
                "content": f"[Subtask {i}/{len(subtasks)}] {subtask}"
            })
            persistent_messages.append({
                "role": "assistant",
                "content": result_summary
            })

        return "\n\n---\n\n".join(results)

    def _execute_task(self, task: str) -> str:
        """Execute a single task with ReAct loop."""
        # Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}
        ]
        return self._execute_task_with_messages(task, messages)

    def _execute_task_with_messages(self, task: str, messages: List[Dict[str, Any]]) -> str:
        """Execute task with given message history."""
        iteration = 0
        last_three_tools = []  # For cycle detection

        while iteration < self.max_iterations:
            iteration += 1

            # Get context (possibly summarized if too long)
            current_messages = self.context_manager.get_context_for_llm(messages)

            # Build tools list
            available_tools = self._get_available_tools()

            # Call LLM with streaming
            try:
                # Use a spinner that updates when the agent decides on a tool
                with self.console.status("[bold blue]Agent:[/bold blue] Thinking...", spinner="dots") as status:
                    response = completion(
                        model=self.model,
                        messages=current_messages,
                        tools=available_tools,
                        tool_choice="auto",
                        temperature=0.1,  # DETERMINISTIC
                        max_tokens=2048,
                        stream=True  # ENABLE STREAMING
                    )

                    # Collect streamed response
                    assistant_text = ""
                    tool_calls_data = []

                    for chunk in response:
                        if not hasattr(chunk, 'choices') or not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        # Stream text content
                        if hasattr(delta, 'content') and delta.content:
                            token = delta.content
                            assistant_text += token

                        # Collect tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                if tc_delta.index is not None:
                                    # Ensure we have enough slots
                                    while len(tool_calls_data) <= tc_delta.index:
                                        tool_calls_data.append({
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })

                                    tc = tool_calls_data[tc_delta.index]

                                    if tc_delta.id:
                                        tc["id"] = tc_delta.id
                                    if hasattr(tc_delta, 'function') and tc_delta.function:
                                        if tc_delta.function.name:
                                            tc["function"]["name"] = tc_delta.function.name
                                            # Update spinner to show the tool decision in real-time
                                            status.update(f"[bold blue]Agent:[/bold blue] Thinking... (deciding to use [cyan]{tc['function']['name']}[/cyan])")

                                        if tc_delta.function.arguments:
                                            tc["function"]["arguments"] += tc_delta.function.arguments

            except Exception as e:
                if self.verbose:
                    self.console.print(f"[dim]LLM error: {e}[/dim]")
                return "✗ LLM overhead"

            content = assistant_text

            # Check if we have tool calls
            tool_calls = [tc for tc in tool_calls_data if tc["id"] and tc["function"]["name"]]

            # If no tool calls, we're done
            if not tool_calls:
                if self.verbose:
                    self.console.print("[bold blue]Agent:[/bold blue] Finalising response...")
                final_content = content or "Task completed."
                messages.append({"role": "assistant", "content": final_content})
                return final_content

# Show thought if we have tool calls and verbose
            if tool_calls and content and self.verbose:
                self.console.print(Panel(Markdown(content), title="[bold]Thought[/bold]", border_style="yellow"))

            # Process tool calls
            # Add the assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            })

            # Execute each tool call
            # We use print here (not spinner) to keep a history of actions taken
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_id = tool_call["id"]

                # Parse arguments
                try:
                    raw_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as e:
                    tool_result = f"✗ Error parsing tool arguments: {e}"
                    if self.verbose:
                        self.console.print(f"[bold red]Tool Error:[/bold red] {tool_result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result
                    })
                    continue

                # Narrative tool usage — always shown so the user sees actions taken
                msg = f"Using tool [cyan]{tool_name}[/cyan]..."
                if tool_name == "read_file":
                    msg = f"Reading file [cyan]{tool_args.get('path', 'unknown')}[/cyan]..."
                elif tool_name == "write_file":
                    msg = f"Writing to file [cyan]{tool_args.get('path', 'unknown')}[/cyan]..."
                elif tool_name == "replace_lines":
                    msg = f"Editing file [cyan]{tool_args.get('path', 'unknown')}[/cyan]..."
                elif tool_name == "list_files":
                    msg = "Listing files..."
                elif tool_name == "search_files":
                    msg = f"Searching files for [cyan]'{tool_args.get('pattern', '')}'[/cyan]..."

                self.console.print(f"[bold blue]Agent:[/bold blue] {msg}")

                # Cycle detection - check for identical tool calls
                tool_call_signature = json.dumps({"name": tool_name, "args": tool_args}, sort_keys=True)
                last_three_tools.append(tool_call_signature)
                if len(last_three_tools) > 3:
                    last_three_tools.pop(0)

                if len(last_three_tools) == 3 and len(set(last_three_tools)) == 1:
                    return f"✗ Agent stuck in loop (called {tool_name} with same arguments 3 times). Stopping."

                # Execute the tool
                try:
                    tool_result = execute_tool(
                        tool_name,
                        tool_args,
                        self.permission_manager,
                        rag_instance=self.rag
                    )
                except Exception as e:
                    tool_result = f"✗ Error executing {tool_name}: {e}"
                    if self.verbose:
                        self.console.print(f"[bold red]Tool Error: {e}[/bold red]")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result
                })

        # Max iterations reached
        return f"✗ Reached maximum iterations ({self.max_iterations}). Task may not be complete."

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of tools based on configuration."""
        tools = TOOLS.copy()

        # Add RAG search tool if enabled
        if self.use_rag and self.rag:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search the codebase using semantic similarity. Finds relevant functions, classes, and code snippets based on meaning, not just keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of what code you're looking for"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            })

        return tools
