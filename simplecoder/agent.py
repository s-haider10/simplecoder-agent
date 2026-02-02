# simplecoder/agent.py
# for main agent logic. Use a ReAct loop.

import os
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from litellm import completion
from .tools import TOOLS, execute_tool
from .permissions import PermissionManager
from .context import ContextManager


# System prompt for the agent
SYSTEM_PROMPT = """You are SimpleCoder, a helpful coding assistant that helps users with programming tasks.

You have access to tools for reading, writing, editing, listing, and searching files. Use these tools to help users with their coding tasks.

When given a task:
1. Think about what you need to do
2. Use the available tools to accomplish the task
3. Provide clear feedback about what you did

Be concise but helpful. When you've completed a task, summarize what you did.

Available tools:
- read_file: Read the contents of a file
- write_file: Write content to a file (creates if needed)
- edit_file: Edit a file by replacing specific text
- list_files: List files matching a pattern
- search_files: Search for text patterns in files
"""


class Agent:
    """ReAct-style coding agent."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        max_iterations: int = 10,
        verbose: bool = False,
        use_planning: bool = False,
        use_rag: bool = False,
        rag_embedder: str = "gemini/gemini-embedding-001",
        rag_index_pattern: str = "**/*.py",
        output_dir: str = "output",
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_planning = use_planning
        self.use_rag = use_rag
        self.rag_embedder = rag_embedder
        self.rag_index_pattern = rag_index_pattern

        # Output directory for generated files
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            self.planner = TaskPlanner(model=model, verbose=verbose)

        if self.verbose:
            print(f"[AGENT] Initialized with model={model}, max_iterations={max_iterations}")
            print(f"[AGENT] Output directory: {self.output_dir}")
            if self.use_rag:
                print(f"[AGENT] RAG enabled with embedder={rag_embedder}")
            if self.use_planning:
                print("[AGENT] Task planning enabled")

    def run(self, task: str) -> str:
        """Execute a task using the ReAct loop."""
        if self.verbose:
            print(f"[AGENT] Task: {task}")

        # If planning is enabled, decompose the task first
        if self.use_planning and self.planner:
            subtasks = self.planner.decompose(task)

            if len(subtasks) > 1:
                if self.verbose:
                    print(f"[PLANNER] Decomposed into {len(subtasks)} subtasks")

                results = []
                for i, subtask in enumerate(subtasks, 1):
                    if self.verbose:
                        print(f"[PLANNER] Executing subtask {i}/{len(subtasks)}: {subtask}")

                    result = self._execute_task(subtask)
                    results.append(f"**Subtask {i}: {subtask}**\n{result}")

                return "\n\n---\n\n".join(results)
            else:
                # Single task, no decomposition needed
                return self._execute_task(task)
        else:
            return self._execute_task(task)

    def _execute_task(self, task: str) -> str:
        """Execute a single task with ReAct loop."""
        # Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}
        ]

        iteration = 0
        last_three_tools = []  # For cycle detection

        while iteration < self.max_iterations:
            iteration += 1

            # Get context (possibly summarized if too long)
            current_messages = self.context_manager.get_context_for_llm(messages)

            if self.verbose:
                token_estimate = self.context_manager.estimate_tokens(current_messages)
                print(f"[CONTEXT] Messages: {len(current_messages)}, Est. tokens: {token_estimate}")

            # Build tools list
            available_tools = self._get_available_tools()

            # Call LLM
            try:
                response = completion(
                    model=self.model,
                    messages=current_messages,
                    tools=available_tools,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=2048,
                )
            except Exception as e:
                error_msg = f"Error calling LLM: {e}"
                if self.verbose:
                    print(f"[ERROR] {error_msg}")
                return error_msg

            # Parse response
            choice = response.choices[0]
            assistant_message = choice.message
            finish_reason = choice.finish_reason

            if self.verbose:
                content_preview = (assistant_message.content or "")[:150]
                print(f"[ITERATION {iteration}] Finish reason: {finish_reason}")
                if content_preview:
                    print(f"[RESPONSE] {content_preview}...")

            # Check if we have tool calls
            tool_calls = getattr(assistant_message, 'tool_calls', None)

            # If no tool calls, we're done
            if not tool_calls:
                # Add final response to messages
                final_content = assistant_message.content or "Task completed."
                messages.append({"role": "assistant", "content": final_content})
                return final_content

            # Process tool calls
            # First, add the assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_id = tool_call.id

                # Parse arguments
                try:
                    if isinstance(tool_call.function.arguments, str):
                        tool_args = json.loads(tool_call.function.arguments)
                    else:
                        tool_args = tool_call.function.arguments or {}
                except json.JSONDecodeError as e:
                    tool_result = f"Error parsing tool arguments: {e}"
                    if self.verbose:
                        print(f"[TOOL ERROR] {tool_result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result
                    })
                    continue

                if self.verbose:
                    print(f"[TOOL] Executing: {tool_name}({tool_args})")

                # Cycle detection - check for identical tool calls (same name AND args)
                tool_call_signature = json.dumps({"name": tool_name, "args": tool_args}, sort_keys=True)
                last_three_tools.append(tool_call_signature)
                if len(last_three_tools) > 3:
                    last_three_tools.pop(0)

                if len(last_three_tools) == 3 and len(set(last_three_tools)) == 1:
                    return f"Agent appears stuck in a loop (called {tool_name} with same arguments 3 times). Stopping."

                # Execute the tool
                try:
                    tool_result = execute_tool(
                        tool_name,
                        tool_args,
                        self.permission_manager,
                        rag_instance=self.rag,
                        output_dir=self.output_dir
                    )
                except Exception as e:
                    tool_result = f"Error executing {tool_name}: {e}"
                    if self.verbose:
                        print(f"[TOOL ERROR] {tool_result}")

                if self.verbose:
                    result_preview = tool_result[:200] if tool_result else "(empty)"
                    print(f"[TOOL RESULT] {result_preview}...")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result
                })

        # Max iterations reached
        return f"Reached maximum iterations ({self.max_iterations}). The task may not be fully complete."

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
