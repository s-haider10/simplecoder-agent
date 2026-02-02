# simplecoder/tools.py
# for tool functions and schemas.
# At minimum, you need tools to list, read, search, write, and edit source files.

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_output_path(path: str, output_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to the output directory.

    Args:
        path: The path provided by the tool call
        output_dir: The output directory to prefix (None = current directory)

    Returns:
        Resolved Path object
    """
    file_path = Path(path)

    # If path is absolute, use it directly (allows explicit absolute paths)
    if file_path.is_absolute():
        return file_path

    # If no output_dir specified or it's cwd, use path as-is
    if output_dir is None or output_dir == Path.cwd().resolve():
        return file_path

    # Prefix relative paths with output_dir
    return output_dir / file_path


# Tool schemas for Gemini function calling (OpenAI-compatible format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents. Use this to examine existing code or files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the current directory matching a glob pattern. Useful for exploring project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files (e.g., '**/*.py' for all Python files, '*.txt' for txt files in current dir)",
                        "default": "**/*.py"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return",
                        "default": 20
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing a specific text snippet with new text. The old_text must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace (must match exactly)"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The new text to replace old_text with"
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a text pattern in files. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for (case-insensitive substring match)"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern for files to search in (e.g., '**/*.py')",
                        "default": "**/*.py"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return",
                        "default": 20
                    }
                },
                "required": ["pattern"]
            }
        }
    }
]


def read_file(path: str) -> str:
    """Read and return file contents."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add some context about the file
        lines = content.count('\n') + 1
        return f"File: {path} ({lines} lines)\n\n{content}"

    except UnicodeDecodeError:
        return f"Error: File appears to be binary, cannot read: {path}"
    except PermissionError:
        return f"Error: Permission denied reading: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, output_dir: Optional[Path] = None) -> str:
    """Write content to file, creating directories as needed."""
    try:
        file_path = _resolve_output_path(path, output_dir)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        lines = content.count('\n') + 1
        return f"Successfully wrote {len(content)} characters ({lines} lines) to {file_path}"

    except PermissionError:
        return f"Error: Permission denied writing to: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_files(pattern: str = "**/*.py", limit: int = 20) -> str:
    """List files matching glob pattern."""
    try:
        files = list(Path(".").glob(pattern))

        # Sort by modification time (most recent first)
        files.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)

        # Apply limit
        files = files[:limit]

        if not files:
            return f"No files found matching pattern: {pattern}"

        result_lines = [f"Found {len(files)} files matching '{pattern}':"]
        for f in files:
            try:
                size = f.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                result_lines.append(f"  {f} ({size_str})")
            except Exception:
                result_lines.append(f"  {f}")

        return "\n".join(result_lines)

    except Exception as e:
        return f"Error listing files: {e}"


def edit_file(path: str, old_text: str, new_text: str, output_dir: Optional[Path] = None) -> str:
    """Replace text in file."""
    try:
        file_path = _resolve_output_path(path, output_dir)

        if not file_path.exists():
            return f"Error: File not found: {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_text not in content:
            # Provide helpful info about what's in the file
            preview = content[:500] + "..." if len(content) > 500 else content
            return f"Error: Text not found in {file_path}. The text to replace was not found.\n\nFile preview:\n{preview}"

        # Count occurrences
        count = content.count(old_text)

        # Replace only first occurrence
        new_content = content.replace(old_text, new_text, 1)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        if count > 1:
            return f"Successfully edited {file_path}. Replaced 1 of {count} occurrences."
        else:
            return f"Successfully edited {file_path}."

    except UnicodeDecodeError:
        return f"Error: File appears to be binary: {path}"
    except PermissionError:
        return f"Error: Permission denied editing: {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def search_files(pattern: str, file_pattern: str = "**/*.py", limit: int = 20) -> str:
    """Search for text pattern in files."""
    try:
        files = list(Path(".").glob(file_pattern))
        matches = []
        pattern_lower = pattern.lower()

        for file_path in files:
            if len(matches) >= limit:
                break

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern_lower in line.lower():
                            matches.append({
                                "file": str(file_path),
                                "line": line_num,
                                "content": line.strip()[:100]
                            })
                            if len(matches) >= limit:
                                break
            except (UnicodeDecodeError, PermissionError):
                continue

        if not matches:
            return f"No matches found for '{pattern}' in {file_pattern}"

        result_lines = [f"Found {len(matches)} matches for '{pattern}':"]
        for m in matches:
            result_lines.append(f"  {m['file']}:{m['line']}: {m['content']}")

        return "\n".join(result_lines)

    except Exception as e:
        return f"Error searching files: {e}"


def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    permission_manager,
    rag_instance=None,
    output_dir: Optional[Path] = None
) -> str:
    """Execute a tool with permission checking."""

    # Check permissions for file write operations
    if tool_name in ["write_file", "edit_file"]:
        path = args.get("path", "")
        resolved_path = _resolve_output_path(path, output_dir)

        # Check if trying to modify source code
        path_str = str(resolved_path)
        if "simplecoder/" in path_str or path_str.startswith("simplecoder") or "/simplecoder/" in path_str:
            return "I am not allowed by the creator to modify the source code."

        if not permission_manager.has_permission("write", str(resolved_path)):
            return f"Permission denied: Cannot write to {resolved_path}. The agent needs write permission for this file."

    # Check permissions for read operations (usually allowed)
    if tool_name == "read_file":
        path = args.get("path", "")
        if not permission_manager.has_permission("read", path):
            return f"Permission denied: Cannot read {path}. This file is in a protected location."

    # Dispatch to appropriate function
    if tool_name == "read_file":
        return read_file(args["path"])

    elif tool_name == "write_file":
        return write_file(args["path"], args["content"], output_dir)

    elif tool_name == "list_files":
        return list_files(
            args.get("pattern", "**/*.py"),
            args.get("limit", 20)
        )

    elif tool_name == "edit_file":
        return edit_file(
            args["path"],
            args["old_text"],
            args["new_text"],
            output_dir
        )

    elif tool_name == "search_files":
        return search_files(
            args["pattern"],
            args.get("file_pattern", "**/*.py"),
            args.get("limit", 20)
        )

    elif tool_name == "search_code":
        # RAG-based semantic search
        if rag_instance is None:
            return "Error: RAG not initialized. Use --use-rag flag to enable semantic code search."
        return search_code(args["query"], args.get("top_k", 5), rag_instance)

    else:
        return f"Unknown tool: {tool_name}"


def search_code(query: str, top_k: int = 5, rag_instance=None) -> str:
    """Search codebase using semantic embeddings (RAG)."""
    if not rag_instance:
        return "Error: RAG not initialized. Use --use-rag flag."

    results = rag_instance.search(query, top_k)

    if not results:
        return "No relevant code found for your query."

    output_lines = ["**Found Code Snippets:**\n"]
    for i, result in enumerate(results, 1):
        output_lines.append(f"{i}. **{result['name']}** ({result['type']})")
        output_lines.append(f"   File: `{result['path']}:{result['line']}`")
        output_lines.append(f"   Relevance: {result['score']:.2f}")
        code_preview = result['code'][:300] + "..." if len(result['code']) > 300 else result['code']
        output_lines.append(f"   ```python\n{code_preview}\n   ```\n")

    return "\n".join(output_lines)
