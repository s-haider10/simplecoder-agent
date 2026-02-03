# simplecoder/tools.py
# Deterministic file operations with atomic writes and line-based editing.

import json
import tempfile
import shutil
import difflib
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# PATH UTILITIES
# =============================================================================

def _normalize_path(path: str) -> str:
    """
    Remove any leading 'output/' prefix to prevent path doubling.

    Examples:
        "hello.py" → "hello.py"
        "output/hello.py" → "hello.py"
        "output/output/hello.py" → "hello.py"
    """
    path = path.lstrip('/')

    # Strip ALL leading "output/" prefixes
    while path.startswith('output/'):
        path = path[7:]  # len('output/') == 7

    return path


def _find_project_root() -> Path:
    """
    Find the project root by searching for .git or pyproject.toml.

    Returns the first parent directory containing either marker.
    Defaults to current directory if not found.
    """
    current = Path.cwd()

    for parent in [current, *current.parents]:
        if (parent / '.git').exists():
            return parent
        if (parent / 'pyproject.toml').exists():
            return parent

    return current


def _validate_path_safety(path: str, base_dir: Path) -> bool:
    """
    Strict security check: ensure path doesn't escape base_dir via symlinks.

    Returns True if path is safely within base_dir.
    """
    try:
        resolved = (base_dir / path).resolve()
        base_resolved = base_dir.resolve()
        resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False


# =============================================================================
# TOOL SCHEMAS (OpenAI-compatible format)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file with line numbers. MUST call this before replace_lines to see current content and get exact line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (relative to project root, no 'output/' prefix needed)"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line (1-indexed), optional"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line (1-indexed), optional"
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
            "description": "Create or overwrite a file. Uses atomic write (safe from corruption). Path is relative to project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (no 'output/' prefix needed)"
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
            "name": "replace_lines",
            "description": "Replace lines deterministically using line numbers (NOT text matching). ALWAYS read_file first to see line numbers. Line numbers are 1-indexed and inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to replace (1-indexed, inclusive)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to replace (1-indexed, inclusive)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to replace the lines with (can be multiline)"
                    }
                },
                "required": ["path", "start_line", "end_line", "new_content"]
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
                        "description": "Glob pattern to match files (e.g., '**/*.py' for all Python files)",
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


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def read_file(path: str, start_line: int = None, end_line: int = None, permission_manager=None) -> str:
    """
    Read file content with line numbers for easy reference.

    This allows agent to:
    1. See line numbers
    2. Know exact line range for replace_lines()

    Args:
        path: File to read
        start_line: Optional starting line (1-indexed)
        end_line: Optional ending line (1-indexed)
        permission_manager: Access control

    Returns:
        Content with line numbers, or error message
    """
    # Step 1: Normalize path
    clean_path = _normalize_path(path)

    # Step 2: Find file
    project_root = _find_project_root()
    if not _validate_path_safety(clean_path, project_root):
        return f"✗ Path {clean_path} is outside project"

    full_path = project_root / clean_path

    if not full_path.exists():
        return f"✗ File not found: {clean_path}"

    try:
        # Step 3: Read file
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Step 4: Prevent context window explosion
        if len(lines) > 500 and start_line is None:
            return (
                f"✗ File too large ({len(lines)} lines)\n\n"
                f"Use start_line/end_line to read a section:\n"
                f"  read_file(path, start_line=1, end_line=50)\n"
                f"  read_file(path, start_line=100, end_line=150)"
            )

        # Step 5: Optional line range
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = len(lines)

        # Validate range
        if start_line < 1 or end_line > len(lines) or end_line < start_line:
            return f"✗ Invalid range: {start_line}-{end_line} (file has {len(lines)} lines)"

        # Step 6: Format with line numbers
        output = [f"File: {clean_path} ({len(lines)} lines total)\n"]
        for i in range(start_line - 1, end_line):
            line_num = i + 1
            line_text = lines[i].rstrip()
            output.append(f"{line_num:4d} | {line_text}")

        return "\n".join(output)

    except UnicodeDecodeError:
        return f"✗ File appears to be binary: {clean_path}"
    except PermissionError:
        return f"✗ Permission denied: {clean_path}"
    except Exception as e:
        return f"✗ Error reading {clean_path}: {e}"


def write_file(path: str, content: str, permission_manager=None) -> str:
    """
    Create or overwrite a file with content.

    Uses atomic write pattern: write to temp file, then rename.
    This prevents corruption if process dies mid-write.

    Args:
        path: File path (relative to project root)
        content: File content to write
        permission_manager: PermissionManager instance for access control

    Returns:
        Success message or error string
    """
    # Step 1: Normalize path (remove output/ prefixes)
    clean_path = _normalize_path(path)

    # Step 2: Permission check
    if permission_manager and not permission_manager.has_permission("write", clean_path):
        return f"✗ Permission denied: Cannot write to {clean_path}"

    # Step 3: Find project root
    project_root = _find_project_root()

    # Step 4: Security check (prevent path traversal)
    if not _validate_path_safety(clean_path, project_root):
        return f"✗ Security error: Path {clean_path} is outside project root"

    # Step 5: Construct full path
    full_path = project_root / clean_path

    # Step 6: Create parent directories
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 7: ATOMIC WRITE
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=full_path.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Atomic rename
        shutil.move(tmp_path, str(full_path))

        lines = content.count('\n') + 1
        return f"✓ Created: {clean_path} ({lines} lines)"

    except Exception as e:
        # Clean up temp file if it exists
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except:
                pass
        return f"✗ Error writing {clean_path}: {e}"


def replace_lines(path: str, start_line: int, end_line: int, new_content: str, permission_manager=None) -> str:
    """
    Replace a range of lines in a file deterministically.

    PROTOCOL (Agent MUST follow):
    1. Call read_file(path) first to see current content with line numbers
    2. Identify exact start_line and end_line (1-indexed)
    3. Call replace_lines(path, start_line, end_line, new_content)
    4. Read file again to verify

    Args:
        path: File to edit
        start_line: First line to replace (1-indexed, inclusive)
        end_line: Last line to replace (1-indexed, inclusive)
        new_content: New content (can be multiline)
        permission_manager: Access control

    Returns:
        Unified diff showing changes, or error message
    """
    # Step 1: Normalize path
    clean_path = _normalize_path(path)

    # Step 2: Permission check
    if permission_manager and not permission_manager.has_permission("write", clean_path):
        return f"✗ Permission denied: Cannot edit {clean_path}"

    # Step 3: Find project root and validate path
    project_root = _find_project_root()
    if not _validate_path_safety(clean_path, project_root):
        return f"✗ Security error: Path {clean_path} is outside project"

    full_path = project_root / clean_path

    # Step 4: File must exist
    if not full_path.exists():
        return f"✗ File not found: {clean_path}"

    tmp_path = None
    try:
        # Step 5: Read current file
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Step 6: Validate line numbers (1-indexed)
        if start_line < 1:
            return f"✗ Invalid start_line={start_line}: must be >= 1"
        if end_line < start_line:
            return f"✗ Invalid range: end_line ({end_line}) < start_line ({start_line})"
        if end_line > len(lines):
            return f"✗ Invalid end_line={end_line}: file only has {len(lines)} lines"

        # Step 7: Prepare new lines
        new_lines = []
        for line in new_content.split('\n'):
            if not line.endswith('\n'):
                new_lines.append(line + '\n')
            else:
                new_lines.append(line)

        # Step 8: Build new file content (0-indexed conversion)
        old_section = lines[start_line - 1:end_line]
        new_content_lines = (
            lines[:start_line - 1] +      # Keep lines before replacement
            new_lines +                    # Insert new content
            lines[end_line:]               # Keep lines after replacement
        )

        # Step 9: Generate unified diff for display
        diff = list(difflib.unified_diff(
            old_section,
            new_lines,
            lineterm='',
            fromfile=f"{clean_path} (before)",
            tofile=f"{clean_path} (after)"
        ))

        diff_text = '\n'.join(line.rstrip() for line in diff) if diff else "(no visible diff)"

        # Step 10: ATOMIC WRITE (temp → rename)
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=full_path.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp:
            tmp.writelines(new_content_lines)
            tmp_path = tmp.name

        shutil.move(tmp_path, str(full_path))

        # Step 11: Return diff for verification
        return f"✓ Edited {clean_path} (lines {start_line}-{end_line}):\n\n{diff_text}"

    except UnicodeDecodeError:
        return f"✗ File appears to be binary: {clean_path}"
    except Exception as e:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except:
                pass
        return f"✗ Error editing {clean_path}: {e}"


def list_files(pattern: str = "**/*.py", limit: int = 20) -> str:
    """List files matching glob pattern."""
    try:
        project_root = _find_project_root()
        files = list(project_root.glob(pattern))

        # Sort by modification time (most recent first)
        files.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)

        # Apply limit
        files = files[:limit]

        if not files:
            return f"No files found matching pattern: {pattern}"

        result_lines = [f"Found {len(files)} files matching '{pattern}':"]
        for f in files:
            try:
                # Show relative path from project root
                rel_path = f.relative_to(project_root)
                size = f.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                result_lines.append(f"  {rel_path} ({size_str})")
            except Exception:
                result_lines.append(f"  {f}")

        return "\n".join(result_lines)

    except Exception as e:
        return f"✗ Error listing files: {e}"


def search_files(pattern: str, file_pattern: str = "**/*.py", limit: int = 20) -> str:
    """Search for text pattern in files."""
    try:
        project_root = _find_project_root()
        files = list(project_root.glob(file_pattern))
        matches = []
        pattern_lower = pattern.lower()

        for file_path in files:
            if len(matches) >= limit:
                break

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern_lower in line.lower():
                            rel_path = file_path.relative_to(project_root)
                            matches.append({
                                "file": str(rel_path),
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
        return f"✗ Error searching files: {e}"


def search_code(query: str, top_k: int = 5, rag_instance=None) -> str:
    """Search codebase using semantic embeddings (RAG)."""
    if not rag_instance:
        return "✗ RAG not initialized. Use --use-rag flag."

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


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    permission_manager,
    rag_instance=None,
    output_dir: Optional[Path] = None  # Kept for backward compatibility, but ignored
) -> str:
    """Execute a tool with permission checking."""

    # Check if trying to modify source code
    if tool_name in ["write_file", "replace_lines"]:
        path = args.get("path", "")
        clean_path = _normalize_path(path)

        if "simplecoder/" in clean_path or clean_path.startswith("simplecoder"):
            return "✗ Cannot modify source code in simplecoder/"

    # Dispatch to appropriate function
    if tool_name == "read_file":
        return read_file(
            args["path"],
            args.get("start_line"),
            args.get("end_line"),
            permission_manager
        )

    elif tool_name == "write_file":
        return write_file(args["path"], args["content"], permission_manager)

    elif tool_name == "replace_lines":
        return replace_lines(
            args["path"],
            args["start_line"],
            args["end_line"],
            args["new_content"],
            permission_manager
        )

    elif tool_name == "list_files":
        return list_files(
            args.get("pattern", "**/*.py"),
            args.get("limit", 20)
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
            return "✗ RAG not initialized. Use --use-rag flag to enable semantic code search."
        return search_code(args["query"], args.get("top_k", 5), rag_instance)

    # Legacy support for edit_file (redirect to error)
    elif tool_name == "edit_file":
        return (
            "✗ edit_file is deprecated. Use replace_lines instead:\n"
            "1. First call read_file(path) to see line numbers\n"
            "2. Then call replace_lines(path, start_line, end_line, new_content)"
        )

    else:
        return f"✗ Unknown tool: {tool_name}"
