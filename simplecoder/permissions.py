# simplecoder/permissions.py
# for managing task- and session-level permissions from the user for reading, writing files, etc.

import json
import os
from pathlib import Path
from fnmatch import fnmatch
from typing import Optional


class PermissionManager:
    """Manage file read/write permissions with session persistence."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.session_dir = Path.home() / ".simplecoder"
        self.session_file = self.session_dir / f"permissions_{self.session_id}.json"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.permissions = self._load_permissions()

        # Default deny patterns (files we never touch)
        self.system_deny_patterns = [
            ".git/*",
            ".git/**",
            ".env",
            ".env.*",
            ".venv/*",
            ".venv/**",
            "__pycache__/*",
            "__pycache__/**",
            "*.pyc",
            ".pytest_cache/*",
            ".pytest_cache/**",
            "node_modules/*",
            "node_modules/**",
            ".ssh/*",
            ".ssh/**",
            # Protect simplecoder source code
            "simplecoder/*",
            "simplecoder/**",
        ]

    def has_permission(self, action: str, path: str) -> bool:
        """Check if action is permitted on path."""
        # Normalize path
        try:
            normalized_path = str(Path(path).resolve())
        except Exception:
            normalized_path = path

        # Also check relative path for pattern matching
        relative_path = path

        # Check system deny patterns first
        for pattern in self.system_deny_patterns:
            if fnmatch(normalized_path, pattern) or fnmatch(relative_path, pattern):
                return False
            # Also check if the pattern matches any part of the path
            if fnmatch(Path(path).name, pattern):
                return False

        if action == "read":
            # Default allow for read unless explicitly denied
            return not self._is_denied(path, "read")

        elif action == "write":
            # Write requires explicit permission OR is in current working directory
            if self._is_allowed(path, "write"):
                return True

            # Auto-allow writes in current working directory subtree
            try:
                cwd = Path.cwd().resolve()
                target = Path(path).resolve()
                if str(target).startswith(str(cwd)):
                    # Auto-grant permission for this path
                    self.allow("write", path)
                    return True
            except Exception:
                pass

            return False

        return False

    def allow(self, action: str, path: str):
        """Grant permission for action on path."""
        if action not in self.permissions:
            self.permissions[action] = {"allowed": [], "denied": []}

        # Normalize path for storage
        normalized = str(Path(path))

        if normalized not in self.permissions[action]["allowed"]:
            self.permissions[action]["allowed"].append(normalized)

        # Remove from denied if present
        if normalized in self.permissions[action]["denied"]:
            self.permissions[action]["denied"].remove(normalized)

        self._save_permissions()

    def deny(self, action: str, path: str):
        """Deny permission for action on path."""
        if action not in self.permissions:
            self.permissions[action] = {"allowed": [], "denied": []}

        normalized = str(Path(path))

        if normalized not in self.permissions[action]["denied"]:
            self.permissions[action]["denied"].append(normalized)

        # Remove from allowed if present
        if normalized in self.permissions[action]["allowed"]:
            self.permissions[action]["allowed"].remove(normalized)

        self._save_permissions()

    def _is_allowed(self, path: str, action: str) -> bool:
        """Check if path is explicitly allowed."""
        if action not in self.permissions:
            return False

        normalized = str(Path(path))

        for pattern in self.permissions[action].get("allowed", []):
            if fnmatch(normalized, pattern) or fnmatch(path, pattern):
                return True
            # Exact match
            if normalized == pattern or path == pattern:
                return True

        return False

    def _is_denied(self, path: str, action: str) -> bool:
        """Check if path is explicitly denied."""
        if action not in self.permissions:
            return False

        normalized = str(Path(path))

        for pattern in self.permissions[action].get("denied", []):
            if fnmatch(normalized, pattern) or fnmatch(path, pattern):
                return True

        return False

    def _generate_session_id(self) -> str:
        """Generate session ID from current directory."""
        return str(abs(hash(os.getcwd())))[:8]

    def _load_permissions(self) -> dict:
        """Load permissions from disk."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "read": {"allowed": [], "denied": []},
            "write": {"allowed": [], "denied": []}
        }

    def _save_permissions(self):
        """Save permissions to disk."""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.permissions, f, indent=2)
        except IOError as e:
            print(f"[PERMISSIONS] Warning: Could not save permissions: {e}")

    def get_status(self) -> str:
        """Return human-readable permission status."""
        lines = [f"Session ID: {self.session_id}"]

        for action in ["read", "write"]:
            if action in self.permissions:
                allowed = self.permissions[action].get("allowed", [])
                denied = self.permissions[action].get("denied", [])
                if allowed:
                    lines.append(f"{action.capitalize()} allowed: {', '.join(allowed[:5])}")
                if denied:
                    lines.append(f"{action.capitalize()} denied: {', '.join(denied[:5])}")

        return "\n".join(lines) if len(lines) > 1 else "No custom permissions set."
