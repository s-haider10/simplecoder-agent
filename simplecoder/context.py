# simplecoder/context.py
# for context management and compacting.
# LLMs have token limits and long conversations may eventually exceed their context window, despite our efforts to contain codebases themselves via RAG.
# We ask you to implement a toolkit for managing this: track context use, and summarize conversation history when exceeding a certain point (you can choose to set this heuristically rather than being model specific, for the course of this assignment).
# You should allow for keeping the last k (configurable number) messages intact.

import json
from typing import List, Dict, Any
from rich.console import Console


class ContextManager:
    """Manage conversation context with token tracking and summarization."""

    def __init__(
        self,
        max_tokens: int = 6000,
        keep_last_k: int = 8,
        verbose: bool = False
    ):
        """
        Initialize the context manager.

        Args:
            max_tokens: Threshold at which to trigger summarization
            keep_last_k: Number of recent messages to keep intact when summarizing
            verbose: Whether to print debug information
        """
        self.max_tokens = max_tokens
        self.keep_last_k = keep_last_k
        self.verbose = verbose
        self.summarization_count = 0
        self.console = Console()

    def estimate_tokens(self, content: Any) -> int:
        """
        Conservative token estimation (better than len/4).

        Uses ~1 token per 2.5 characters for more accurate estimates.
        This is conservative to prevent context overflow.

        Args:
            content: String, dict, or list to estimate tokens for

        Returns:
            Estimated token count
        """
        if content is None:
            return 0

        if isinstance(content, str):
            # Conservative: ~1 token per 2.5 characters
            return max(1, int(len(content) / 2.5))

        elif isinstance(content, dict):
            # JSON serialization + overhead for structure
            try:
                serialized = json.dumps(content)
                # Conservative estimate + 50 tokens overhead for structure
                return max(1, int(len(serialized) / 2.5)) + 50
            except (TypeError, ValueError):
                # Fallback for non-serializable dicts
                return sum(
                    self.estimate_tokens(k) + self.estimate_tokens(v)
                    for k, v in content.items()
                )

        elif isinstance(content, list):
            return sum(self.estimate_tokens(item) for item in content)

        else:
            # Convert to string for other types
            return self.estimate_tokens(str(content))

    def get_context_for_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get context for LLM, summarizing if necessary.

        Args:
            messages: Full conversation history

        Returns:
            Possibly compacted conversation history
        """
        if self._should_summarize(messages):
            return self._compact_history(messages)
        return messages

    def _should_summarize(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Check if context needs summarization.

        Conditions:
        1. Total tokens exceed max_tokens threshold
        2. Have enough messages to summarize (more than keep_last_k + 2)

        Summarizes sooner (at 4000 tokens) to prevent overflow.

        Args:
            messages: Conversation history

        Returns:
            True if summarization is needed
        """
        # Need at least some older messages to summarize
        min_messages_for_summary = self.keep_last_k + 2  # Keep system + user + last k
        if len(messages) <= min_messages_for_summary:
            return False

        # Also require at least 10 messages before summarizing
        if len(messages) < 10:
            return False

        total_tokens = sum(
            self.estimate_tokens(msg.get("content", ""))
            for msg in messages
        )

        # Lower threshold (4000) to summarize sooner and prevent overflow
        threshold = min(self.max_tokens, 4000)
        should_summarize = total_tokens > threshold

        if self.verbose and should_summarize:
            self.console.print("[bold magenta]Memory:[/bold magenta] Summarizing conversation history to manage context...")

        return should_summarize

    def _compact_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compact history by summarizing older messages.

        Strategy:
        1. Keep system message (if present)
        2. Keep the original user task (first user message)
        3. Summarize middle portion
        4. Keep last k messages intact

        Args:
            messages: Full conversation history

        Returns:
            Compacted conversation history
        """
        self.summarization_count += 1

        # Separate messages
        system_msg = None
        first_user_msg = None
        rest_messages = []

        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            elif msg.get("role") == "user" and first_user_msg is None:
                first_user_msg = msg
            else:
                rest_messages.append(msg)

        # If we have fewer messages than keep_last_k, no point in summarizing
        if len(rest_messages) <= self.keep_last_k:
            return messages

        # Split into old (to summarize) and recent (to keep).
        # IMPORTANT: never split an assistant[tool_calls] / tool group — Gemini
        # requires every tool message to have its parent assistant immediately before it.
        # Walk backwards from the cut point and push it earlier if it lands mid-group.
        cut = len(rest_messages) - self.keep_last_k
        while cut > 0:
            msg = rest_messages[cut]
            # If the cut lands on a tool message, keep scanning back until we find
            # the assistant that owns it (or the start of the list).
            if msg.get("role") == "tool":
                cut -= 1
                continue
            # If the cut lands right after an assistant with tool_calls, that assistant
            # must stay with its tool messages — move cut before it.
            if cut > 0 and rest_messages[cut - 1].get("role") == "assistant" and rest_messages[cut - 1].get("tool_calls"):
                cut -= 1
                continue
            break

        old_messages = rest_messages[:cut]
        recent_messages = rest_messages[cut:]

        # Create extractive summary of old messages
        summary = self._extractive_summary(old_messages)

        # Build compacted history
        compacted = []

        if system_msg:
            compacted.append(system_msg)

        if first_user_msg:
            compacted.append(first_user_msg)

        # Add summary as a system message
        if summary:
            compacted.append({
                "role": "system",
                "content": f"[Conversation Summary]\n{summary}\n[End Summary - Recent messages follow]"
            })

        # Add recent messages
        compacted.extend(recent_messages)

        return compacted

    def _extractive_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create an extractive summary of messages.

        Strategy: Keep important content like code snippets and key actions,
        but truncate long explanations.

        Args:
            messages: Messages to summarize

        Returns:
            Summary string
        """
        summary_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if not content:
                continue

            # Handle tool messages specially
            if role == "tool":
                # Summarize tool results briefly
                if "Successfully" in content:
                    summary_parts.append(f"• Tool: {content.split('.')[0]}.")
                elif "Error" in content:
                    summary_parts.append(f"• Tool error: {content[:100]}")
                continue

            # For assistant messages, extract key information
            if role == "assistant":
                # Keep code blocks intact but summarize text
                if "```" in content:
                    # Extract first code block preview
                    parts = content.split("```")
                    if len(parts) >= 2:
                        code_preview = parts[1][:150]
                        summary_parts.append(f"• Assistant provided code: ```{code_preview}...```")
                    else:
                        summary_parts.append(f"• Assistant: {content[:150]}...")
                elif len(content) > 200:
                    # Truncate long responses
                    first_sentence = content.split(".")[0] + "."
                    summary_parts.append(f"• Assistant: {first_sentence}")
                else:
                    summary_parts.append(f"• Assistant: {content}")
                continue

            # For user messages (not the first one)
            if role == "user":
                if len(content) > 100:
                    summary_parts.append(f"• User asked about: {content[:100]}...")
                else:
                    summary_parts.append(f"• User: {content}")

        # Limit summary length
        if len(summary_parts) > 15:
            summary_parts = summary_parts[:10] + ["..."] + summary_parts[-4:]

        return "\n".join(summary_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about context management."""
        return {
            "summarization_count": self.summarization_count,
            "max_tokens": self.max_tokens,
            "keep_last_k": self.keep_last_k
        }
