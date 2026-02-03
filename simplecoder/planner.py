# simplecoder/planner.py
# for task planning and decomposition.
# Should take a task description and return a set of subtasks, with a procedure for managing their incremental completion towards the goal.

from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console
from rich.panel import Panel


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """Represents a subtask in a task plan."""
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None


@dataclass
class TaskPlan:
    """Represents a decomposed task with subtasks."""
    original_task: str
    subtasks: List[SubTask] = field(default_factory=list)
    current_index: int = 0

    def mark_current_complete(self, result: str = ""):
        """Mark the current subtask as complete and advance."""
        if self.current_index < len(self.subtasks):
            self.subtasks[self.current_index].status = TaskStatus.COMPLETED
            self.subtasks[self.current_index].result = result
            self.current_index += 1

    def mark_current_failed(self, error: str = ""):
        """Mark the current subtask as failed."""
        if self.current_index < len(self.subtasks):
            self.subtasks[self.current_index].status = TaskStatus.FAILED
            self.subtasks[self.current_index].result = error

    def get_current_subtask(self) -> Optional[SubTask]:
        """Get the current subtask to work on."""
        if self.current_index < len(self.subtasks):
            return self.subtasks[self.current_index]
        return None

    def is_complete(self) -> bool:
        """Check if all subtasks are complete."""
        return self.current_index >= len(self.subtasks)

    def get_progress(self) -> str:
        """Get a progress summary."""
        completed = sum(1 for t in self.subtasks if t.status == TaskStatus.COMPLETED)
        return f"{completed}/{len(self.subtasks)} subtasks completed"


class TaskPlanner:
    """Decomposes complex tasks into manageable subtasks."""

    def __init__(self, model: str = "gemini/gemini-3-pro-preview", verbose: bool = False):
        """
        Initialize the task planner.

        Args:
            model: LLM model to use for decomposition
            verbose: Whether to print debug information
        """
        self.model = model
        self.verbose = verbose
        self.console = Console()

        # Keywords that suggest a task needs decomposition
        self.complex_keywords = [
            " and ", " with ", " then ", " also ",
            "multiple", "several", "create", "build",
            "implement", "refactor", "update", "fix"
        ]

    def decompose(self, task: str) -> List[str]:
        """
        Decompose a task into subtasks.

        Uses a combination of heuristics and LLM-based decomposition:
        1. Short, simple tasks are returned as-is
        2. Complex tasks are decomposed using the LLM

        Args:
            task: The task description

        Returns:
            List of subtask descriptions
        """
        # Heuristic: Simple tasks don't need decomposition
        if self._is_simple_task(task):
            return [task]

        # Use LLM to decompose complex tasks
        if self.verbose:
            self.console.print("[bold blue]Planner:[/bold blue] Analyzing task and creating plan...")

        subtasks = self._llm_decompose(task)

        return subtasks

    def _is_simple_task(self, task: str) -> bool:
        """
        Determine if a task is simple enough to not need decomposition.

        Heuristics:
        - Task is short (< 50 characters)
        - No complex keywords present
        - Single action verb

        Args:
            task: The task description

        Returns:
            True if the task is simple
        """
        # Short tasks are usually simple
        if len(task) < 50:
            # But check for complex keywords
            task_lower = task.lower()
            for keyword in self.complex_keywords:
                if keyword in task_lower:
                    return False
            return True

        return False

    def _llm_decompose(self, task: str) -> List[str]:
        """
        Use LLM to decompose a complex task.

        Args:
            task: The complex task description

        Returns:
            List of subtasks
        """
        prompt = f"""Break down this coding task into 2-4 clear, sequential subtasks.
Each subtask should be a specific, actionable step.
Output only the subtasks, one per line, as a numbered list (1., 2., etc.).
Do not include explanations or extra text.

Task: {task}

Subtasks:"""

        try:
            from litellm import completion

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )

            text = response.choices[0].message.content or ""

            # Parse the numbered list
            subtasks = self._parse_numbered_list(text)

            # Validate we got reasonable subtasks
            if subtasks and len(subtasks) >= 1:
                return subtasks

        except Exception as e:
            if self.verbose:
                print(f"[bold red][PLANNER] LLM decomposition failed:[/bold red] {e}")

        # Fallback: try simple heuristic decomposition
        return self._heuristic_decompose(task)

    def _parse_numbered_list(self, text: str) -> List[str]:
        """
        Parse a numbered list from text.

        Args:
            text: Text containing numbered list

        Returns:
            List of items
        """
        subtasks = []

        for line in text.strip().split("\n"):
            line = line.strip()

            if not line:
                continue

            # Check if line starts with a number
            if line[0].isdigit():
                # Remove numbering (e.g., "1.", "1)", "1:")
                parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1) if ")" in line[:3] else [line[:1], line[1:]]

                if len(parts) > 1:
                    subtask = parts[1].strip()
                else:
                    subtask = line

                # Clean up common prefixes
                subtask = subtask.lstrip(".- ")

                if subtask and len(subtask) > 3:
                    subtasks.append(subtask)

            # Also handle bullet points
            elif line.startswith(("-", "*", "•")):
                subtask = line[1:].strip()
                if subtask and len(subtask) > 3:
                    subtasks.append(subtask)

        return subtasks

    def _heuristic_decompose(self, task: str) -> List[str]:
        """
        Simple heuristic-based task decomposition.

        Looks for conjunctions like "and", "then" to split tasks.

        Args:
            task: The task description

        Returns:
            List of subtasks
        """
        # Try splitting on common conjunctions
        task_lower = task.lower()
        subtasks = []

        # Split on " and " or " then "
        if " and " in task_lower:
            parts = task.split(" and ", 1)
            if len(parts) == 2 and len(parts[0]) > 10 and len(parts[1]) > 10:
                subtasks = [p.strip() for p in parts]

        elif " then " in task_lower:
            parts = task.split(" then ", 1)
            if len(parts) == 2:
                subtasks = [p.strip() for p in parts]

        # If we got subtasks, return them
        if subtasks:
            return subtasks

        # Otherwise return original task
        return [task]

    def create_plan(self, task: str) -> TaskPlan:
        """
        Create a full task plan with status tracking.

        Args:
            task: The task description

        Returns:
            TaskPlan object with subtasks
        """
        subtask_descriptions = self.decompose(task)

        subtasks = [SubTask(description=desc) for desc in subtask_descriptions]

        return TaskPlan(original_task=task, subtasks=subtasks)
