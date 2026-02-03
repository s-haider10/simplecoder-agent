# simplecoder/rag.py
# for RAG for code search.
# When working on large codebases, we cannot fit all code into the context window.
# RAG (Retrieval-Augmented Generation) is the traditional solution to this. Conventionally, we would slice the content into chunks and retrieve relevant components via text or embedding search.
# We ask that you use abstract syntax trees (ASTs) to chunk code in a more principled way (e.g. into functions, classes, etc.) and implement embedding-based RAG on top of these representations.

import ast
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from rich.console import Console


@dataclass
class CodeChunk:
    """Represents a semantic chunk of code (function, class, etc.)."""
    name: str
    chunk_type: str  # function, class, async_function, method
    code: str
    path: str
    start_line: int
    end_line: int = 0
    docstring: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self):
        return f"{self.chunk_type}({self.name}) at {self.path}:{self.start_line}"


class RAG:
    """Retrieval-Augmented Generation for code using AST-based chunking."""

    def __init__(
        self,
        embedder: str = "gemini/gemini-embedding-001",
        index_pattern: str = "**/*.py",
        cache_dir: str = ".simplecoder",
        verbose: bool = False
    ):
        """
        Initialize RAG system.

        Args:
            embedder: Embedding model to use (via litellm)
            index_pattern: Glob pattern for files to index
            cache_dir: Directory to store embedding cache
            verbose: Whether to print debug information
        """
        self.embedder = embedder
        self.index_pattern = index_pattern
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.console = Console()

        self.chunks: List[CodeChunk] = []
        self.embeddings_cache_path = self.cache_dir / "embeddings.pkl"

        # Flag for fallback to keyword search
        self.use_keyword_fallback = False

        # Build index on init
        self._build_index()

    def _build_index(self):
        """Build the code index from Python files."""
        # Check if we have a valid cache
        if self._load_cache():
            return

        if self.verbose:
            self.console.print("[bold yellow]RAG:[/bold yellow] Indexing codebase for semantic search...")

        # Find all matching files
        py_files = list(Path(".").glob(self.index_pattern))

        # Parse each file and extract chunks
        for py_file in py_files:
            # Skip cache and virtual environment directories
            if any(part.startswith('.') or part in ['__pycache__', 'venv', '.venv', 'node_modules']
                   for part in py_file.parts):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()

                chunks = self._extract_chunks(code, str(py_file))
                self.chunks.extend(chunks)

            except (SyntaxError, UnicodeDecodeError) as e:
                if self.verbose:
                    print(f"[RAG] Skipping {py_file}: {e}")
            except Exception as e:
                if self.verbose:
                    print(f"[RAG] Error parsing {py_file}: {e}")

        # Embed all chunks
        if self.chunks:
            self._embed_chunks()
            self._save_cache()

    def _extract_chunks(self, code: str, file_path: str) -> List[CodeChunk]:
        """
        Extract semantic chunks from Python code using AST.

        Args:
            code: Python source code
            file_path: Path to the source file

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return chunks

        lines = code.split('\n')

        for node in ast.walk(tree):
            chunk = None

            if isinstance(node, ast.FunctionDef):
                chunk = self._create_chunk(node, "function", file_path, lines)

            elif isinstance(node, ast.AsyncFunctionDef):
                chunk = self._create_chunk(node, "async_function", file_path, lines)

            elif isinstance(node, ast.ClassDef):
                chunk = self._create_chunk(node, "class", file_path, lines)

            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        node: ast.AST,
        chunk_type: str,
        file_path: str,
        lines: List[str]
    ) -> CodeChunk:
        """Create a CodeChunk from an AST node."""
        # Get the source code for this node
        try:
            chunk_code = ast.unparse(node)
        except Exception:
            # Fallback to extracting from source lines
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            chunk_code = '\n'.join(lines[start:end])

        # Extract docstring if present
        docstring = None
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value

        return CodeChunk(
            name=node.name,
            chunk_type=chunk_type,
            code=chunk_code,
            path=file_path,
            start_line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            docstring=docstring
        )

    def _embed_chunks(self):
        """Embed all chunks using the embedding model."""
        if not self.chunks:
            return

        # Create text representations for embedding
        # Include name, type, and a preview of the code
        texts = []
        for chunk in self.chunks:
            # Build a searchable representation
            text = f"{chunk.name} ({chunk.chunk_type})"
            if chunk.docstring:
                text += f": {chunk.docstring[:200]}"
            else:
                # Use first 150 chars of code as context
                text += f": {chunk.code[:150]}"
            texts.append(text)

        try:
            from litellm import embedding

            # Batch embed (some models have limits, so batch in groups)
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = embedding(model=self.embedder, input=batch)

                for j, emb_data in enumerate(response.data):
                    idx = i + j
                    if idx < len(self.chunks):
                        # Handle different response formats
                        if isinstance(emb_data, dict):
                            emb = emb_data.get("embedding", [])
                        else:
                            emb = getattr(emb_data, 'embedding', [])
                        self.chunks[idx].embedding = np.array(emb)

        except Exception as e:
            if self.verbose:
                print(f"[RAG] Embedding failed: {e}")
                print("[RAG] Falling back to keyword search")
            self.use_keyword_fallback = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant code chunks.

        Args:
            query: Natural language search query
            top_k: Number of results to return

        Returns:
            List of matching chunks with metadata
        """
        if not self.chunks:
            return []

        # Check if we should use keyword fallback
        if self.use_keyword_fallback or not any(c.embedding is not None for c in self.chunks):
            return self._keyword_search(query, top_k)

        try:
            from litellm import embedding

            # Embed the query
            response = embedding(model=self.embedder, input=[query])

            # Handle different response formats
            if isinstance(response.data[0], dict):
                query_embedding = np.array(response.data[0].get("embedding", []))
            else:
                query_embedding = np.array(response.data[0].embedding)

            # Calculate cosine similarity with all chunks
            similarities = []
            for chunk in self.chunks:
                if chunk.embedding is not None and len(chunk.embedding) > 0:
                    # Cosine similarity
                    dot_product = np.dot(query_embedding, chunk.embedding)
                    norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)

                    if norm_product > 0:
                        sim = dot_product / norm_product
                    else:
                        sim = 0

                    similarities.append((chunk, float(sim)))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top-k results
            results = []
            for chunk, score in similarities[:top_k]:
                results.append({
                    "name": chunk.name,
                    "type": chunk.chunk_type,
                    "path": chunk.path,
                    "line": chunk.start_line,
                    "code": chunk.code,
                    "docstring": chunk.docstring,
                    "score": score
                })

            return results

        except Exception as e:
            if self.verbose:
                print(f"[RAG] Search failed: {e}")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching chunks
        """
        if self.verbose:
            print("[RAG] Using keyword search fallback")

        query_words = set(query.lower().split())
        results = []

        for chunk in self.chunks:
            # Calculate simple keyword match score
            searchable = (chunk.name + " " + chunk.code + " " + (chunk.docstring or "")).lower()
            score = sum(1 for word in query_words if word in searchable)

            if score > 0:
                # Bonus for name matches
                if any(word in chunk.name.lower() for word in query_words):
                    score += 2

                results.append((chunk, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Convert to output format
        output = []
        for chunk, score in results[:top_k]:
            output.append({
                "name": chunk.name,
                "type": chunk.chunk_type,
                "path": chunk.path,
                "line": chunk.start_line,
                "code": chunk.code,
                "docstring": chunk.docstring,
                "score": score / max(len(query_words), 1)  # Normalize
            })

        return output

    def _save_cache(self):
        """Save chunks with embeddings to disk."""
        try:
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            if self.verbose:
                print(f"[RAG] Saved cache to {self.embeddings_cache_path}")
        except Exception as e:
            if self.verbose:
                print(f"[RAG] Cache save failed: {e}")

    def _load_cache(self) -> bool:
        """
        Load chunks from cache if valid.

        Returns:
            True if cache was loaded successfully
        """
        if not self.embeddings_cache_path.exists():
            return False

        try:
            with open(self.embeddings_cache_path, 'rb') as f:
                self.chunks = pickle.load(f)

            # Validate cache has embeddings
            if self.chunks and any(c.embedding is not None for c in self.chunks):
                return True

        except Exception as e:
            if self.verbose:
                print(f"[RAG] Cache load failed: {e}")

        return False

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.embeddings_cache_path.exists():
            self.embeddings_cache_path.unlink()
            if self.verbose:
                print("[RAG] Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the RAG index."""
        return {
            "total_chunks": len(self.chunks),
            "embedded_chunks": sum(1 for c in self.chunks if c.embedding is not None),
            "cache_exists": self.embeddings_cache_path.exists(),
            "using_keyword_fallback": self.use_keyword_fallback
        }
