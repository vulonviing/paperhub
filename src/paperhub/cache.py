"""SQLite cache for paper metadata, PDF text, and summaries.

The cache is the difference between a polite second run and a duplicate
charge to the LLM provider. It is intentionally simple: sqlite3 with one small
schema migration for language-aware summary cache keys. The schema is created
on first use.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from .models import PaperMeta, PaperSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_meta (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    abstract TEXT,
    upvotes INTEGER NOT NULL DEFAULT 0,
    num_comments INTEGER NOT NULL DEFAULT 0,
    submitted_by TEXT,
    published_at TEXT,
    hf_url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pdf_text (
    arxiv_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    chars INTEGER NOT NULL,
    extracted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summary (
    arxiv_id TEXT NOT NULL,
    model TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (arxiv_id, model, language)
);
"""


class Cache:
    """SQLite-backed cache rooted at a single directory.

    The same directory holds the SQLite DB and a `pdfs/` subdirectory used
    by the downloader for raw PDF files.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "paperhub.sqlite"
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_summary_language(conn)

    def _migrate_summary_language(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(summary)").fetchall()}
        if "language" in columns:
            return
        conn.executescript(
            """
            ALTER TABLE summary RENAME TO summary_legacy;

            CREATE TABLE summary (
                arxiv_id TEXT NOT NULL,
                model TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (arxiv_id, model, language)
            );

            INSERT OR REPLACE INTO summary (arxiv_id, model, language, payload, created_at)
            SELECT arxiv_id, model, 'tr', payload, created_at FROM summary_legacy;

            DROP TABLE summary_legacy;
            """
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def put_meta(self, paper: PaperMeta) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_meta
                (arxiv_id, title, authors_json, abstract, upvotes, num_comments,
                 submitted_by, published_at, hf_url, pdf_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.arxiv_id,
                    paper.title,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.abstract,
                    paper.upvotes,
                    paper.num_comments,
                    paper.submitted_by,
                    paper.published_at.isoformat() if paper.published_at else None,
                    paper.hf_url,
                    paper.pdf_url,
                ),
            )

    def get_meta(self, arxiv_id: str) -> PaperMeta | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM paper_meta WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        if row is None:
            return None
        published = row["published_at"]
        return PaperMeta(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            abstract=row["abstract"],
            upvotes=row["upvotes"],
            num_comments=row["num_comments"],
            submitted_by=row["submitted_by"],
            published_at=date.fromisoformat(published) if published else None,
            hf_url=row["hf_url"],
            pdf_url=row["pdf_url"],
        )

    def put_pdf_text(self, arxiv_id: str, text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pdf_text (arxiv_id, text, chars)
                VALUES (?, ?, ?)
                """,
                (arxiv_id, text, len(text)),
            )

    def get_pdf_text(self, arxiv_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text FROM pdf_text WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        return row["text"] if row else None

    def put_summary(self, summary: PaperSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO summary (arxiv_id, model, language, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    summary.arxiv_id,
                    summary.model_used,
                    summary.language,
                    summary.model_dump_json(),
                ),
            )

    def get_summary(
        self,
        arxiv_id: str,
        model: str,
        language: str = "en",
    ) -> PaperSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM summary
                WHERE arxiv_id = ? AND model = ? AND language = ?
                """,
                (arxiv_id, model, language),
            ).fetchone()
        if row is None:
            return None
        try:
            return PaperSummary.model_validate_json(row["payload"])
        except Exception:
            return None

    def clear(self, *, summaries: bool = True, pdfs: bool = True) -> dict[str, int]:
        """Delete cached data. Returns counts of deleted rows/files."""
        deleted: dict[str, int] = {"summaries": 0, "pdf_text": 0, "pdf_files": 0}
        with self._connect() as conn:
            if summaries:
                deleted["summaries"] = conn.execute("DELETE FROM summary").rowcount
            if pdfs:
                deleted["pdf_text"] = conn.execute("DELETE FROM pdf_text").rowcount
        if pdfs:
            for pdf_file in self.pdf_dir.glob("*.pdf"):
                pdf_file.unlink(missing_ok=True)
                deleted["pdf_files"] += 1
        return deleted

    def stats(self) -> dict[str, int]:
        """Return row counts for each cached table."""
        with self._connect() as conn:
            return {
                "summaries": conn.execute("SELECT COUNT(*) FROM summary").fetchone()[0],
                "pdf_text": conn.execute("SELECT COUNT(*) FROM pdf_text").fetchone()[0],
                "pdf_files": sum(1 for _ in self.pdf_dir.glob("*.pdf")),
            }
