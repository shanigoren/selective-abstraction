import json
import os
import sqlite3
import threading
import time
import atexit
from typing import Any, Callable, Optional
import re


# Resolve the repository root (parent of this `cache/` directory).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Cache files are runtime outputs, not repository resources.  The directory is
# only created when a caller opts in and a database connection is opened.
_CACHE_ROOT = os.path.abspath(
    os.environ.get("SA_CACHE_DIR") or os.path.join(_REPO_ROOT, "outputs", "cache")
)

# Default DB used only by the generic `result_cache` (e.g., tests or
# future generic use-cases).
DEFAULT_DB_PATH = os.path.join(_CACHE_ROOT, "result_cache.sqlite3")

# Dedicated DBs for the two main heavy tasks so they can grow
# independently and be managed separately.
COVERAGE_DB_PATH = os.path.join(_CACHE_ROOT, "coverage_cache.sqlite3")
CORRECTNESS_DB_PATH = os.path.join(_CACHE_ROOT, "correctness_cache.sqlite3")

# Bump this if the semantics of what is stored for a given task_type change.
CACHE_VERSION = 1

def remove_first_last_brackets(text):
    # remove the dot from the end of the text if it exists
    if text.endswith('.'):
        text = text[:-1]
    # Remove the first set of brackets
    text = re.sub(r'^\s*\[([^\]]*)\]', r'\1', text, count=1)
    # Remove the last set of brackets
    text = re.sub(r'\[([^\]]*)\]\s*$', r'\1', text, count=1)
    # put the dot back 
    text = text + '.'
    return text


def cache_correctness_key_for_abstraction(text: str) -> str:
    """
    Build a canonical cache key for a correctness abstraction.

    We intentionally ignore run-time configuration such as model name or
    temperature. The goal is to reuse annotations for the same logical
    abstraction wherever it appears.
    """
    return remove_first_last_brackets(text).strip()



class ResultCache:
    """
    SQLite-backed cache for expensive evaluation results.

    Design goals:
    - Safe to use from many processes and threads.
    - Simple API for callers:
        - claim_or_wait(...)   → either get an existing result or become
                                 the "owner" responsible for computing it.
        - store_success(...)
        - get_or_compute(...)  → convenience wrapper for purely-sync flows.
    - The only thing cached is the *result payload* (JSON-serializable),
      not configuration such as model, temperature, etc.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        # Lightweight mutex to ensure init_db is only run once per process.
        self._init_lock = threading.Lock()
        self._initialized = False
        # Track active connections for cleanup (though we create new ones per call)
        self._cleanup_registered = False
        # Register cleanup on exit as a safety net
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection.

        We intentionally create a fresh connection per call rather than
        sharing across threads. SQLite connections are not fully
        thread-safe by default, but the database file itself is safely
        shared across processes.
        """
        parent = os.path.dirname(os.path.abspath(self._db_path))
        os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS results_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            task_type TEXT NOT NULL,
                            abstraction_key TEXT NOT NULL,
                            version INTEGER NOT NULL,
                            status TEXT NOT NULL,
                            result_json TEXT,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            UNIQUE(task_type, abstraction_key, version)
                        );
                        """
                    )
                    # Clean up any legacy 'running' rows from the older
                    # claim-or-wait design. The new implementation only
                    # uses status='success', and leaves concurrency to
                    # first-writer-wins on insert.
                    try:
                        conn.execute(
                            "DELETE FROM results_cache WHERE status != 'success';"
                        )
                    except sqlite3.OperationalError:
                        # If the table schema is older or missing the status
                        # column for some reason, ignore and continue.
                        pass
                self._initialized = True
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def claim_or_wait(
        self,
        task_type: str,
        abstraction_key: str,
        *,
        version: int = CACHE_VERSION,
        poll_interval: float = 0.25,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Look up an existing cached result for a key.

        This method now has **read-only** semantics:

        Returns:
            - A previously computed result (JSON-decoded) if present.
            - None if no successful result is cached yet.

        Notes:
            - The older \"claim the key and wait for the owner\" behaviour
              has been removed. Multiple workers are now free to compute
              the same key concurrently; the first successful writer wins.
            - The `poll_interval` and `timeout` parameters are retained for
              backward compatibility but are ignored.
        """
        self._ensure_initialized()
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT status, result_json
                FROM results_cache
                WHERE task_type = ? AND abstraction_key = ? AND version = ?
                """,
                (task_type, abstraction_key, version),
            )
            row = cur.fetchone()
            if row is None:
                return None
            status, result_json = row
            if status == "success" and result_json is not None:
                try:
                    return json.loads(result_json)
                except Exception:
                    # If result_json is somehow corrupted, treat as missing.
                    return None
            return None
        finally:
            conn.close()

    def store_success(
        self,
        task_type: str,
        abstraction_key: str,
        result_obj: Any,
        *,
        version: int = CACHE_VERSION,
    ) -> bool:
        """
        Store a successful result for a key.

        Semantics:
            - If no row exists yet for (task_type, abstraction_key, version),
              insert a new row with status='success'.
            - If a row already exists (from another worker), leave it
              unchanged – the *first* successful writer wins.

        Returns:
            - True if this call inserted a new row.
            - False if a row already existed and was left unchanged.
        """
        self._ensure_initialized()
        conn = self._connect()
        try:
            payload = json.dumps(result_obj)
            now_ts = time.time()
            with conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO results_cache (
                        task_type,
                        abstraction_key,
                        version,
                        status,
                        result_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, 'success', ?, ?, ?)
                    """,
                    (task_type, abstraction_key, version, payload, now_ts, now_ts),
                )
            return cur.rowcount == 1
        finally:
            conn.close()

    def get_or_compute(
        self,
        task_type: str,
        abstraction_key: str,
        compute_fn: Callable[[], Any],
        *,
        version: int = CACHE_VERSION,
        poll_interval: float = 0.25,
        timeout: Optional[float] = 60,
    ) -> Any:
        """
        Convenience wrapper for purely synchronous flows:

        1) Try to reuse an existing cached result.
        2) If none exists, run `compute_fn`, attempt to store the result,
           and return the *canonical* cached value.
        3) If another worker wins the race and stores first, all callers
           will see that worker's result.

        Notes:
            - Multiple workers may compute the same key concurrently.
              The cache behaves as \"first successful writer wins\".
            - `poll_interval` and `timeout` are retained for backward
              compatibility but are ignored – there is no waiting on
              other workers.
        """
        source = None
        # Step 1: attempt to reuse an existing cached result.
        existing = self.claim_or_wait(
            task_type,
            abstraction_key,
            version=version,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if existing is not None:
            source = "cache"
            return existing, source

        # Step 2: compute locally. If we fail, another worker may have
        # completed in the meantime, so we re-check the cache before
        # surfacing the error.
        try:
            result = compute_fn()
            source = "agent run"
        except Exception:
            fallback = self.claim_or_wait(
                task_type,
                abstraction_key,
                version=version,
                poll_interval=poll_interval,
                timeout=timeout,
            )
            if fallback is not None:
                source = "another worker"
                return fallback, source
            raise

        # Step 3: best-effort store – only the first writer wins.
        if result is not None:
            self.store_success(task_type, abstraction_key, result, version=version)

        # Step 4: read back the canonical payload. This ensures that all
        # callers see the same result even if multiple workers computed
        # concurrently.
        canonical = self.claim_or_wait(
            task_type,
            abstraction_key,
            version=version,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if canonical:
            if not source:
                source = "cache (canonical)"
            return canonical, source
        return result, source

    def checkpoint_wal(self) -> None:
        """
        Checkpoint the WAL file to reduce the chance of issues on abrupt termination.
        This is a best-effort operation that may fail if the database is locked.
        """
        if not self._initialized:
            return
        try:
            conn = self._connect()
            try:
                # Checkpoint the WAL file to merge it into the main database
                # This reduces the chance of corruption if the process is killed
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.OperationalError:
                # Database might be locked by another process, that's okay
                pass
            finally:
                conn.close()
        except Exception:
            # Ignore any errors during checkpoint - this is best-effort cleanup
            pass

    def cleanup(self) -> None:
        """
        Clean up cache connections. Called on normal exit or via signal handlers.
        This is a best-effort operation to ensure connections are closed.
        """
        try:
            # Checkpoint WAL to reduce corruption risk
            self.checkpoint_wal()
        except Exception:
            # Ignore errors during cleanup
            pass


# Shared singletons used across the codebase.
#
# - `coverage_cache`    → coverage/coverage.py
# - `correctness_cache` → correctness/wikipedia_factcheck_agent/agent.py
# - `result_cache`      → generic / tests (shares DEFAULT_DB_PATH)
coverage_cache = ResultCache(COVERAGE_DB_PATH)
correctness_cache = ResultCache(CORRECTNESS_DB_PATH)
result_cache = ResultCache()


