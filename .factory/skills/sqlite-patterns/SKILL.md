| name | description |
|------|-------------|
| sqlite-patterns | Patterns for SQLite database operations in llama-webui |

## SQLite Database Patterns

Use this skill when working with the SQLite database layer in `app_state.py` or related modules.

### Thread Safety

- Always use `threading.RLock()` for thread-safe database access
- Wrap all database operations with the lock context manager

### Connection Setup

```python
self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
self._conn.row_factory = sqlite3.Row
self._lock = threading.RLock()
```

### Schema Definition

- Use `executescript()` for multi-statement schema creation
- Use `IF NOT EXISTS` for table creation
- Set appropriate SQLite pragmas: `journal_mode=WAL`, `synchronous=NORMAL`

### Query Patterns

```python
with self._lock:
    row = self._conn.execute("SELECT ...", (param,)).fetchone()
    # or for multiple rows
    rows = self._conn.execute("SELECT ...").fetchall()
```

### INSERT with RETURNING

```python
with self._lock:
    cur = self._conn.execute("INSERT INTO table (col) VALUES (?)", (value,))
    self._conn.commit()
    inserted_id = cur.lastrowid
```

### Foreign Keys

- Always use `ON DELETE CASCADE` for foreign key relationships
- Use explicit foreign key constraints in table definitions
