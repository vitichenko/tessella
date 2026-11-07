use rusqlite::Connection;
use tessella_core::error::DomainError;
use tracing::{debug, info, warn};

/// How long a write waits for another **process** to release the lock before failing (`TASK-42`/S48).
///
/// `busy_timeout` defaults to `0`, so any collision was an immediate `SQLITE_BUSY` — and before S48
/// the schema migrations and the reputation writes swallowed exactly that error, which is how a
/// database that refuses writes looked perfectly healthy. Each process opens **one** connection (it
/// lives behind a `Mutex`), so the contention this protects against is cross-process: a second worker
/// on the same file, the operator's `sqlite3`, or a WAL checkpoint.
pub const SQLITE_BUSY_TIMEOUT_MS: u32 = 5_000;

/// Columna de `ledger` de la que sale la clave de deduplicación de un asiento federado.
pub const LEDGER_DEDUP_INDEX: &str = "idx_ledger_dedup";

/// The SQL type a migration declares for a column (`"INTEGER"` in
/// `ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 0`), or `None` when the statement does not add that
/// column.
///
/// It exists to compare what the schema **declares** with what `PRAGMA table_info` reports: SQLite
/// cannot change the type of an existing column in place, so the only honest thing a migration can do
/// with a divergent type is say it out loud (see [`ensure_column`]).
pub fn declared_type_in<'a>(ddl: &'a str, column: &str) -> Option<&'a str> {
    let marker = format!("ADD COLUMN {column} ");
    let start = ddl.find(&marker)? + marker.len();
    ddl[start..].split_whitespace().next()
}

/// The type SQLite has declared for `table.column`, or `None` when the column does not exist
/// (`PRAGMA table_info` returns one row per column: `cid`, `name`, `type`, …).
pub fn column_declaration(
    conn: &Connection,
    table: &str,
    column: &str,
) -> Result<Option<String>, DomainError> {
    let inspect = |e: rusqlite::Error| {
        DomainError::Storage(format!("Cannot inspect the schema of {table}: {e}"))
    };

    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(inspect)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })
        .map_err(inspect)?;

    for row in rows {
        let (name, declared_type) = row.map_err(inspect)?;
        if name == column {
            return Ok(Some(declared_type));
        }
    }
    Ok(None)
}

/// Adds `column` to `table` when it is missing (`TASK-42`/S48 · SP-09).
///
/// Every migration used to be a `let _ = conn.execute("ALTER TABLE …")`, which made two very
/// different events indistinguishable: «the column is already there» (the normal case from the second
/// boot onwards) and «the database is locked / the disk is full / the table is not the one this build
/// expects». The first is a no-op with a `debug!`; the second is an error that names the statement, so
/// a half-migrated schema is visible at boot instead of months later (`no such table: karma_ledger`
/// was the same pattern, `TASK-30`).
///
/// An existing column is never migrated — SQLite does not retype a column in place — but a declared
/// type that differs from the schema gets a `warn!`: a value stored under the other type is a
/// round-trip nobody has tested.
pub fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    ddl: &str,
) -> Result<(), DomainError> {
    match column_declaration(conn, table, column)? {
        Some(declared) => {
            if let Some(expected) = declared_type_in(ddl, column)
                && !declared.is_empty()
                && !declared.eq_ignore_ascii_case(expected)
            {
                warn!(
                    "⚠️ {table}.{column} is declared as '{declared}' where the schema expects \
                     '{expected}': not migrating it (SQLite does not retype a column in place), but \
                     a value written under the other type may not round-trip"
                );
            }
            debug!("🧱 {table}.{column} already exists: nothing to migrate");
            Ok(())
        }
        None => conn
            .execute(ddl, [])
            .map(|_| ())
            .map_err(|e| DomainError::Storage(format!("Schema migration failed ({ddl}): {e}"))),
    }
}

/// Renames `from` to `to` when the old name is the one the database has (`TASK-42`/S48).
///
/// `ALTER TABLE … RENAME COLUMN` fails when the column does not exist, so the old `let _ =` also hid
/// the case that matters most: a `weight_manifests` with **neither** name, i.e. a database this build
/// cannot read. That now refuses to boot naming both candidates instead of carrying on with a schema
/// nobody understands.
pub fn ensure_column_renamed(
    conn: &Connection,
    table: &str,
    from: &str,
    to: &str,
) -> Result<(), DomainError> {
    if column_declaration(conn, table, to)?.is_some() {
        debug!("🧱 {table}.{to} already exists: nothing to rename");
        return Ok(());
    }
    if column_declaration(conn, table, from)?.is_none() {
        return Err(DomainError::Storage(format!(
            "{table} has neither '{from}' nor '{to}': the table is not the one this build expects \
             (schema drift), refusing to boot on a database it cannot read"
        )));
    }

    let ddl = format!("ALTER TABLE {table} RENAME COLUMN {from} TO {to}");
    conn.execute(&ddl, [])
        .map(|_| ())
        .map_err(|e| DomainError::Storage(format!("Schema migration failed ({ddl}): {e}")))?;
    info!("🧱 Renamed {table}.{from} → {to}");
    Ok(())
}

/// Makes sure `idx_ledger_dedup` is the **content** key and not the old per-second one (SP-02).
///
/// The migration keeps the index **name** on purpose: an operator who reverts the commit finds an
/// index of that name already there (`IF NOT EXISTS`), so going back cannot fail on a database that
/// contains two entries the old key would have collapsed. It is a no-op when the shape is already
/// the wanted one, so it does not rebuild anything on a normal boot.
pub fn ensure_ledger_dedup_index(conn: &rusqlite::Connection) -> Result<(), rusqlite::Error> {
    const WANTED: [&str; 2] = ["source", "source_hash"];

    let current: Vec<String> = {
        let mut stmt = conn.prepare(&format!("PRAGMA index_info({LEDGER_DEDUP_INDEX})"))?;
        let columns: Result<Vec<String>, rusqlite::Error> =
            stmt.query_map([], |row| row.get::<_, String>(2))?.collect();
        columns?
    };

    if current == WANTED {
        return Ok(());
    }

    conn.execute(&format!("DROP INDEX IF EXISTS {LEDGER_DEDUP_INDEX}"), [])?;
    conn.execute(
        &format!(
            "CREATE UNIQUE INDEX IF NOT EXISTS {LEDGER_DEDUP_INDEX} \
             ON ledger(source, source_hash) WHERE source = 'federated'"
        ),
        [],
    )?;

    // A brand-new database has no index at all: creating it is not «migrating» anything and saying so
    // would be noise (and a lie about what happened).
    if current.is_empty() {
        tracing::debug!("🧱 Created {LEDGER_DEDUP_INDEX} on (source, source_hash)");
    } else {
        tracing::info!(
            "🧱 Migrated {LEDGER_DEDUP_INDEX} from {:?} to {:?}: the old key had one-second \
             resolution and collapsed distinct federated entries (SP-02)",
            current,
            WANTED
        );
    }

    Ok(())
}

/// Initialises pragmas, creates initial schema tables, and executes all incremental migrations.
pub fn run_migrations(conn: &Connection) -> Result<(), DomainError> {
    conn.execute_batch(&format!(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};"
    ))
    .map_err(|e| DomainError::Storage(format!("Failed to enable WAL: {}", e)))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS peer_credits (
            peer_id TEXT PRIMARY KEY,
            credits_earned REAL DEFAULT 0.0,
            credits_spent REAL DEFAULT 0.0,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS karma_ledger (
            peer_id TEXT PRIMARY KEY,
            balance INTEGER NOT NULL DEFAULT 0
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS discovered_nodes (
            peer_id TEXT PRIMARY KEY,
            host TEXT,
            port INTEGER,
            vram_mb INTEGER,
            role TEXT,
            model_id TEXT,
            p2p_address TEXT,
            http_api_url TEXT,
            relay_address TEXT,
            status TEXT DEFAULT 'Healthy',
            failed_requests INTEGER DEFAULT 0,
            successful_requests INTEGER DEFAULT 0,
            latency_p99_ms INTEGER DEFAULT 0,
            last_error TEXT,
            slash_count INTEGER DEFAULT 0,
            blacklisted BOOLEAN DEFAULT 0,
            queue_depth INTEGER DEFAULT 0,
            public_ip TEXT,
            public_api_port INTEGER,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    for (column, ddl) in [
        (
            "http_api_url",
            "ALTER TABLE discovered_nodes ADD COLUMN http_api_url TEXT",
        ),
        (
            "failed_requests",
            "ALTER TABLE discovered_nodes ADD COLUMN failed_requests INTEGER DEFAULT 0",
        ),
        (
            "successful_requests",
            "ALTER TABLE discovered_nodes ADD COLUMN successful_requests INTEGER DEFAULT 0",
        ),
        (
            "latency_p99_ms",
            "ALTER TABLE discovered_nodes ADD COLUMN latency_p99_ms INTEGER DEFAULT 0",
        ),
        (
            "last_error",
            "ALTER TABLE discovered_nodes ADD COLUMN last_error TEXT",
        ),
        (
            "slash_count",
            "ALTER TABLE discovered_nodes ADD COLUMN slash_count INTEGER DEFAULT 0",
        ),
        (
            "blacklisted",
            "ALTER TABLE discovered_nodes ADD COLUMN blacklisted BOOLEAN DEFAULT 0",
        ),
        (
            "queue_depth",
            "ALTER TABLE discovered_nodes ADD COLUMN queue_depth INTEGER DEFAULT 0",
        ),
        (
            "public_ip",
            "ALTER TABLE discovered_nodes ADD COLUMN public_ip TEXT",
        ),
        (
            "public_api_port",
            "ALTER TABLE discovered_nodes ADD COLUMN public_api_port INTEGER",
        ),
    ] {
        ensure_column(conn, "discovered_nodes", column, ddl)?;
    }

    conn.execute(
        "CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            peer_id TEXT,
            entry_type TEXT,
            amount REAL,
            proof TEXT,
            request_id TEXT,
            output_hash BLOB,
            source TEXT DEFAULT 'local',
            source_hash TEXT,
            ts INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    ensure_ledger_dedup_index(conn).map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value INTEGER
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS active_worker_assignment (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL,
            assignment_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS reputation_scores (
            peer_id TEXT PRIMARY KEY,
            score REAL DEFAULT 1.0,
            last_updated INTEGER DEFAULT 0
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reputation_peer_id ON reputation_scores(peer_id)",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS weight_manifests (
            model_id TEXT NOT NULL,
            filename TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL,
            blake3_hash BLOB NOT NULL,
            size_bytes INTEGER NOT NULL,
            registered_at INTEGER NOT NULL,
            tracker_public_key BLOB NOT NULL,
            tracker_signature BLOB NOT NULL,
            PRIMARY KEY (model_id, role)
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS transfer_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_peer_id TEXT NOT NULL,
            receiver_peer_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            role TEXT NOT NULL,
            blake3_hash BLOB NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            timestamp INTEGER NOT NULL,
            receipt_json TEXT NOT NULL
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            peer_id TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS failure_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reporter_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_failure_reports \
         ON failure_reports(reporter_id, target_id, timestamp)",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS pending_work_proofs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serialized_proof TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS work_tickets (
            request_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            entry_peer_id TEXT NOT NULL,
            issued_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS work_ticket_consumptions (
            request_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            credits REAL NOT NULL,
            consumed_at INTEGER NOT NULL,
            PRIMARY KEY (request_id, worker_id)
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS known_peers (
            peer_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    conn.execute(
        "INSERT OR IGNORE INTO ledger_meta (key, value) VALUES ('federation_cursor', 0)",
        [],
    )
    .map_err(|e| DomainError::Storage(e.to_string()))?;

    for (table, column, ddl) in [
        (
            "discovered_nodes",
            "status",
            "ALTER TABLE discovered_nodes ADD COLUMN status TEXT DEFAULT 'Healthy'",
        ),
        (
            "ledger",
            "ts",
            "ALTER TABLE ledger ADD COLUMN ts INTEGER DEFAULT 0",
        ),
        (
            "weight_manifests",
            "filename",
            "ALTER TABLE weight_manifests ADD COLUMN filename TEXT NOT NULL DEFAULT ''",
        ),
        (
            "transfer_receipts",
            "size_bytes",
            "ALTER TABLE transfer_receipts ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0",
        ),
    ] {
        ensure_column(conn, table, column, ddl)?;
    }

    ensure_column_renamed(
        conn,
        "weight_manifests",
        "musivum_public_key",
        "tracker_public_key",
    )?;
    ensure_column_renamed(
        conn,
        "weight_manifests",
        "musivum_signature",
        "tracker_signature",
    )?;

    Ok(())
}
