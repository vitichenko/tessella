# tessella-storage

> Local persistence for models, configuration, and ledger.

## Responsibility

Handles disk access. Implements the SQLite database used by the worker to persist its local state, including the signed work-proof Write-Ahead Log (`pending_work_proofs`). Also abstracts the local model cache (HuggingFace cache). Depends on `rusqlite` and `tessella-core`.

## Modules

| File | Content |
|---------|-----------|
| `src/lib.rs` | `SqliteStorage` abstractions and safe CRUD operations. |
| `src/hf_downloader.rs` | Complementary tools for file caching. |

## Usage

```rust
use tessella_storage::SqliteStorage;
```
