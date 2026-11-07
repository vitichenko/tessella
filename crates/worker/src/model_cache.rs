//! Opt-in pruning of the model cache (`TASK-73` · S80 · finding F-5 of the S74 spike).
//!
//! # The problem, measured
//!
//! The HuggingFace cache is a **global** directory shared by every node of the machine
//! (`~/.cache/huggingface`, **11 GB** with six models on the reference machine) and the unit of a
//! download is the **shard**, never the layer: serving **one** layer of a 7B pulls a **4 877,7 MB**
//! file, and three layers of a 0,5B pull its whole **988,1 MB** `model.safetensors` (measured in S80).
//! `hf_loader::resolve_shards_for_layers` downloads only the shards the *current* range needs, but
//! until S80 nothing removed what a *previous* range needed: `reload_shard` adopted the new range and
//! left everything else on disk (`grep -rn 'remove_file' crates/inference/src` = 0). A node whose range
//! moves three times ends up holding **every shard of the model** (measured on the real 7B index:
//! ranges `[0,9)`, `[9,19)`, `[19,28)` touch 3 of 3 shards, and the last range needs 1) — dead weight
//! forever, on a node whose disk is as finite as its VRAM.
//!
//! # What this module does
//!
//! [`prune`] walks the directory of **this node's own model** and deletes what the new assignment does
//! not need: the `.safetensors`/`.gguf` shards it no longer serves. In the HuggingFace layout a shard
//! is a **symlink** in `snapshots/<rev>/` pointing into `blobs/`, so the symlink is removed first and
//! the blob only when nothing else under the model directory still resolves to it — which is what
//! keeps `config.json`, `tokenizer.json` and another revision of the same model alive.
//!
//! # The three guards (each one has a test in this file)
//!
//! 1. **Off by default** ([`CachePruneMode::Off`], `model_cache_prune`): nothing is even looked at, so
//!    a mesh behaves exactly as it did before S80.
//! 2. **Confinement**: only the model directory the shards live in — never the hub root, never a
//!    sibling model, never a shard whose target escapes that directory. The needed files are compared
//!    **canonically**, so a symlink and its blob are seen as one thing.
//! 3. **No other live process**: every node that loads weights takes a **shared** lock on
//!    `<model dir>/.tessella-cache.lock` for as long as it runs ([`hold_shards`], called from
//!    `NetworkManager::acquire_weights`); the prune converts *its own* lock to exclusive with
//!    `try_lock`, which fails while anybody else holds a shared one, and downgrades back afterwards.
//!    Measured on this machine (S80): converting the process's own lock is granted, and another
//!    process holding a shared lock makes `try_lock` return `WouldBlock` — both directions are what
//!    this module relies on.
//!
//! It is deliberately **not** a garbage collector: it never walks the hub root, it never deletes the
//! `refs/` of a repo, and it does not know about models this node never served.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use tracing::{debug, info, warn};

use crate::config::CachePruneMode;

/// The lock file every node that serves a model holds a **shared** lock on. It lives inside the model
/// directory (not the hub root), so the lock is per model and two models never contend.
pub const LOCK_FILE_NAME: &str = ".tessella-cache.lock";

/// The extensions a shard can have on disk. Everything else (`config.json`, `tokenizer.json`, the
/// `refs/` of a repo, the HF blobs themselves, which are extension-less) is never a prune candidate.
const SHARD_EXTENSIONS: [&str; 2] = ["safetensors", "gguf"];

/// The locks this process holds, keyed by model directory. A `File` is kept alive for the whole life
/// of the process on purpose: dropping it would release the lock, which is exactly what makes a
/// neighbour's prune able to delete the shards this node is serving.
static HELD_LOCKS: LazyLock<Mutex<HashMap<PathBuf, File>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The directory of the **model** a shard belongs to: never the shared hub root.
///
/// Two layouts exist on disk and both are understood:
///
/// * the HuggingFace cache, `…/hub/models--<org>--<name>/snapshots/<rev>/model-00001-of-….safetensors`
///   ⇒ the `models--…` directory, because its `blobs/` and `snapshots/` are shared by **every**
///   revision of that model and stepping above it would reach a sibling model;
/// * a local `models_dir`, `…/models/<name>/model.safetensors` ⇒ the shard's own directory, and only
///   when that directory really is a model directory (`config.json` inside it). A shard dropped
///   directly into `models_dir` has no model directory to confine the prune to, so it is refused
///   (fail closed) instead of pruning every model in the folder.
pub fn model_root_of(shard_paths: &[PathBuf]) -> Option<PathBuf> {
    let first = shard_paths.first()?;
    for ancestor in first.ancestors().skip(1) {
        let is_hub_model = ancestor
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("models--"));
        if is_hub_model {
            return Some(ancestor.to_path_buf());
        }
    }
    let parent = first.parent()?;
    parent
        .join("config.json")
        .is_file()
        .then(|| parent.to_path_buf())
}

/// The path of the lock file of a model directory.
fn lock_path(root: &Path) -> PathBuf {
    root.join(LOCK_FILE_NAME)
}

/// Opens (creating it if needed) the lock file of a model directory. `write` is requested because a
/// lock on a read-only file descriptor is refused on some platforms.
fn open_lock_file(root: &Path) -> io::Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path(root))
}

/// The held locks of this process, tolerating a poisoned mutex: the map only ever holds file handles.
fn held_locks() -> std::sync::MutexGuard<'static, HashMap<PathBuf, File>> {
    HELD_LOCKS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Takes a **shared** lock on the model directory of these shards and holds it for the life of the
/// process. Idempotent per directory, and never blocking on a neighbour: shared locks are compatible,
/// so a machine running several nodes of the same model keeps working.
///
/// It is called by every path that loads weights (`NetworkManager::acquire_weights`), which is what
/// turns "is anybody else serving this model right now?" into an answer the kernel gives. Returns the
/// directories it now holds (empty when it already held them or when there is nothing to hold).
pub fn hold_shards(shard_paths: &[PathBuf]) -> Vec<PathBuf> {
    let Some(root) = model_root_of(shard_paths) else {
        return Vec::new();
    };
    let mut held = held_locks();
    if held.contains_key(&root) {
        return Vec::new();
    }
    match open_lock_file(&root).and_then(|file| file.lock_shared().map(|()| file)) {
        Ok(file) => {
            held.insert(root.clone(), file);
            debug!(
                model_dir = %root.display(),
                "holding a shared lock on the model cache (no neighbour will prune it while this \
                 process lives)"
            );
            vec![root]
        }
        Err(error) => {
            warn!(
                model_dir = %root.display(),
                %error,
                "could not take the shared lock on the model cache: this node will not protect its \
                 shards from a neighbour's prune (the load itself is unaffected)"
            );
            Vec::new()
        }
    }
}

/// A file the prune removed (or, in `dry-run`, would remove).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeletedFile {
    /// The path as it was on disk (a symlink of a HF snapshot, a local shard file, or a blob).
    pub path: PathBuf,
    /// Bytes freed by removing it. A symlink whose blob is kept frees 0.
    pub bytes: u64,
}

/// What a prune did, or would do.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PruneReport {
    /// The model directory the prune was confined to.
    pub root: PathBuf,
    /// The files it removed (or, in `dry-run`, the plan).
    pub deleted: Vec<DeletedFile>,
    /// The bytes those files occupied.
    pub freed_bytes: u64,
    /// Shard files the new assignment still needs, left alone.
    pub kept: usize,
    /// `true` when nothing was removed.
    pub dry_run: bool,
}

/// The result of asking for a prune.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PruneOutcome {
    /// `model_cache_prune = off` (the default): nothing was looked at.
    Disabled,
    /// The shard paths do not identify a model directory (see [`model_root_of`]): fail closed.
    UnknownModelRoot,
    /// Another live process holds this model open: **not a single file** was touched.
    LockedByAnotherProcess { root: PathBuf },
    /// The lock file could not be created or locked: fail closed.
    LockFailed { root: PathBuf, error: String },
    /// The cache was inspected (and, unless `dry_run`, pruned).
    Inspected(PruneReport),
}

/// Prunes the shards of `root` that the new assignment does not need, keeping every path in `needed`.
///
/// The comparison with `needed` is **canonical**, so passing the symlink of a HF snapshot protects its
/// blob too. Nothing outside `root` is ever touched, and nothing at all is touched when another live
/// process holds the model (guard 3 of the module docs).
pub fn prune(mode: CachePruneMode, root: &Path, needed: &HashSet<PathBuf>) -> PruneOutcome {
    if !mode.is_on() {
        return PruneOutcome::Disabled;
    }
    let guard = match exclusive_guard(root) {
        Ok(Some(guard)) => guard,
        Ok(None) => {
            return PruneOutcome::LockedByAnotherProcess {
                root: root.to_path_buf(),
            }
        }
        Err(error) => {
            return PruneOutcome::LockFailed {
                root: root.to_path_buf(),
                error: error.to_string(),
            }
        }
    };
    let report = inspect_and_delete(mode, root, needed);
    drop(guard); // downgrades the lock back to shared: the node keeps protecting its own cache
    PruneOutcome::Inspected(report)
}

/// Prunes the cache of the model these shards belong to, keeping exactly them.
///
/// This is what `NetworkManager::reload_shard` calls once the new range is live: `shard_paths` are the
/// shards the new assignment resolved, so they are the files that must stay. Returns `None` when the
/// prune is off (the default): there is nothing to log, because nothing happened.
pub fn prune_after_reload(mode: CachePruneMode, shard_paths: &[PathBuf]) -> Option<PruneOutcome> {
    if !mode.is_on() {
        return None;
    }
    let Some(root) = model_root_of(shard_paths) else {
        return Some(PruneOutcome::UnknownModelRoot);
    };
    let needed: HashSet<PathBuf> = shard_paths.iter().cloned().collect();
    Some(prune(mode, &root, &needed))
}

/// Reports an outcome the way the node reports it: silent when the prune is off, the whole plan when
/// it is a `dry-run` (that mode exists to be read), and one line per deleted file afterwards.
pub fn log_outcome(outcome: &PruneOutcome) {
    match outcome {
        PruneOutcome::Disabled => {}
        PruneOutcome::UnknownModelRoot => warn!(
            "a cache prune was requested but the shard paths do not identify a model directory: \
             nothing was inspected (fail closed)"
        ),
        PruneOutcome::LockedByAnotherProcess { root } => info!(
            model_dir = %root.display(),
            "another live process is serving this model: the cache was NOT pruned"
        ),
        PruneOutcome::LockFailed { root, error } => warn!(
            model_dir = %root.display(),
            %error,
            "the cache lock could not be taken: nothing was pruned (fail closed)"
        ),
        PruneOutcome::Inspected(report) if report.deleted.is_empty() => debug!(
            model_dir = %report.root.display(),
            kept = report.kept,
            "the model cache holds nothing this range does not need"
        ),
        PruneOutcome::Inspected(report) if report.dry_run => {
            info!(
                model_dir = %report.root.display(),
                files = report.deleted.len(),
                freed_mb = report.freed_bytes / (1024 * 1024),
                kept = report.kept,
                "model_cache_prune = dry-run: this is what would be deleted"
            );
            for file in &report.deleted {
                info!(
                    path = %file.path.display(),
                    mb = file.bytes / (1024 * 1024),
                    "  (dry-run) would delete"
                );
            }
        }
        PruneOutcome::Inspected(report) => {
            info!(
                model_dir = %report.root.display(),
                files = report.deleted.len(),
                freed_mb = report.freed_bytes / (1024 * 1024),
                kept = report.kept,
                "🧹 model cache pruned: the shards this node no longer serves are gone"
            );
            for file in &report.deleted {
                debug!(path = %file.path.display(), mb = file.bytes / (1024 * 1024), "deleted");
            }
        }
    }
}

/// A shard found on disk, with everything the prune needs to decide about it.
struct Candidate {
    /// The path as it is in the directory walk (the symlink of a HF snapshot, or a local file).
    path: PathBuf,
    /// Where it really points (canonical): itself for a local file, the blob for a HF symlink.
    target: PathBuf,
    /// The size of `target` (a symlink itself occupies ~0, its blob does not).
    bytes: u64,
    /// Whether `path` is a symlink.
    symlink: bool,
}

/// Holds the **exclusive** lock of a prune and downgrades it back to shared when dropped, so the node
/// that pruned keeps protecting its own cache for the rest of its life. Without this, a node that
/// holds a shared lock would leave the directory unprotected (or locked exclusively) after pruning.
struct PruneGuard(File);

impl Drop for PruneGuard {
    fn drop(&mut self) {
        if let Err(error) = self.0.lock_shared() {
            warn!(
                %error,
                "could not downgrade the cache lock back to shared after the prune: this node keeps \
                 the model directory to itself until it exits"
            );
        }
    }
}

/// Takes the exclusive lock of the model directory. `Ok(None)` means another live process holds it,
/// which is the answer the caller must respect: not a single file is touched.
fn exclusive_guard(root: &Path) -> io::Result<Option<PruneGuard>> {
    // The lock this process already holds (it loaded this model) is reused through `try_clone`: `dup`
    // shares the open file description, so `try_lock` *converts* our own shared lock instead of adding
    // one — which is how a node prunes its own cache while a neighbour would block it.
    let held = held_locks().get(root).map(File::try_clone).transpose()?;
    let file = match held {
        Some(file) => file,
        None => open_lock_file(root)?,
    };
    match file.try_lock() {
        Ok(()) => Ok(Some(PruneGuard(file))),
        // Measured in S80: a shared lock held by *another* process (or a plain `flock -s`) makes this
        // return `WouldBlock`, while converting this process's own lock is granted.
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(error)) => Err(error),
    }
}

/// Whether a file name is a shard (and therefore a prune candidate).
fn is_shard(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            let extension = extension.to_ascii_lowercase();
            SHARD_EXTENSIONS.contains(&extension.as_str())
        })
}

/// Every shard under `root`, without following directory symlinks (the HF layout has none, and a
/// symlinked directory would leave the confinement).
fn scan_shards(root: &Path) -> Vec<Candidate> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(path);
                continue;
            }
            if !is_shard(&path) {
                continue;
            }
            let Ok(target) = fs::canonicalize(&path) else {
                continue;
            };
            found.push(Candidate {
                bytes: fs::metadata(&path).map(|data| data.len()).unwrap_or(0),
                symlink: metadata.file_type().is_symlink(),
                path,
                target,
            });
        }
    }
    found.sort_by(|left, right| left.path.cmp(&right.path));
    found
}

/// The canonical form of every path of a set, keeping the original when it cannot be resolved.
fn canonical_of(paths: &HashSet<PathBuf>) -> HashSet<PathBuf> {
    paths
        .iter()
        .map(|path| fs::canonicalize(path).unwrap_or_else(|_| path.clone()))
        .collect()
}

/// Every path that a **surviving reference** under `root` still resolves to, ignoring the entries this
/// prune is removing. It is what keeps a blob alive for `config.json`, `tokenizer.json` or another
/// revision of the same model, and it is computed the same way in `dry-run` (where nothing has been
/// removed yet) and in a real prune (where the shards are already gone).
///
/// Only entries that *point somewhere else* count as a reference: the payload files inside `blobs/` of
/// a HuggingFace cache are extension-less blobs themselves, and letting one be its own reference would
/// keep **every** blob alive forever (the `dry-run` test caught exactly that: the orphan blob never
/// reached the plan). A local layout has no blobs to protect — there the shard *is* the file, and it is
/// the canonical comparison with `needed` that keeps it.
fn surviving_targets(root: &Path, removed: &HashSet<PathBuf>) -> HashSet<PathBuf> {
    let mut survivors = HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(path);
                continue;
            }
            if removed.contains(&path) || !metadata.file_type().is_symlink() {
                continue;
            }
            if let Ok(target) = fs::canonicalize(&path) {
                survivors.insert(target);
            }
        }
    }
    survivors
}

/// Confines the walk to the model directory (or refuses), removes what the new range does not need,
/// and removes a blob only when nothing that survives still resolves to it.
fn inspect_and_delete(mode: CachePruneMode, root: &Path, needed: &HashSet<PathBuf>) -> PruneReport {
    let mut report = PruneReport {
        root: root.to_path_buf(),
        dry_run: !mode.deletes(),
        ..PruneReport::default()
    };
    let canonical_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let needed_canonical = canonical_of(needed);

    // Decide: every shard of this model that the new assignment does not need.
    let mut unneeded = Vec::new();
    for candidate in scan_shards(root) {
        let is_needed =
            needed.contains(&candidate.path) || needed_canonical.contains(&candidate.target);
        if is_needed {
            report.kept += 1;
            continue;
        }
        if !candidate.target.starts_with(&canonical_root) {
            // Guard 2: a shard whose target escapes the model directory is reported and left alone.
            warn!(
                shard = %candidate.path.display(),
                target = %candidate.target.display(),
                model_dir = %root.display(),
                "a shard of this model does not point inside its own directory: left alone"
            );
            report.kept += 1;
            continue;
        }
        unneeded.push(candidate);
    }

    // Pass 1: the shards themselves — the symlink of a HF snapshot, or the file of a local layout.
    let mut blobs: Vec<PathBuf> = Vec::new();
    let mut removed: HashSet<PathBuf> = HashSet::new();
    let mut failures = 0_usize;
    for candidate in &unneeded {
        if mode.deletes() {
            if let Err(error) = fs::remove_file(&candidate.path) {
                warn!(
                    path = %candidate.path.display(),
                    %error,
                    "could not delete a shard this range does not need"
                );
                failures += 1;
                report.kept += 1;
                continue;
            }
        }
        removed.insert(candidate.path.clone());
        if candidate.symlink {
            blobs.push(candidate.target.clone());
            report.deleted.push(DeletedFile {
                path: candidate.path.clone(),
                bytes: 0,
            });
        } else {
            report.freed_bytes += candidate.bytes;
            report.deleted.push(DeletedFile {
                path: candidate.path.clone(),
                bytes: candidate.bytes,
            });
        }
    }

    // Pass 2: the blobs of the HF layout. Skipped entirely when a shard could not be removed, because
    // the orphan check below is only sound if pass 1 did what it said it would.
    if failures == 0 {
        blobs.sort();
        blobs.dedup();
        let survivors = surviving_targets(root, &removed);
        for blob in blobs {
            if needed_canonical.contains(&blob) || survivors.contains(&blob) {
                debug!(
                    blob = %blob.display(),
                    "the blob of a pruned shard is still referenced by this model: kept"
                );
                continue;
            }
            let bytes = fs::metadata(&blob).map(|data| data.len()).unwrap_or(0);
            if mode.deletes() {
                if let Err(error) = fs::remove_file(&blob) {
                    warn!(
                        path = %blob.display(),
                        %error,
                        "could not delete the blob of a shard this range does not need"
                    );
                    continue;
                }
            }
            report.freed_bytes += bytes;
            report.deleted.push(DeletedFile { path: blob, bytes });
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway tree under `$TMPDIR` that deletes itself at the end of the test.
    struct TempTree(PathBuf);

    impl TempTree {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("tessella-s80-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create the temporary tree");
            Self(path)
        }

        fn join(&self, relative: &str) -> PathBuf {
            self.0.join(relative)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Whether a path exists **as an entry**: a dangling symlink still counts.
    fn exists(path: &Path) -> bool {
        fs::symlink_metadata(path).is_ok()
    }

    fn write_file(path: &Path, bytes: usize) {
        fs::create_dir_all(path.parent().expect("a parent directory")).expect("create the parent");
        fs::write(path, vec![7_u8; bytes]).expect("write the file");
    }

    /// Copies a directory **as it is**, symlinks included: a HuggingFace cache is nothing but symlinks
    /// into `blobs`, so a copy that followed them would not be the same layout.
    fn copy_tree(from: &Path, to: &Path) {
        let metadata = fs::symlink_metadata(from).expect("read the source");
        if metadata.file_type().is_symlink() {
            let target = fs::read_link(from).expect("read the link");
            std::os::unix::fs::symlink(target, to).expect("recreate the link");
            return;
        }
        if metadata.is_dir() {
            fs::create_dir_all(to).expect("create the directory");
            for entry in fs::read_dir(from)
                .expect("read the source directory")
                .flatten()
            {
                copy_tree(&entry.path(), &to.join(entry.file_name()));
            }
            return;
        }
        fs::copy(from, to).expect("copy the file");
    }

    /// The bytes a tree really holds: **regular files only**. A symlink is not its target, and the prune
    /// does not count it either, so the two numbers can be added up (`freed_bytes` includes a blob once,
    /// never twice for two shards that point at it).
    fn tree_bytes(root: &Path) -> u64 {
        let mut total = 0;
        let mut stack = vec![root.to_path_buf()];
        while let Some(directory) = stack.pop() {
            let Ok(entries) = fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(metadata) = fs::symlink_metadata(entry.path()) else {
                    continue;
                };
                if metadata.is_dir() {
                    stack.push(entry.path());
                } else if !metadata.file_type().is_symlink() {
                    total += metadata.len();
                }
            }
        }
        total
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    /// The real repository the measurement runs on: the one `TESSELLA_S80_MEASURE_REPO` names, or the
    /// **smallest** one of the developer's HuggingFace cache that holds a shard. The measurement copies it,
    /// so the point is the arithmetic, not the size of the model.
    fn real_repo_to_measure() -> Option<PathBuf> {
        if let Ok(explicit) = std::env::var("TESSELLA_S80_MEASURE_REPO") {
            return Some(PathBuf::from(explicit));
        }
        let cache = std::env::var("HF_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface")
            })
            .join("hub");
        let mut repos: Vec<(u64, PathBuf)> = fs::read_dir(&cache)
            .ok()?
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("models--"))
            .filter_map(|entry| {
                let path = entry.path();
                (!scan_shards(&path).is_empty()).then(|| (tree_bytes(&path), path))
            })
            .collect();
        repos.sort();
        repos.into_iter().next().map(|(_, path)| path)
    }

    /// Builds a HuggingFace-shaped model cache: `hub/models--acme--tiny/{blobs,snapshots/<rev>}`, with
    /// each shard a symlink of the snapshot pointing into `blobs` (the real layout). Returns the tree,
    /// the model directory and one shard path per name, in the order given.
    ///
    /// The blob of a shard is `blobs/blob-<shard name without extension>`, and the extension is left out
    /// on purpose: the real HF cache names its blobs after the content hash and **without** extension,
    /// which is what keeps them out of the shard scan of the prune (they are only reachable through the
    /// symlink that points at them).
    fn hf_cache(
        name: &str,
        rev: &str,
        shards: &[(&str, usize)],
    ) -> (TempTree, PathBuf, Vec<PathBuf>) {
        let tree = TempTree::new(name);
        let model = tree.join("hub/models--acme--tiny");
        let blobs = model.join("blobs");
        let snapshot = model.join(format!("snapshots/{rev}"));
        fs::create_dir_all(&blobs).expect("create blobs");
        fs::create_dir_all(&snapshot).expect("create the snapshot");
        // `config.json` is a real file here on purpose: the prune never looks at anything that is not
        // a shard, whatever its shape.
        write_file(&snapshot.join("config.json"), 16);
        let mut paths = Vec::new();
        for (shard, bytes) in shards {
            let blob = blobs.join(format!("blob-{}", shard.trim_end_matches(".safetensors")));
            write_file(&blob, *bytes);
            let link = snapshot.join(shard);
            std::os::unix::fs::symlink(&blob, &link).expect("symlink the shard");
            paths.push(link);
        }
        (tree, model, paths)
    }

    #[test]
    fn model_root_of_confines_to_the_model_directory_of_each_layout() {
        let (tree, model, shards) = hf_cache(
            "root-hf",
            "rev1",
            &[("model-00001-of-00002.safetensors", 64)],
        );
        assert_eq!(
            model_root_of(&shards),
            Some(model.clone()),
            "the hug hub root and the snapshot directory must both stay out of reach"
        );
        drop(tree);

        // Local layout: the shard's own directory, but only when it really is a model directory.
        let local = TempTree::new("root-local");
        let local_shard = local.join("models/tiny/model.safetensors");
        write_file(&local_shard, 64);
        assert_eq!(
            model_root_of(std::slice::from_ref(&local_shard)),
            None,
            "a shard without a config.json next to it does not identify a model: fail closed"
        );
        write_file(&local.join("models/tiny/config.json"), 16);
        assert_eq!(
            model_root_of(std::slice::from_ref(&local_shard)),
            Some(local.join("models/tiny")),
            "the local layout is confined to the model directory, not to models_dir"
        );
        assert_eq!(model_root_of(&[]), None, "no shard, no root");
    }

    #[test]
    fn by_default_the_prune_does_not_even_look_at_the_cache() {
        let (tree, model, shards) = hf_cache(
            "default-off",
            "rev1",
            &[
                ("model-00001-of-00002.safetensors", 128),
                ("model-00002-of-00002.safetensors", 256),
            ],
        );
        let needed: HashSet<PathBuf> = [shards[0].clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::Off, &model, &needed);
        assert_eq!(
            outcome,
            PruneOutcome::Disabled,
            "off is the default: nothing happens"
        );

        for shard in &shards {
            assert!(
                exists(shard),
                "{} must still be there with the prune off",
                shard.display()
            );
        }
        assert_eq!(
            fs::read_dir(model.join("blobs")).expect("blobs").count(),
            2,
            "not a single blob may be reclaimed with the prune off"
        );
        assert_eq!(
            prune_after_reload(CachePruneMode::Off, &shards),
            None,
            "with the prune off the caller has nothing to log either"
        );
        drop(tree);
    }

    #[test]
    fn dry_run_lists_the_plan_and_deletes_nothing() {
        let (tree, model, shards) = hf_cache(
            "dry-run",
            "rev1",
            &[
                ("model-00001-of-00002.safetensors", 128),
                ("model-00002-of-00002.safetensors", 4096),
            ],
        );
        let needed: HashSet<PathBuf> = [shards[0].clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::DryRun, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("a dry run must report its plan, got {outcome:?}");
        };
        assert!(
            report.dry_run,
            "the report must say that nothing was deleted"
        );
        assert_eq!(
            report.kept, 1,
            "the shard of the new range is the one that stays"
        );
        let planned: Vec<&Path> = report
            .deleted
            .iter()
            .map(|file| file.path.as_path())
            .collect();
        assert!(
            planned.contains(&shards[1].as_path()),
            "the shard of the old range is in the plan: {planned:?}"
        );
        assert!(
            report
                .deleted
                .iter()
                .any(|file| file.path.ends_with("blob-model-00002-of-00002")),
            "the orphan blob is in the plan too (a dry run must not overstate what it would free)"
        );
        assert_eq!(report.freed_bytes, 4096, "the freed bytes of the old shard");

        for shard in &shards {
            assert!(
                exists(shard),
                "a dry run deletes nothing: {} is still there",
                shard.display()
            );
        }
        assert_eq!(
            fs::read_dir(model.join("blobs")).expect("blobs").count(),
            2,
            "a dry run deletes no blob"
        );
        drop(tree);
    }

    #[test]
    fn on_deletes_the_unneeded_shard_and_its_orphan_blob() {
        let (tree, model, shards) = hf_cache(
            "prune-on",
            "rev1",
            &[
                ("model-00001-of-00002.safetensors", 128),
                ("model-00002-of-00002.safetensors", 4096),
            ],
        );
        let needed: HashSet<PathBuf> = [shards[0].clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::On, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("the prune must report what it did, got {outcome:?}");
        };
        assert!(!report.dry_run, "this one really ran");
        assert!(exists(&shards[0]), "the shard of the new range survives");
        assert!(
            !exists(&shards[1]),
            "the shard of the old range is gone: {}",
            shards[1].display()
        );
        assert_eq!(
            report.freed_bytes, 4096,
            "the whole shard is reclaimed, not a layer"
        );
        assert_eq!(
            report.deleted.len(),
            2,
            "the symlink of the snapshot and its orphan blob: {:?}",
            report.deleted
        );
        let blobs: Vec<PathBuf> = fs::read_dir(model.join("blobs"))
            .expect("blobs")
            .map(|entry| entry.expect("entry").path())
            .collect();
        assert_eq!(
            blobs.len(),
            1,
            "only the blob of the surviving shard is left: {blobs:?}"
        );
        assert!(
            exists(&model.join("snapshots/rev1/config.json")),
            "a file that is not a shard is never a candidate"
        );
        drop(tree);
    }

    #[test]
    fn the_prune_never_leaves_the_model_directory() {
        let (tree, model, shards) = hf_cache(
            "confine",
            "rev1",
            &[
                ("model-00001-of-00002.safetensors", 128),
                ("model-00002-of-00002.safetensors", 256),
            ],
        );
        // A sibling model of the same hub and a loose shard in the hub root: both out of reach.
        let other = tree.join("hub/models--acme--other");
        write_file(&other.join("blobs/blob-other"), 512);
        let other_shard = other.join("snapshots/rev1/model-00001-of-00001.safetensors");
        fs::create_dir_all(other_shard.parent().expect("snapshot")).expect("create the snapshot");
        std::os::unix::fs::symlink(other.join("blobs/blob-other"), &other_shard)
            .expect("symlink the sibling shard");
        let loose = tree.join("hub/loose.safetensors");
        write_file(&loose, 32);
        let needed: HashSet<PathBuf> = [shards[0].clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::On, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("the prune must report what it did, got {outcome:?}");
        };

        assert!(
            exists(&other_shard),
            "a sibling model of the hub is out of reach"
        );
        assert!(
            exists(&other.join("blobs/blob-other")),
            "and so is its blob"
        );
        assert!(
            exists(&loose),
            "a loose shard of the hub root is out of reach too"
        );
        assert!(
            exists(&shards[0]),
            "the shard of the new range of this model stays"
        );
        assert!(!exists(&shards[1]), "the unneeded shard of this model goes");
        assert_eq!(
            report.deleted.len(),
            2,
            "that shard and its blob, nothing else"
        );
        drop(tree);
    }

    #[test]
    fn a_shard_that_points_outside_the_model_directory_is_left_alone() {
        let (tree, model, shards) = hf_cache(
            "escape",
            "rev1",
            &[("model-00001-of-00001.safetensors", 128)],
        );
        let outside = tree.join("outside/blob-elsewhere");
        write_file(&outside, 2048);
        let escaping = model.join("snapshots/rev1/model-00002-of-00002.safetensors");
        std::os::unix::fs::symlink(&outside, &escaping).expect("symlink the escaping shard");

        // Nothing is needed: the prune would remove everything it is allowed to remove.
        let outcome = prune(CachePruneMode::On, &model, &HashSet::new());
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("the prune must report what it did, got {outcome:?}");
        };

        assert!(
            exists(&escaping),
            "a shard whose target escapes the model directory is reported and left alone"
        );
        assert!(exists(&outside), "…and its target is never touched");
        assert!(
            !exists(&shards[0]),
            "a shard inside the model directory is still pruned"
        );
        assert_eq!(
            report.deleted.len(),
            2,
            "the symlink and the blob of the shard inside"
        );
        drop(tree);
    }

    /// Runs `flock -s -n <lock> -c 'echo locked'` in a **different process**: `Some(true)` when a
    /// neighbour can take the shared lock right now, `Some(false)` when it cannot, `None` when this
    /// machine has no `flock(1)` to play either part.
    fn neighbour_can_take_a_shared_lock(lock: &Path) -> Option<bool> {
        let output = std::process::Command::new("flock")
            .arg("-s")
            .arg("-n")
            .arg(lock)
            .arg("-c")
            .arg("echo locked")
            .output()
            .ok()?;
        Some(output.status.success())
    }

    /// Starts a neighbouring node: a child process holding the **shared** lock of the model, blocked
    /// until it is killed. Returns once the child has actually taken it.
    fn spawn_neighbour_holder(lock: &Path) -> Option<std::process::Child> {
        use std::io::BufRead;
        let mut child = std::process::Command::new("flock")
            .arg("-s")
            .arg(lock)
            .arg("-c")
            .arg("echo locked; sleep 30")
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        let mut line = String::new();
        std::io::BufReader::new(child.stdout.take()?)
            .read_line(&mut line)
            .ok()?;
        (line.trim() == "locked").then_some(child)
    }

    #[test]
    fn a_node_prunes_its_own_cache_even_though_it_holds_the_lock() {
        let (tree, model, shards) = hf_cache(
            "self-prune",
            "rev1",
            &[
                ("model-00001-of-00002.safetensors", 128),
                ("model-00002-of-00002.safetensors", 256),
            ],
        );
        // The node is serving this model, so it holds its own shared lock (exactly what
        // `acquire_weights` does through `hold_shards`).
        assert_eq!(hold_shards(&shards), vec![model.clone()]);
        let needed: HashSet<PathBuf> = [shards[0].clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::On, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("a node must be able to prune its own cache, got {outcome:?}");
        };
        assert!(
            !exists(&shards[1]),
            "its own lock is converted, not an obstacle: the reload must be able to free the shard"
        );
        assert_eq!(report.deleted.len(), 2, "the shard and its blob");

        // The lock is *still held* after the prune (a third party cannot take it exclusively) and it
        // is shared again, so a neighbour loading the same model is not blocked by our prune.
        let fresh = open_lock_file(&model).expect("open the lock file again");
        assert!(
            fresh.try_lock().is_err(),
            "the model must stay protected after its own prune"
        );
        drop(fresh);
        match neighbour_can_take_a_shared_lock(&lock_path(&model)) {
            Some(shared) => assert!(
                shared,
                "the prune must downgrade its own lock back to shared, or a neighbour could not \
                 load this model any more"
            ),
            None => eprintln!("skipping the downgrade check: flock(1) is not available"),
        }
        drop(tree);
    }

    #[test]
    fn another_live_process_blocks_the_prune() {
        let (tree, model, shards) = hf_cache(
            "neighbour",
            "rev1",
            &[("model-00001-of-00001.safetensors", 512)],
        );
        let Some(mut neighbour) = spawn_neighbour_holder(&lock_path(&model)) else {
            eprintln!("skipping: flock(1) is not available to play the neighbouring node");
            return;
        };

        // Nothing is needed, so this prune would delete everything it is allowed to delete…
        let outcome = prune(CachePruneMode::On, &model, &HashSet::new());

        let _ = neighbour.kill();
        let _ = neighbour.wait();

        assert_eq!(
            outcome,
            PruneOutcome::LockedByAnotherProcess {
                root: model.clone()
            },
            "a live neighbour holding the model must stop the prune, not race it"
        );
        assert!(
            exists(&shards[0]),
            "a live neighbour's shard is never deleted"
        );
        assert!(
            exists(&model.join("blobs/blob-model-00001-of-00001")),
            "…and neither is its blob"
        );
        drop(tree);
    }

    #[test]
    fn a_blob_shared_with_another_revision_survives() {
        let tree = TempTree::new("shared-blob");
        let model = tree.join("hub/models--acme--tiny");
        let blobs = model.join("blobs");
        let shared = blobs.join("blob-shared");
        let orphan = blobs.join("blob-orphan");
        write_file(&shared, 4096);
        write_file(&orphan, 2048);
        // Two revisions of the same model resolve to the **same** blob (HF deduplicates by content)…
        let mut revision = Vec::new();
        for rev in ["rev1", "rev2"] {
            let snapshot = model.join(format!("snapshots/{rev}"));
            fs::create_dir_all(&snapshot).expect("create the snapshot");
            write_file(&snapshot.join("config.json"), 8);
            let link = snapshot.join("model-00001-of-00001.safetensors");
            std::os::unix::fs::symlink(&shared, &link).expect("symlink the shared shard");
            revision.push(link);
        }
        // …and this node serves `rev2`, while a shard of a range it does not hold has a blob of its own.
        let stale = model.join("snapshots/rev2/model-00002-of-00002.safetensors");
        std::os::unix::fs::symlink(&orphan, &stale).expect("symlink the orphan shard");
        let needed: HashSet<PathBuf> = [revision[1].clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::On, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("the prune must report what it did, got {outcome:?}");
        };

        assert!(
            !exists(&stale),
            "the shard this range does not need goes: {:?}",
            report.deleted
        );
        assert!(
            !exists(&orphan),
            "its blob went with it: it was referenced by nothing else"
        );
        assert_eq!(
            report.deleted.len(),
            2,
            "the stale symlink and its orphan blob"
        );
        assert_eq!(report.freed_bytes, 2048, "the bytes of that shard only");
        assert!(
            exists(&shared),
            "the blob the served revision resolves to is kept, whatever else pointed at it"
        );
        assert!(
            exists(&revision[0]),
            "and the entry of the other revision stays: it names the very same file, so deleting it \
             would free nothing and would break a load pinned to that revision"
        );
        drop(tree);
    }

    #[test]
    fn a_local_model_directory_is_pruned_by_file_and_confined() {
        let tree = TempTree::new("local-prune");
        let model = tree.join("models/tiny");
        write_file(&model.join("config.json"), 16);
        let keep = model.join("model-00001-of-00002.safetensors");
        let old = model.join("model-00002-of-00002.safetensors");
        write_file(&keep, 128);
        write_file(&old, 1024);
        let sibling = tree.join("models/other/model.safetensors");
        write_file(&sibling, 4096);
        write_file(&tree.join("models/other/config.json"), 16);

        assert_eq!(
            model_root_of(std::slice::from_ref(&keep)),
            Some(model.clone()),
            "the local layout is pruned inside the model directory, not inside models_dir"
        );
        let needed: HashSet<PathBuf> = [keep.clone()].into_iter().collect();

        let outcome = prune(CachePruneMode::On, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("the prune must report what it did, got {outcome:?}");
        };

        assert!(!exists(&old), "the local shard of the old range goes");
        assert!(exists(&keep), "the local shard of the new range stays");
        assert!(
            exists(&model.join("config.json")),
            "config.json is not a shard"
        );
        assert!(
            exists(&sibling),
            "another model of models_dir is out of reach"
        );
        assert_eq!(report.freed_bytes, 1024);
        assert_eq!(
            report.deleted.len(),
            1,
            "a local layout has no blob to clean up"
        );
        drop(tree);
    }

    /// The measurement of the card (`TASK-73`): how many bytes a real reassignment frees, with **real**
    /// shard sizes and the real prune — on a **copy** of a real repository, never on the shared cache
    /// (which another node of this machine may be using).
    ///
    /// `#[ignore]`d because it needs a populated HuggingFace cache and copies a few hundred MB:
    ///
    /// ```text
    /// cargo test -p tessella-worker --lib measure_a_real -- --ignored --nocapture
    /// ```
    ///
    /// `TESSELLA_S80_MEASURE_REPO` picks the repository to copy; the default is the smallest one in the
    /// cache. `HF_HOME` (or `~/.cache/huggingface`) is where it looks for it. When the repository ships a
    /// single shard (a node downloads only the slice it serves, so one shard per repository is the normal
    /// state of a cache) the measurement builds the second one from **the real bytes** of the first and says
    /// so in its output: the sizes are never invented, the layout is what two reassignments leave behind.
    #[test]
    #[ignore = "needs a populated HF cache and copies a real repository (hundreds of MB)"]
    fn measure_a_real_reassignment_on_a_copy_of_a_real_repo() {
        let Some(source) = real_repo_to_measure() else {
            eprintln!(
                "S80 measurement · no HuggingFace repository found: export a cache or name one with \
                 TESSELLA_S80_MEASURE_REPO"
            );
            return;
        };
        let tree = TempTree::new("measure");
        let name = source
            .file_name()
            .expect("a repository name")
            .to_string_lossy()
            .into_owned();
        let model = tree.join(&format!("hub/{name}"));
        copy_tree(&source, &model);

        let mut shards = scan_shards(&model);
        let built = shards.len() < 2;
        if built {
            // The cache of this machine usually holds **one** shard of a repository (the node downloaded
            // only the slice it served) and the card's scenario needs two. The second shard is built from
            // the real one, blob and symlink exactly as HuggingFace lays them out: **the bytes are real**
            // (no number is invented), the layout is what two reassignments leave behind and the prune
            // that runs is the real one.
            let origin = shards.first().expect("a real shard to build the tree from");
            let real_blob = fs::canonicalize(&origin.path).expect("the real shard resolves");
            let second_blob = model.join("blobs/blob-second-shard");
            fs::copy(&real_blob, &second_blob).expect("copy the real shard");
            let second_link = origin
                .path
                .with_file_name("model-00002-of-00002.safetensors");
            std::os::unix::fs::symlink(&second_blob, &second_link).expect("link the second shard");
            shards = scan_shards(&model);
        }

        assert!(
            shards.len() >= 2,
            "the measurement needs two shards to have a range to give back"
        );
        // The reassignment of the card: this node served the whole model and now serves one slice, so
        // every shard but the kept one is what it has to give back.
        let kept = shards[0].path.clone();
        let kept_target = fs::canonicalize(&kept).expect("the kept shard resolves");
        let needed: HashSet<PathBuf> = [kept.clone()].into_iter().collect();
        let before = tree_bytes(&model);

        let outcome = prune(CachePruneMode::On, &model, &needed);
        let PruneOutcome::Inspected(report) = outcome else {
            panic!("the prune must report what it did, got {outcome:?}");
        };
        let after = tree_bytes(&model);

        eprintln!(
            "S80 measurement · {} · {} shards scanned, 1 served ({})\n  layout: {}\n  before: {before} \
             bytes ({:.1} MiB) → after: {after} bytes ({:.1} MiB)",
            source.display(),
            shards.len(),
            kept.display(),
            if built {
                "the cache ships one shard, so the second one is a copy of its real bytes"
            } else {
                "exactly as the cache ships it"
            },
            mib(before),
            mib(after)
        );
        eprintln!(
            "  the prune reports {} files and {} bytes ({:.1} MiB); the model directory lost {} bytes \
             ({:.1} MiB)",
            report.deleted.len(),
            report.freed_bytes,
            mib(report.freed_bytes),
            before - after,
            mib(before - after)
        );

        assert!(
            exists(&kept),
            "the shard of the range this node serves survives"
        );
        assert!(
            report.freed_bytes > 0,
            "the reassignment has to give something back"
        );
        assert_eq!(
            before - after,
            report.freed_bytes,
            "the bytes the prune reports must be the bytes the directory really lost"
        );
        for candidate in scan_shards(&model) {
            assert_eq!(
                candidate.target,
                kept_target,
                "every shard left resolves to the one this node serves: {}",
                candidate.path.display()
            );
        }
        drop(tree);
    }
}
