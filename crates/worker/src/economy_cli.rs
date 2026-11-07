use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand, Debug, Clone)]
pub enum EconomyCommands {
    /// Check your credit balance
    Balance,
    /// Verify the global transaction ledger
    Ledger {
        #[arg(short, long, default_value_t = 10)]
        limit: u32,
        #[command(subcommand)]
        sub: Option<LedgerSubcommands>,
    },
    /// Check the reputation score of a peer
    Reputation {
        #[arg(index = 1)]
        peer_id: String,
    },
    /// Manage weight manifests
    Weights {
        #[command(subcommand)]
        sub: WeightsSubcommands,
    },
    /// List transfer receipts for a peer
    Receipts {
        #[arg(long)]
        peer_id: String,
    },
    /// Check the status of the musivum federation
    Federation {
        #[command(subcommand)]
        sub: FederationSubcommands,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum FederationSubcommands {
    /// Show federation connectivity and sync status
    Status,
}

#[derive(Subcommand, Debug, Clone)]
pub enum WeightsSubcommands {
    /// List all canonical weight manifests
    List,
    /// Verify a local weight file against the canonical manifest
    Verify {
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        role: String, // JSON Role string
        #[arg(long)]
        file: String,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum LedgerSubcommands {
    /// Verify the ledger integrity using a signed snapshot
    Verify {
        #[arg(short, long)]
        tracker_url: Option<String>,
    },
}

/// The environment key the orchestrator's `admin_token` is read from.
///
/// It is the key the rest of the deployment uses, and it has opened
/// `/ledger/snapshot` for `economy ledger verify` since `TASK-34`/S40. Since `TASK-67`/S73
/// (**D-26**) the whole **economy plane** is the operator's — `GET /ledger`, `/credits` and
/// `/reputation` —, so `balance`, `ledger` and `reputation` need it too.
const ADMIN_TOKEN_ENV: &str = "MUSIVUM_ADMIN_TOKEN";

/// [`ADMIN_TOKEN_ENV`], trimmed, when the operator exported a non-empty one.
fn admin_token() -> Option<String> {
    std::env::var(ADMIN_TOKEN_ENV)
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

/// Adds `Authorization: Bearer <MUSIVUM_ADMIN_TOKEN>` when [`admin_token`] is set.
///
/// Without it the request goes out unchanged on purpose: the orchestrator answers `401` (it has a
/// token and none was sent) or `503` (it has none configured), and [`economy_refusal_hint`] turns
/// either status into the sentence that says what to configure.
fn with_admin_token(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match admin_token() {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// The sentence that explains an economy refusal, or `""` for a status the operator cannot fix by
/// configuring a token (`TASK-67`/S73).
fn economy_refusal_hint(status: reqwest::StatusCode) -> &'static str {
    match status.as_u16() {
        401 | 403 => {
            " — set the orchestrator's admin_token in MUSIVUM_ADMIN_TOKEN (the economy surface is \
             the operator's since S73)"
        }
        503 => " — the orchestrator has no usable admin_token configured (`openssl rand -hex 32`)",
        _ => "",
    }
}

pub async fn handle_economy_commands(
    cmd: EconomyCommands,
    tracker_url: String,
    peer_id: String,
) -> Result<()> {
    match cmd {
        EconomyCommands::Balance => {
            use colored::Colorize;
            use comfy_table::Table;

            let client = reqwest::Client::new();
            let resp = with_admin_token(
                client.get(format!("{}/credits?peer_id={}", tracker_url, peer_id)),
            )
            .send()
            .await?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                println!(
                    "{}",
                    format!(
                        "❌ Failed to read the balance of {}: {}{} ({})",
                        peer_id,
                        status,
                        economy_refusal_hint(status),
                        body.chars().take(200).collect::<String>()
                    )
                    .red()
                );
                return Ok(());
            }
            let json: serde_json::Value = resp.json().await?;

            let mut table = Table::new();
            table.set_header(vec!["Account Details", "Value"]);
            table.add_row(vec!["Peer ID".bold().to_string(), peer_id.to_string()]);

            let balance = json["balance"].as_f64().unwrap_or(0.0);
            let balance_colored = if balance > 0.0 {
                format!("{:.2} credits", balance).green().bold().to_string()
            } else {
                format!("{:.2} credits", balance).red().bold().to_string()
            };
            table.add_row(vec!["Balance".bold().to_string(), balance_colored]);

            println!("\n{}", "💳 Credit Balance".cyan().bold());
            println!("{}", table);
        }
        EconomyCommands::Ledger { limit, sub } => {
            if let Some(LedgerSubcommands::Verify {
                tracker_url: sub_url,
            }) = sub
            {
                let final_url = sub_url.unwrap_or(tracker_url);
                verify_ledger_snapshot(&final_url).await?;
                return Ok(());
            }

            let client = reqwest::Client::new();
            let resp =
                with_admin_token(client.get(format!("{}/ledger?limit={}", tracker_url, limit)))
                    .send()
                    .await?;
            let status = resp.status();
            if !status.is_success() {
                use colored::Colorize;
                let body = resp.text().await.unwrap_or_default();
                println!(
                    "{}",
                    format!(
                        "❌ Failed to read the ledger: {}{} ({})",
                        status,
                        economy_refusal_hint(status),
                        body.chars().take(200).collect::<String>()
                    )
                    .red()
                );
                return Ok(());
            }
            // `TASK-42`/S48: el endpoint sirve **asientos** y lo más reciente primero. Antes el cuerpo
            // era un array de *strings* (cada asiento serializado dentro del JSON) y este `resp.json()`
            // fallaba siempre, así que este comando nunca llegó a imprimir una tabla.
            let entries: Vec<tessella_core::LedgerEntry> = resp.json().await?;

            println!("\n📖 Global Ledger (most recent first):");
            println!("{:-<100}", "");
            println!(
                "{:<20} | {:<10} | {:<40} | {:<10}",
                "Timestamp", "Type", "Peer ID", "Amount"
            );
            println!("{:-<100}", "");
            for entry in entries.iter() {
                let type_str = match entry.entry_type {
                    tessella_core::LedgerEntryType::Earned => "EARNED",
                    tessella_core::LedgerEntryType::Spent => "SPENT",
                    tessella_core::LedgerEntryType::Slashed => "SLASHED",
                };
                println!(
                    "{:<20} | {:<10} | {:<40} | {:<10.2}",
                    entry.timestamp, type_str, entry.peer_id, entry.amount
                );
            }
            println!("{:-<100}\n", "");
        }
        EconomyCommands::Reputation {
            peer_id: target_peer_id,
        } => {
            let client = reqwest::Client::new();
            let resp = with_admin_token(client.get(format!(
                "{}/reputation?peer_id={}",
                tracker_url, target_peer_id
            )))
            .send()
            .await?;

            if !resp.status().is_success() {
                use colored::Colorize;
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                println!(
                    "{}",
                    format!(
                        "❌ Failed to fetch reputation for {}: {}{} ({})",
                        target_peer_id,
                        status,
                        economy_refusal_hint(status),
                        body.chars().take(200).collect::<String>()
                    )
                    .red()
                );
                return Ok(());
            }

            let json: serde_json::Value = resp.json().await?;
            let score = json["score"].as_f64().unwrap_or(0.0);
            let status = json["status"].as_str().unwrap_or("Unknown");
            let last_updated = json["last_updated"].as_str().unwrap_or("Never");

            use colored::Colorize;
            use comfy_table::Table;

            let mut table = Table::new();
            table.set_header(vec!["Reputation Metric", "Value"]);
            table.add_row(vec!["Peer ID".bold().to_string(), target_peer_id.clone()]);
            table.add_row(vec![
                "Reputation Score".bold().to_string(),
                format!("{:.4}", score).yellow().bold().to_string(),
            ]);

            let status_colored = match status {
                "Healthy" => status.green().bold().to_string(),
                "Degraded" => status.yellow().bold().to_string(),
                _ => status.red().bold().to_string(),
            };
            table.add_row(vec!["Status".bold().to_string(), status_colored]);
            table.add_row(vec![
                "Last Updated".bold().to_string(),
                last_updated.to_string(),
            ]);

            println!("\n{}", "⭐ Peer Reputation Details".cyan().bold());
            println!("{}", table);
        }
        EconomyCommands::Weights { sub } => match sub {
            WeightsSubcommands::List => {
                let client = reqwest::Client::new();
                let resp = client
                    .get(format!("{}/weights/manifests", tracker_url))
                    .send()
                    .await?;
                let manifests: Vec<tessella_core::models::WeightManifest> = resp.json().await?;

                println!("\n📦 Canonical Weight Manifests:");
                println!("{:-<100}", "");
                println!(
                    "{:<15} | {:<20} | {:<10} | {:<10} | {:<15}",
                    "Model ID", "Role", "Hash", "Size(MB)", "Registered At"
                );
                println!("{:-<100}", "");
                for m in manifests {
                    let role_str = serde_json::to_string(&m.role).unwrap_or_default();
                    let size_mb = m.size_bytes as f64 / 1024.0 / 1024.0;
                    let hash_short = hex::encode(&m.blake3_hash[..4]);
                    println!(
                        "{:<15} | {:<20} | {:<10} | {:<10.2} | {:<15}",
                        m.model_id, role_str, hash_short, size_mb, m.registered_at
                    );
                }
                println!("{:-<100}\n", "");
            }
            WeightsSubcommands::Verify {
                model_id,
                role,
                file,
            } => {
                println!(
                    "🔍 Verifying {} integrity against canonical manifest...",
                    file
                );
                let client = reqwest::Client::new();
                let resp = client
                    .get(format!(
                        "{}/weights/manifest?model_id={}&role={}",
                        tracker_url, model_id, role
                    ))
                    .send()
                    .await?;

                if resp.status() == 404 {
                    println!(
                        "❌ FAILED: No canonical manifest found for {}/{}",
                        model_id, role
                    );
                    return Ok(());
                }

                let manifest: tessella_core::models::WeightManifest = resp.json().await?;

                if let Err(e) = manifest.verify() {
                    println!("❌ FAILED: Musivum signature on manifest is INVALID: {}", e);
                    return Ok(());
                }

                let data = std::fs::read(&file)?;
                let hash = blake3::hash(&data);
                let actual_hash: [u8; 32] = hash.into();

                if actual_hash == manifest.blake3_hash {
                    println!("✅ MATCH: Integrity verified for {}.", file);
                    println!("BLAKE3: {:x?}", actual_hash);
                } else {
                    println!("🚨 MISMATCH: File hash does not match canonical manifest!");
                    println!("Expected: {:x?}", manifest.blake3_hash);
                    println!("Actual:   {:x?}", actual_hash);
                }
            }
        },
        EconomyCommands::Receipts {
            peer_id: target_peer_id,
        } => {
            let client = reqwest::Client::new();
            let resp = client
                .get(format!(
                    "{}/receipts?peer_id={}",
                    tracker_url, target_peer_id
                ))
                .send()
                .await?;
            let receipts: Vec<tessella_core::models::TransferReceipt> = resp.json().await?;

            println!("\n📜 Transfer Receipts for {}:", target_peer_id);
            println!("{:-<120}", "");
            println!(
                "{:<20} | {:<20} | {:<10} | {:<10} | {:<10} | {:<15}",
                "Sender", "Receiver", "Model", "Role", "Hash", "Timestamp"
            );
            println!("{:-<120}", "");
            for r in receipts {
                let hash_short = hex::encode(&r.blake3_hash[..4]);
                println!(
                    "{:<20} | {:<20} | {:<10} | {:<10} | {:<10} | {:<15}",
                    r.sender_peer_id,
                    r.receiver_peer_id,
                    r.model_id,
                    format!("{:?}", r.role),
                    hash_short,
                    r.timestamp
                );
            }
            println!("{:-<120}\n", "");
        }
        EconomyCommands::Federation { sub } => match sub {
            FederationSubcommands::Status => {
                let client = reqwest::Client::new();
                let resp = client
                    .get(format!("{}/federation/status", tracker_url))
                    .send()
                    .await?;
                let json: serde_json::Value = resp.json().await?;

                use colored::Colorize;
                use comfy_table::Table;

                let role = json["role"].as_str().unwrap_or("Unknown");
                let json_peer_id = json["peer_id"].as_str().unwrap_or("Unknown");
                let ledger_height = json["ledger_height"].as_u64().unwrap_or(0);

                let mut table = Table::new();
                table.set_header(vec!["Federation Metric", "Value"]);
                table.add_row(vec![
                    "Federation Role".bold().to_string(),
                    role.magenta().bold().to_string(),
                ]);
                table.add_row(vec![
                    "Musivum Peer ID".bold().to_string(),
                    json_peer_id.to_string(),
                ]);
                table.add_row(vec![
                    "Ledger Height".bold().to_string(),
                    ledger_height.to_string().green().to_string(),
                ]);

                println!("\n{}", "🛰️ Musivum Federation Status".cyan().bold());
                println!("{}", table);

                if role == "primary" {
                    let empty_vec = vec![];
                    let replicas = json["connected_replicas"].as_array().unwrap_or(&empty_vec);
                    if !replicas.is_empty() {
                        let mut rep_table = Table::new();
                        rep_table.set_header(vec!["Replica URL", "Connected Since", "Sync Mode"]);
                        for r in replicas {
                            rep_table.add_row(vec![
                                r["url"].as_str().unwrap_or("Unknown").to_string(),
                                r["connected_since"]
                                    .as_str()
                                    .unwrap_or("Unknown")
                                    .to_string(),
                                r["mode"]
                                    .as_str()
                                    .unwrap_or("Unknown")
                                    .yellow()
                                    .bold()
                                    .to_string(),
                            ]);
                        }
                        println!("\n🛰️  {}", "Connected Replicas".bold());
                        println!("{}", rep_table);
                    } else {
                        println!("\n🛰️  {}", "No active replicas connected".yellow());
                    }
                }
            }
        },
    }
    Ok(())
}

async fn verify_ledger_snapshot(tracker_url: &str) -> Result<()> {
    println!("🔍 Fetching ledger snapshot from {}...", tracker_url);
    let client = reqwest::Client::new();
    // `TASK-34`/S40: `/ledger/snapshot` is an admin endpoint — the ledger holds balances and work
    // proofs —, so copying it needs the token the operator configured on the primary. It is the same
    // `MUSIVUM_ADMIN_TOKEN` the rest of the economy plane uses since `TASK-67`/S73: without it the
    // primary answers `401`/`503` and the message below says exactly that instead of an opaque status.
    let request = with_admin_token(client.get(format!("{}/ledger/snapshot", tracker_url)));
    if admin_token().is_none() {
        println!(
            "ℹ️ {ADMIN_TOKEN_ENV} is not set: if the primary protects its ledger, this call answers \
             401/503"
        );
    }

    let resp = request.send().await?;
    let status = resp.status();

    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        // `404` is the only refusal that is about the age of the primary, not about the token.
        let hint = if status.as_u16() == 404 {
            " — this primary predates S40: it has no /ledger/snapshot route"
        } else {
            economy_refusal_hint(status)
        };
        return Err(anyhow::anyhow!(
            "Failed to fetch snapshot: {} ({}){}",
            status,
            body.chars().take(200).collect::<String>(),
            hint
        ));
    }

    let snapshot: tessella_core::models::LedgerSnapshot = resp.json().await?;

    match snapshot.verify() {
        Ok(_) => {
            let slashed_count = snapshot
                .entries
                .iter()
                .filter(|e| matches!(e.entry.entry_type, tessella_core::LedgerEntryType::Slashed))
                .count();

            println!("\n✅ Ledger Verification SUCCESSFUL");
            println!("{:-<40}", "");
            println!("Musivum PeerID:  {}", snapshot.musivum_peer_id);
            println!("Total Entries:   {}", snapshot.entries.len());
            println!("Ledger Height:   {}", snapshot.snapshot_height);
            println!("Slashed Nodes:   {}", slashed_count);
            println!("Snapshot Hash:   {:x?}", &snapshot.snapshot_hash[..8]);
            println!("Timestamp:       {}", snapshot.timestamp);
            println!("{:-<40}\n", "");
        }
        Err(e) => {
            println!("❌ FAILED: Ledger Verification failed: {}", e);
        }
    }

    Ok(())
}
