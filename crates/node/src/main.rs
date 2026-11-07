use clap::{Parser, Subcommand};
use dialoguer::{Input, Select};
use tessella_worker::config::{CachePruneMode, WorkerConfig};
use tessella_worker::service::WorkerService;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(author, version, about = "Tessella P2P Worker Daemon", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "worker_config.toml")]
    config: String,

    #[arg(long, default_value_t = false)]
    clean_cache: bool,

    #[arg(long)]
    port: Option<u16>,

    #[arg(long, alias = "musivum-url")]
    tracker_url: Option<String>,

    #[arg(long, alias = "vram")]
    vram_gb: Option<f32>,

    #[arg(long)]
    model_id: Option<String>,

    /// Prune the cache shards this node's new assignment does not need (`S80`/`TASK-73`):
    /// `off` (default) · `dry-run` · `on`. Overrides `model_cache_prune` of the config file.
    #[arg(long, value_name = "MODE")]
    model_cache_prune: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize configuration and keys interactively
    Init,
    /// Clean configuration and cryptographic keys
    #[command(alias = "reset")]
    CleanConfig,
    /// Manage worker cryptographic keys
    Keys {
        #[command(subcommand)]
        sub: KeysSubcommands,
    },
    /// Economy and Federation commands
    #[command(flatten)]
    Economy(tessella_worker::economy_cli::EconomyCommands),
}

#[derive(Subcommand, Debug)]
enum KeysSubcommands {
    /// Read the node key and print the public PeerId and Ed25519 Public Key
    Show,
    /// Force the creation of a new keypair (prompts before overwriting)
    Generate,
}

#[derive(Debug, serde::Deserialize)]
struct ModelDiscoveryInfo {
    model_id: String,
    active_workers: usize,
}

async fn run_first_run_wizard(path: &std::path::Path) -> anyhow::Result<WorkerConfig> {
    println!("Welcome to Tessella! Let's set up your node.");

    let orchestrator_url: String = Input::new()
        .with_prompt("Orchestrator URL")
        .default("https://tracker1.musivum.dev".into())
        .interact_text()?;

    println!("🔍 Querying orchestrator for active models...");
    let client = reqwest::Client::new();
    let mut discovered_models = Vec::new();
    let fetch_url = format!(
        "{}/api/network/models",
        orchestrator_url.trim_end_matches('/')
    );
    if let Ok(res) = client.get(&fetch_url).send().await {
        if res.status().is_success() {
            if let Ok(models) = res.json::<Vec<ModelDiscoveryInfo>>().await {
                discovered_models = models;
            }
        }
    }

    let default_model: String = if !discovered_models.is_empty() {
        let mut options: Vec<String> = discovered_models
            .iter()
            .map(|m| format!("{} ({} active workers)", m.model_id, m.active_workers))
            .collect();
        options.push("Manual Entry (Custom HuggingFace ID)".to_string());

        let selection = Select::new()
            .with_prompt("Select Model ID")
            .items(&options)
            .default(0)
            .interact()?;

        if selection < discovered_models.len() {
            discovered_models[selection].model_id.clone()
        } else {
            Input::new()
                .with_prompt("Default Model ID")
                .default("Qwen/Qwen2.5-0.5B-Instruct".into())
                .interact_text()?
        }
    } else {
        println!("⚠️ Could not discover active models. Falling back to manual entry.");
        Input::new()
            .with_prompt("Default Model ID")
            .default("Qwen/Qwen2.5-0.5B-Instruct".into())
            .interact_text()?
    };

    let vram_mb: u32 = Input::new()
        .with_prompt("How much VRAM (in MB) do you want to dedicate?")
        .default(2048)
        .interact_text()?;

    let models_dir: String = Input::new()
        .with_prompt("Directory to store model weights")
        .default("./models".into())
        .interact_text()?;

    let home_dir =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    let default_key = home_dir
        .join(".config/tessella/node.key")
        .to_string_lossy()
        .into_owned();

    let keypair_path: String = Input::new()
        .with_prompt("Path to cryptographic keypair file")
        .default(default_key)
        .interact_text()?;

    let api_port: u16 = Input::new()
        .with_prompt("Local gateway API port")
        .default(8080)
        .interact_text()?;

    let p2p_port: u16 = Input::new()
        .with_prompt("P2P network port")
        .default(9080)
        .interact_text()?;

    // Generate the TOML content in unified format
    let config_content = format!(
        r#"# Global identity and orchestrator
orchestrator_url = "{}"
keypair_path = "{}"
default_model = "{}"

[worker]
vram_mb = {}
models_dir = "{}"
api_port = {}
p2p_port = {}
"#,
        orchestrator_url, keypair_path, default_model, vram_mb, models_dir, api_port, p2p_port
    );

    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid config path"))?;
    if !parent.exists() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, config_content)?;
    println!(
        "✅ Configuration saved to {}! Starting node...",
        path.display()
    );

    // Generate keypair if not exists
    let k_path = std::path::Path::new(&keypair_path);
    if !k_path.exists() {
        let k_parent = k_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid key path"))?;
        if !k_parent.exists() {
            std::fs::create_dir_all(k_parent)?;
        }
        let kp = tessella_network::generate_keypair();
        let bytes = tessella_network::keypair_to_bytes(&kp)?;
        std::fs::write(&keypair_path, bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&keypair_path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&keypair_path, perms)?;
        }
        let peer_id = tessella_network::peer_id_from_keypair(&kp);
        println!("🔑 Generated cryptographic keypair at: {}", keypair_path);
        println!("Peer ID: {}", peer_id);
    }

    let config = WorkerConfig::load(path)?;
    Ok(config)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tessella_core::init_tracing()?;
    let args = Args::parse();

    let global_config = dirs::config_dir()
        .map(|d| d.join("tessella/config.toml"))
        .unwrap_or_default();

    let config_path = if args.config != "worker_config.toml" {
        std::path::PathBuf::from(&args.config)
    } else {
        let local = std::path::PathBuf::from("worker_config.toml");
        if local.exists() {
            local
        } else {
            global_config
        }
    };

    // If subcommand is CleanConfig, we perform it before loading config to prevent errors
    if let Some(Commands::CleanConfig) = &args.command {
        // Find if config exists to determine the port and possible key path
        let mut port = 8080;
        let mut custom_key_path: Option<String> = None;
        if config_path.exists() {
            if let Ok(cfg) = WorkerConfig::load(&config_path) {
                port = cfg.listen_port;
                custom_key_path = cfg.keypair_path.clone();
            }
            std::fs::remove_file(&config_path)?;
            println!("🗑️ Removed configuration file: {}", config_path.display());
        } else {
            println!(
                "ℹ️ No configuration file found at: {}",
                config_path.display()
            );
        }

        // Check if there is a local worker_config.toml
        let local_config = std::path::PathBuf::from("worker_config.toml");
        if local_config.exists() && local_config != config_path {
            if custom_key_path.is_none() {
                if let Ok(cfg) = WorkerConfig::load(&local_config) {
                    custom_key_path = cfg.keypair_path.clone();
                }
            }
            std::fs::remove_file(&local_config)?;
            println!(
                "🗑️ Removed local configuration file: {}",
                local_config.display()
            );
        }

        // Delete node key for the resolved path
        let key_path = custom_key_path.unwrap_or_else(|| format!("node_{}.key", port));
        if std::path::Path::new(&key_path).exists() {
            std::fs::remove_file(&key_path)?;
            println!("🗑️ Removed key file: {}", key_path);
        } else {
            println!("ℹ️ No key file found at: {}", key_path);
        }

        // Also clean up any other node_*.key in current directory
        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    if name_str.starts_with("node_")
                        && name_str.ends_with(".key")
                        && entry.path() != std::path::Path::new(&key_path)
                    {
                        if let Err(e) = std::fs::remove_file(entry.path()) {
                            tracing::warn!("⚠️ Failed to delete key file {}: {}", name_str, e);
                        } else {
                            println!("🗑️ Removed key file: {}", name_str);
                        }
                    }
                }
            }
        }

        // Check for the default client key if it's there
        if let Some(config_dir) = dirs::config_dir() {
            let client_key = config_dir.join("tessella/client.key");
            if client_key.exists() {
                std::fs::remove_file(&client_key)?;
                println!("🗑️ Removed client key file: {}", client_key.display());
            }
        }

        println!("✨ Configuration cleaned successfully!");
        return Ok(());
    }

    // Headless/container detection: when configuration is supplied through the
    // environment (e.g. Docker/RunPod), skip the interactive wizard which needs a
    // TTY and would otherwise abort the process.
    let non_interactive = std::env::var("TESSELLA_NON_INTERACTIVE").is_ok()
        || std::env::var("TRACKER_URL").is_ok()
        || std::env::var("ORCHESTRATOR_URL").is_ok();

    let mut config = if !config_path.exists() {
        if non_interactive {
            info!(
                "No config file found; starting in non-interactive mode from environment/defaults"
            );
            WorkerConfig::load(&config_path)?
        } else {
            run_first_run_wizard(&config_path).await?
        }
    } else {
        WorkerConfig::load(&config_path)?
    };

    // CLI Overrides (highest priority)
    if let Some(p) = args.port {
        config.listen_port = p;
        config.p2p_port = p;
    }
    if let Some(u) = args.tracker_url {
        config.orchestrator_url = u;
    }
    if let Some(v) = args.vram_gb {
        config.vram_budget_gb = v;
    }
    if let Some(m) = args.model_id {
        config.default_model = m;
    }
    if let Some(raw) = &args.model_cache_prune {
        // The same parser the `TESSELLA_MODEL_CACHE_PRUNE` variable and the TOML key use: a typo
        // aborts the boot naming the accepted values instead of leaving the cache in a mode nobody
        // chose (`S80`/`TASK-73`).
        config.model_cache_prune = CachePruneMode::parse(raw)?;
    }

    // Handle subcommands
    if let Some(cmd) = &args.command {
        match cmd {
            Commands::Init => {
                let config_dir = config_path
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("Invalid config path"))?;
                if !config_dir.exists() {
                    std::fs::create_dir_all(config_dir)?;
                }

                // If config doesn't exist, write defaults
                if !config_path.exists() {
                    let keypair_path = config_dir.join("node.key");
                    let config_content = format!(
                        r#"# Global identity and orchestrator
orchestrator_url = "https://tracker1.musivum.dev"
keypair_path = "{}"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 2048
models_dir = "./models"
api_port = 8080 # The local gateway port for the CLI
p2p_port = 9080
"#,
                        keypair_path.to_string_lossy()
                    );
                    std::fs::write(&config_path, config_content)?;
                    println!("📝 Config file generated at: {}", config_path.display());
                    // Reload config to get the correct keypair_path
                    config = WorkerConfig::load(&config_path)?;
                }

                let key_path = config.get_keypair_path();
                let path = std::path::Path::new(&key_path);
                if !path.exists() {
                    let k_parent = path
                        .parent()
                        .ok_or_else(|| anyhow::anyhow!("Invalid key path"))?;
                    if !k_parent.exists() {
                        std::fs::create_dir_all(k_parent)?;
                    }
                    let kp = tessella_network::generate_keypair();
                    let bytes = tessella_network::keypair_to_bytes(&kp)?;
                    std::fs::write(&key_path, bytes)?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mut perms = std::fs::metadata(&key_path)?.permissions();
                        perms.set_mode(0o600);
                        std::fs::set_permissions(&key_path, perms)?;
                    }
                    let peer_id = tessella_network::peer_id_from_keypair(&kp);
                    println!("🔑 Generated cryptographic keypair at: {}", key_path);
                    println!("Peer ID: {}", peer_id);
                } else {
                    let bytes = std::fs::read(&key_path)?;
                    let kp = tessella_network::keypair_from_bytes(&bytes)?;
                    let peer_id = tessella_network::peer_id_from_keypair(&kp);
                    println!("🔑 Found existing cryptographic keypair at: {}", key_path);
                    println!("Peer ID: {}", peer_id);
                }
                println!("🎉 Tessella Worker Initialized Successfully!");
                return Ok(());
            }
            Commands::CleanConfig => {
                // Handled above
                return Ok(());
            }
            Commands::Keys { sub } => {
                match sub {
                    KeysSubcommands::Show => {
                        let key_path = config.get_keypair_path();
                        if !std::path::Path::new(&key_path).exists() {
                            println!("❌ No keypair found at {}. Run `tessella-worker init` or start the worker to generate one.", key_path);
                        } else {
                            let bytes = std::fs::read(&key_path)?;
                            let kp = tessella_network::keypair_from_bytes(&bytes)?;
                            let peer_id = tessella_network::peer_id_from_keypair(&kp);
                            let pub_key_bytes = kp.public().encode_protobuf();
                            println!("\n🔑 Tessella Worker Keys:");
                            println!("Path:       {}", key_path);
                            println!("Peer ID:    {}", peer_id);
                            println!("Public Key: {}", hex::encode(pub_key_bytes));
                        }
                    }
                    KeysSubcommands::Generate => {
                        use dialoguer::Confirm;
                        let key_path = config.get_keypair_path();
                        let path = std::path::Path::new(&key_path);
                        if path.exists() {
                            let confirm = Confirm::new()
                                .with_prompt(format!("🚨 A key already exists at {}. Are you sure you want to overwrite it?", key_path))
                                .default(false)
                                .interact()?;
                            if !confirm {
                                println!("Operation cancelled.");
                                return Ok(());
                            }
                        }

                        if path.exists() {
                            std::fs::remove_file(path)?;
                        }
                        let kp = tessella_network::generate_keypair();
                        let bytes = tessella_network::keypair_to_bytes(&kp)?;
                        std::fs::write(&key_path, bytes)?;
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::PermissionsExt;
                            let mut perms = std::fs::metadata(&key_path)?.permissions();
                            perms.set_mode(0o600);
                            std::fs::set_permissions(&key_path, perms)?;
                        }
                        let peer_id = tessella_network::peer_id_from_keypair(&kp);
                        println!("🔑 Successfully generated and wrote a new keypair!");
                        println!("New Peer ID: {}", peer_id);
                    }
                }
                return Ok(());
            }
            Commands::Economy(economy_cmd) => {
                let key_path = config.get_keypair_path();
                let peer_id = if std::path::Path::new(&key_path).exists() {
                    let bytes = std::fs::read(&key_path)?;
                    let kp = tessella_network::keypair_from_bytes(&bytes)?;
                    tessella_network::peer_id_from_keypair(&kp).to_string()
                } else {
                    "Unknown".to_string()
                };
                tessella_worker::economy_cli::handle_economy_commands(
                    economy_cmd.clone(),
                    config.orchestrator_url.clone(),
                    peer_id,
                )
                .await?;
                return Ok(());
            }
        }
    }

    info!("📖 Loaded configuration from {}", config_path.display());

    if args.clean_cache {
        let model = &config.default_model;
        let folder_name = format!("models--{}", model.replace("/", "--"));
        if let Some(home) = dirs::home_dir() {
            let cache_path = home.join(".cache/huggingface/hub").join(&folder_name);
            if cache_path.exists() {
                match std::fs::remove_dir_all(&cache_path) {
                    Ok(_) => info!("🗑️ Cleared cache for {}", model),
                    Err(e) => tracing::warn!("⚠️ Failed to clear cache: {}", e),
                }
            } else {
                tracing::warn!(
                    "⚠️ Cache directory for {} does not exist at {:?}",
                    model,
                    cache_path
                );
            }
        }
    }

    // `SOLID-89`: the composition root is the only place that knows *which* HTTP surface the daemon
    // gets. `tessella-worker` receives it as a port (`tessella_ingress::ApiHost`) and serves it from
    // the same `select!` that used to call `tessella_api::start_api_server` directly, so the data
    // plane no longer links Axum nor the HTTP crate.
    let api_host = std::sync::Arc::new(tessella_api::axum_host::AxumApiHost::new());
    let service = WorkerService::new(config, api_host)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize worker service: {}", e))?;

    if let Err(e) = service.run().await {
        error!("❌ Worker execution failed: {:?}", e);
        std::process::exit(1);
    }

    Ok(())
}
