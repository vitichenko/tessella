#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::use_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::missing_panics_doc)]

mod config;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::ClientConfig;
use futures_util::StreamExt;
use std::io::Write;
use std::path::PathBuf;
use tessella_core::models::ChatMessage;
use tessella_telemetry::init_logging;

#[derive(Parser, Debug)]
#[command(author, version, about = "Tessella CLI Chat Client")]
struct Cli {
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[arg(short = 'a', long, alias = "api-url")]
    api_url: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize configuration and cryptographic keys
    Init {
        /// Force overwrite of existing config and keys
        #[arg(long)]
        force: bool,
    },
    /// Start a chat generation
    Chat {
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(long)]
        session_file: Option<String>,
        #[arg(short, long)]
        interactive: bool,
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "WORDS"
        )]
        words: Vec<String>,
        #[arg(long, default_value = "chat")]
        mode: String,
        #[arg(long)]
        prefix: Option<String>,
        #[arg(long)]
        suffix: Option<String>,
        #[arg(short, long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, default_value_t = false)]
        timing: bool,
    },
    /// Fill-in-the-middle code completion
    Fim {
        #[arg(short, long)]
        prefix: String,
        #[arg(short, long)]
        suffix: String,
        #[arg(short, long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long)]
        session_file: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let config_path = cli
        .config
        .clone()
        .unwrap_or_else(crate::config::get_default_config_path);

    let config = if config_path.exists() {
        Some(ClientConfig::load(&config_path)?)
    } else {
        None
    };

    let api_url = cli
        .api_url
        .clone()
        .or_else(|| {
            config
                .as_ref()
                .map(|c| format!("http://127.0.0.1:{}", c.worker.api_port))
        })
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());

    let model_id = cli
        .model_id
        .clone()
        .or_else(|| config.as_ref().map(|c| c.default_model.clone()))
        .unwrap_or_else(|| "Qwen/Qwen2.5-0.5B-Instruct".to_string());

    let log_level = config
        .as_ref()
        .map(|c| c.log_level.clone())
        .unwrap_or_else(|| "info".to_string());

    match cli.command {
        Some(Commands::Init { force }) => {
            let config_dir = crate::config::get_tessella_dir();
            let config_path = crate::config::get_default_config_path();
            let keypair_path = config_dir.join("node.key");

            if !config_dir.exists() {
                std::fs::create_dir_all(&config_dir)?;
            }

            // Generate/Write keypair
            let key_existed = keypair_path.exists();
            if !key_existed || force {
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
                println!(
                    "🔑 Generated cryptographic keypair at: {}",
                    keypair_path.display()
                );
                println!("Peer ID: {}", peer_id);
            } else {
                let bytes = std::fs::read(&keypair_path)?;
                let kp = tessella_network::keypair_from_bytes(&bytes)?;
                let peer_id = tessella_network::peer_id_from_keypair(&kp);
                println!(
                    "🔑 Found existing cryptographic keypair at: {}",
                    keypair_path.display()
                );
                println!("Peer ID: {}", peer_id);
            }

            // Write unified config
            if !config_path.exists() || force {
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
            } else {
                println!(
                    "📝 Config file already exists at: {}",
                    config_path.display()
                );
            }

            println!("🎉 Tessella CLI Client and Node Initialized Successfully!");
        }
        Some(Commands::Chat {
            prompt,
            words,
            mode,
            prefix,
            suffix,
            max_tokens,
            temperature,
            timing,
            session_file,
            interactive,
        }) => {
            let _guard = init_logging("client", true, &log_level)?;

            let inference_mode = if mode == "fim" {
                let p =
                    prefix.ok_or_else(|| anyhow::anyhow!("--prefix is required for --mode fim"))?;
                tessella_inference::InferenceMode::Fim {
                    prefix: p,
                    suffix: suffix.unwrap_or_default(),
                }
            } else {
                tessella_inference::InferenceMode::Chat
            };

            let final_prompt = if let tessella_inference::InferenceMode::Chat = inference_mode {
                prompt.or_else(|| {
                    if words.is_empty() {
                        None
                    } else {
                        Some(words.join(" "))
                    }
                })
            } else {
                None
            };

            run_chat_loop(
                final_prompt,
                inference_mode,
                max_tokens,
                temperature,
                api_url,
                model_id,
                timing,
                session_file,
                interactive,
            )
            .await?;
        }
        Some(Commands::Fim {
            prefix,
            suffix,
            max_tokens,
            temperature,
            session_file,
        }) => {
            let _guard = init_logging("client", true, &log_level)?;
            run_fim_stream(
                prefix,
                suffix,
                max_tokens,
                temperature,
                api_url,
                model_id,
                session_file,
            )
            .await?;
        }
        None => {
            println!("Tessella CLI Client. Use --help for commands.");
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_chat_loop(
    prompt: Option<String>,
    mode: tessella_inference::InferenceMode,
    max_tokens: usize,
    temperature: f64,
    api_url: String,
    model_id: String,
    _timing: bool,
    session_file: Option<String>,
    _interactive: bool,
) -> Result<()> {
    let mut history: Vec<ChatMessage> = if let Some(ref path) = session_file {
        if std::path::Path::new(path).exists() {
            let content = std::fs::read_to_string(path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if let Some(p) = prompt {
        if !p.is_empty() {
            history.push(ChatMessage {
                role: "user".to_string(),
                content: p,
            });
        }
    }

    let client = reqwest::Client::new();
    // `TASK-78`/S85 (`D-36`): the id of a conversation is issued by the node that holds its K/V cache,
    // and what a client sends is only a **hint** —honoured only when it names a session that node still
    // has—. This loop therefore starts with no hint at all: the `cli-<uuid>` name it used to mint was
    // never issued by any node, so no node could resume it, and every turn of a CLI session prefilled
    // from scratch. What it keeps for the next turn is the id the answer publishes.
    let mut session_hint: Option<String> = None;

    loop {
        let needs_input =
            history.is_empty() || history.last().map(|m| m.role.as_str()) == Some("assistant");

        if needs_input {
            print!(">> ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            match std::io::stdin().read_line(&mut input) {
                Ok(0) => {
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::Interrupted {
                        break;
                    } else {
                        return Err(e.into());
                    }
                }
            }
            let trimmed = input.trim();
            if trimmed == "exit" || trimmed == "quit" {
                break;
            }
            if trimmed.is_empty() {
                continue;
            }
            history.push(ChatMessage {
                role: "user".to_string(),
                content: trimmed.to_string(),
            });
        }

        if let tessella_inference::InferenceMode::Chat = mode {
            tracing::debug!(
                "Chat history being sent (session hint={:?}): {:?}",
                session_hint,
                history
            );

            let resp_result = match client
                .post(format!(
                    "{}/v1/chat/completions",
                    api_url.trim_end_matches('/')
                ))
                .json(&serde_json::json!({
                    "model": model_id,
                    "messages": history,
                    "stream": true,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    // A hint (`TASK-78`/S85): the id of the previous answer of this loop, or `null` on
                    // the first turn of the conversation.
                    "session_id": session_hint.clone(),
                    "stop": ["<|im_end|>", "<|endoftext|>"]
                }))
                .send()
                .await
            {
                Ok(res) => res,
                Err(e) => {
                    anyhow::bail!(
                        "Network Error: Failed to connect to worker/API at {}: {}",
                        api_url,
                        e
                    );
                }
            };

            if !resp_result.status().is_success() {
                return Err(format_api_error(resp_result, "Chat").await);
            }

            // `TASK-78`/S85 (`D-36`): the answer says which conversation it belongs to —the id this node
            // issued, not the hint— and that is the one the next turn has to name to resume the K/V. A
            // stream publishes it in a header (its body is SSE and its last event is already defined), and
            // the header is read here, before the body is consumed.
            if let Some(issued) = resp_result
                .headers()
                .get(tessella_api::handlers::SESSION_ID_HEADER)
                .and_then(|value| value.to_str().ok())
            {
                session_hint = Some(issued.to_string());
            }

            let stream = resp_result.bytes_stream();

            print!("🤖 Assistant: ");
            std::io::stdout().flush()?;

            let assistant_response = process_sse_stream(stream, |content| {
                print!("{}", content);
                let _ = std::io::stdout().flush();
            })
            .await?;

            println!();

            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: assistant_response,
            });

            if let Some(ref path) = session_file {
                let json = serde_json::to_string_pretty(&history)?;
                std::fs::write(path, json)?;
            }
        } else {
            let prompt_fim = history
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let resp_result = match client
                .post(format!("{}/v1/generate", api_url.trim_end_matches('/')))
                .json(&tessella_api::GenerateRequest {
                    prompt: prompt_fim,
                    mode: Some(mode.clone()),
                    max_tokens: Some(max_tokens),
                    stop: vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
                    temperature: Some(temperature),
                    top_p: None,
                    // S46 (`TASK-40`): a `--seed` / `--top-k` flag for the CLI is not added here;
                    // both fields are optional and the node applies the mesh standard when absent.
                    top_k: None,
                    seed: None,
                })
                .send()
                .await
            {
                Ok(res) => res,
                Err(e) => {
                    anyhow::bail!(
                        "Network Error: Failed to connect to worker/API at {}: {}",
                        api_url,
                        e
                    );
                }
            };

            if !resp_result.status().is_success() {
                return Err(format_api_error(resp_result, "FIM").await);
            }

            let resp: tessella_api::GenerateResponse = resp_result.json().await?;
            println!("🤖 Assistant: {}", resp.text.trim());
            break;
        }
    }

    Ok(())
}

async fn run_fim_stream(
    prefix: String,
    suffix: String,
    max_tokens: usize,
    temperature: f64,
    api_url: String,
    _model_id: String,
    _session_file: Option<String>,
) -> Result<()> {
    let client = reqwest::Client::new();
    let resp_result = match client
        .post(format!("{}/v1/generate", api_url.trim_end_matches('/')))
        .json(&tessella_api::GenerateRequest {
            prompt: String::new(),
            mode: Some(tessella_inference::InferenceMode::Fim { prefix, suffix }),
            max_tokens: Some(max_tokens),
            stop: vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
            temperature: Some(temperature),
            top_p: None,
            // S46 (`TASK-40`): both are optional; omitted ⇒ the mesh standard / no seed.
            top_k: None,
            seed: None,
        })
        .send()
        .await
    {
        Ok(res) => res,
        Err(e) => {
            anyhow::bail!(
                "Network Error: Failed to connect to worker/API at {}: {}",
                api_url,
                e
            );
        }
    };

    if !resp_result.status().is_success() {
        return Err(format_api_error(resp_result, "FIM Stream").await);
    }

    let resp: tessella_api::GenerateResponse = resp_result.json().await?;
    println!("🤖 Assistant: {}", resp.text.trim());

    Ok(())
}

async fn format_api_error(resp: reqwest::Response, context: &str) -> anyhow::Error {
    let status = resp.status();
    let err_text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            return anyhow::anyhow!(
                "API Error ({}) status {}: failed to read error body: {}",
                context,
                status,
                e
            )
        }
    };
    if err_text.trim().is_empty() {
        return anyhow::anyhow!("API Error ({}): status {}", context, status);
    }
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&err_text) {
        if let Some(err) = json.get("error") {
            if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                return anyhow::anyhow!("API Error ({}): {}", context, msg);
            }
        }
    }
    anyhow::anyhow!("API Error ({}): {}", context, err_text)
}

async fn process_sse_stream<F>(
    mut stream: impl futures_util::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
    mut on_content: F,
) -> Result<String>
where
    F: FnMut(&str),
{
    let mut full_response = String::new();
    let mut buffer = String::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        while let Some(idx) = buffer.find('\n') {
            let line = buffer[..idx].trim().to_string();
            buffer = buffer[idx + 1..].to_string();

            if line.starts_with("data: ") {
                let data = line.strip_prefix("data: ").unwrap();
                if data == "[DONE]" {
                    continue;
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(err) = json.get("error") {
                        if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                            anyhow::bail!("API Error (Session): {}", msg);
                        }
                    }
                    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                        if let Some(choice) = choices.first() {
                            if let Some(delta) = choice.get("delta") {
                                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                                {
                                    on_content(content);
                                    full_response.push_str(content);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(full_response)
}
