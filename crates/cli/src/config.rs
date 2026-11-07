use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClientConfig {
    pub orchestrator_url: String,
    pub keypair_path: String,
    pub default_model: String,
    pub worker: ClientWorkerSection,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClientWorkerSection {
    pub api_port: u16,
}

fn default_log_level() -> String {
    "info".to_string()
}

pub fn get_tessella_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tessella")
}

pub fn get_default_config_path() -> PathBuf {
    get_tessella_dir().join("config.toml")
}

impl ClientConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let content = fs::read_to_string(path)?;
        let config: ClientConfig = toml::from_str(&content)?;
        Ok(config)
    }

    #[allow(dead_code)]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), anyhow::Error> {
        let parent = path
            .as_ref()
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid config path"))?;
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
