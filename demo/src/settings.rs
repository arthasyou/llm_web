use config::{Config, ConfigError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub llm: crate::llm::LLMCfg,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let c = Config::builder()
            .add_source(config::File::with_name("config/llm"))
            .build()?;
        c.try_deserialize()
    }
}
