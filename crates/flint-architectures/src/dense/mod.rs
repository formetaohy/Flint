//! The dense GQA transformer family (LLaMA, Gemma, Phi, Qwen): one forward
//! graph configured per family, with dense or MoE FFNs.

mod config;
mod model;
mod weights;

pub use config::{DenseConfig, MoeConfig, PerLayerConfig, RopeSpec};
pub use model::DenseModel;
pub use weights::{dense_plan, dense_role};

