//! Execution session: binds graph inputs, runs nodes in topological order
//! and extracts the declared outputs.

use std::collections::HashMap;
use std::path::Path;

use flint_error::{Error, Result};

use crate::graph::Graph;
use crate::ops;
use crate::tensor::Tensor;

/// A loaded graph plus its runtime value environment.
pub struct Session {
    graph: Graph,
    env: HashMap<String, Tensor>,
}

impl Session {
    /// Loads a model file, decoding initializers into the environment.
    pub fn load(path: &Path) -> Result<Session> {
        let graph = Graph::load(path)?;
        let mut env = HashMap::with_capacity(graph.initializers.len());
        for (name, t) in &graph.initializers {
            env.insert(name.clone(), t.clone());
        }
        Ok(Session { graph, env })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Binds one graph input by name, failing fast on unknown names.
    pub fn set_input(&mut self, name: &str, t: Tensor) -> Result<()> {
        if !self.graph.inputs.iter().any(|i| i.name == name) {
            return Err(Error::Model(format!(
                "graph has no input named {name:?}"
            )));
        }
        self.env.insert(name.to_string(), t);
        Ok(())
    }

    /// Executes every node and returns the declared outputs.
    pub fn run(&mut self) -> Result<HashMap<String, Tensor>> {
        for node in &self.graph.nodes {
            ops::run(node, &mut self.env)?;
        }
        let mut out = HashMap::new();
        for v in &self.graph.outputs {
            let t = self.env.get(&v.name).ok_or_else(|| {
                Error::Model(format!("graph output {:?} was not produced", v.name))
            })?;
            out.insert(v.name.clone(), t.clone());
        }
        Ok(out)
    }
}
