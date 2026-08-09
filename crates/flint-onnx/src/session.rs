use std::collections::HashMap;
use std::path::Path;

use flint_error::{Error, Result};

use crate::graph::Graph;
use crate::ops;
use crate::tensor::Tensor;

pub struct Session {
    graph: Graph,
    env: HashMap<String, Tensor>,
}

impl Session {

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

    pub fn set_input(&mut self, name: &str, t: Tensor) -> Result<()> {
        if !self.graph.inputs.iter().any(|i| i.name == name) {
            return Err(Error::Model(format!(
                "graph has no input named {name:?}"
            )));
        }
        self.env.insert(name.to_string(), t);
        Ok(())
    }

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
