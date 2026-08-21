use thuban_backend::{Backend, Binding, Commands};
use thuban_error::Result;
use thuban_model::Speculator;
use thuban_model::ops::{self, NormSpec};

use super::model::Model;

impl Speculator for Model {
    fn draft(
        &mut self,
        backend: &mut Backend,
        _seq: u32,
        _token: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        assert_eq!(
            hidden.len(),
            self.cfg.hidden as usize,
            "hidden size mismatch"
        );
        backend.write_f32(&self.s.hidden.buf, hidden);
        let cfg = &self.cfg;
        let mut enc = backend.encoder()?;
        {
            let mut commands = Commands::begin(&mut enc);
            ops::norm(
                backend,
                &mut commands,
                &NormSpec::new(self.norm_mode(), 1, cfg.hidden, cfg.norm_eps),
                Binding::Full(&self.s.hidden),
                &self.norm,
                self.norm_bias(self.norm_bias.as_ref()),
                Binding::Full(&self.s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&self.s.normed),
                self.head_weight(),
                Binding::Full(&self.s.logits),
                1,
            )?;
            if let Some(lb) = &self.lm_bias {
                ops::bias(
                    backend,
                    &mut commands,
                    Binding::Full(&self.s.logits),
                    lb,
                    1,
                    cfg.vocab,
                )?;
            }
            if let Some(cap) = cfg.softcap {
                ops::softcap(
                    backend,
                    &mut commands,
                    Binding::Full(&self.s.logits),
                    cfg.vocab,
                    cap,
                )?;
            }
        }
        backend.submit(&mut enc)?;
        backend.read_f32(&self.s.logits.buf, 0, cfg.vocab as usize)
    }

    fn prime(&mut self, _seq: u32) {}

    fn snapshot(&mut self, _backend: &Backend, seq: u32) {
        self.saved_pos[seq as usize] = self.pos[seq as usize];
    }

    fn restore(&mut self, _backend: &Backend, seq: u32) {
        self.pos[seq as usize] = self.saved_pos[seq as usize] + 1;
    }
}
