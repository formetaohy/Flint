use flint_backend::{Backend, Binding, Commands};
use flint_error::{Error, Result};
use flint_model::ops::{self, NormMode};
use flint_model::rows;
use flint_model::{MAX_M, SeqChunk, Speculator};

use super::layers::{FullCtx, full_layer};
use super::model::Qwen35;
use super::weights::Layer;

impl Speculator for Qwen35 {
    fn draft(
        &mut self,
        backend: &mut Backend,
        seq: u32,
        token: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let Some(mtp) = self.mtp.as_ref() else {
            unreachable!("the speculator exists only with an MTP head");
        };
        assert_eq!(
            hidden.len(),
            self.cfg.hidden as usize,
            "hidden size mismatch"
        );
        assert!(
            self.mtp_pos[seq as usize] <= self.pos[seq as usize],
            "draft head ran past the target"
        );
        let limit = self.arena.seq_len(seq);
        if self.mtp_pos[seq as usize] >= limit {
            return Err(Error::Model(format!("context limit {limit} reached")));
        }
        assert!(
            self.arena.covers(seq, self.mtp_pos[seq as usize] + 1),
            "pages must cover the draft"
        );

        let mut ids = vec![0u32; MAX_M as usize];
        ids[0] = token;
        backend.write_u32(&self.s.ids.buf, &ids);
        rows::write_row_meta(
            backend,
            &self.s.meta,
            &[self.mtp_pos[seq as usize]],
            &[seq],
            1,
        );
        backend.write_f32(&self.s.mtp_hidden.buf, hidden);

        let cfg = &self.cfg;
        let chunk = SeqChunk {
            tokens: &ids[..1],
            seq,
            logit_rows: &[],
            hidden_rows: &[],
        };
        let mut enc = backend.encoder()?;
        {
            let mut commands = Commands::begin(&mut enc);
            let s = &self.s;
            ops::embed(
                backend,
                &mut commands,
                &s.ids,
                &self.embed,
                Binding::Full(&s.hidden),
                &ops::EmbedSpec {
                    rows: 1,
                    dim: cfg.hidden,
                    scale: 1.0,
                    split: 0,
                },
            )?;
            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, 1, cfg.hidden, 1e-6),
                Binding::Full(&s.hidden),
                &mtp.pre_fc_norm_embedding,
                Binding::Full(&s.hidden),
                Binding::Full(&s.mtp_emb),
            )?;
            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, 1, cfg.hidden, 1e-6),
                Binding::Full(&s.mtp_hidden),
                &mtp.pre_fc_norm_hidden,
                Binding::Full(&s.mtp_hidden),
                Binding::Full(&s.normed),
            )?;
            ops::concat(
                backend,
                &mut commands,
                Binding::Full(&s.mtp_emb),
                Binding::Full(&s.normed),
                Binding::Full(&s.mtp_concat),
                1,
                cfg.hidden,
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.mtp_concat),
                &mtp.fc,
                Binding::Full(&s.hidden),
                1,
            )?;

            let ctx = FullCtx {
                cfg,
                s,
                cos: &self.cos,
                sin: &self.sin,
            };
            full_layer(
                backend,
                &mut commands,
                &ctx,
                &mtp.layer,
                &mtp.kv,
                1,
                std::slice::from_ref(&chunk),
            )?;

            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, 1, cfg.hidden, 1e-6),
                Binding::Full(&s.hidden),
                &mtp.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.normed),
                self.logit_head(),
                Binding::Full(&s.logits),
                1,
            )?;
        }
        backend.submit(&mut enc)?;

        let logits = backend.read_f32(&self.s.logits.buf, 0, cfg.vocab as usize)?;
        self.mtp_pos[seq as usize] += 1;
        Ok(logits)
    }

    fn advance(
        &mut self,
        backend: &mut Backend,
        seq: u32,
        token: u32,
        hidden: &[f32],
    ) -> Result<()> {
        self.draft(backend, seq, token, hidden).map(|_| ())
    }

    fn prime(&mut self, seq: u32) {
        self.mtp_pos[seq as usize] = self.pos[seq as usize];
    }

    fn snapshot(&mut self, backend: &Backend, seq: u32) {
        self.saved_pos[seq as usize] = self.pos[seq as usize];
        let mut si = 0;
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                self.snap[si]
                    .copy_seq(backend, state, seq)
                    .expect("snapshot copy");
                si += 1;
            }
        }
    }

    fn restore(&mut self, backend: &Backend, seq: u32) {
        self.pos[seq as usize] = self.saved_pos[seq as usize] + 1;
        let mut si = 0;
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                state
                    .copy_seq(backend, &self.snap[si], seq)
                    .expect("restore copy");
                si += 1;
            }
        }
    }
}
