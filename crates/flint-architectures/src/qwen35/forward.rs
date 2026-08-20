use flint_backend::{Backend, Binding, Commands};
use flint_error::{Error, Result};
use flint_model::ops::{self, NormMode};
use flint_model::rows;
use flint_model::{ChunkOut, LanguageModel, MAX_M, SeqChunk, Speculator};

use super::layers::{FullCtx, full_layer, linear_layer};
use super::model::Qwen35;
use super::weights::Layer;

impl LanguageModel for Qwen35 {
    fn forward(&mut self, backend: &mut Backend, batch: &[SeqChunk]) -> Result<Vec<ChunkOut>> {
        let m: u32 = batch.iter().map(SeqChunk::len).sum();
        if m == 0 || m > MAX_M {
            return Err(Error::Model(format!("chunk size {m} outside [1, {MAX_M}]")));
        }
        let mut ids = vec![0u32; MAX_M as usize];
        let mut positions = vec![0u32; MAX_M as usize];
        let mut seqs = vec![0u32; MAX_M as usize];
        let mut row = 0usize;
        for chunk in batch {
            let s = chunk.seq as usize;
            let limit = self.arena.seq_len(chunk.seq);
            if self.pos[s] + chunk.len() > limit {
                return Err(Error::Model(format!("context limit {} reached", limit)));
            }
            assert!(
                self.arena.covers(chunk.seq, self.pos[s] + chunk.len()),
                "pages must cover the chunk"
            );
            for i in 0..chunk.tokens.len() {
                ids[row + i] = chunk.tokens[i];
                positions[row + i] = self.pos[s] + i as u32;
                seqs[row + i] = chunk.seq;
            }
            row += chunk.tokens.len();
        }
        backend.write_u32(&self.s.ids.buf, &ids);
        rows::write_row_meta(backend, &self.s.meta, &positions, &seqs, m);

        let cfg = &self.cfg;
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
                    rows: m,
                    dim: cfg.hidden,
                    scale: 1.0,
                    split: 0,
                },
            )?;

            let ctx = FullCtx {
                cfg,
                s,
                cos: &self.cos,
                sin: &self.sin,
            };
            for layer in &self.layers {
                match layer {
                    Layer::Full { w, kv } => {
                        full_layer(backend, &mut commands, &ctx, w, kv, m, batch)?
                    }
                    Layer::Linear { w, state } => {
                        linear_layer(backend, &mut commands, &ctx, w, state, m, batch)?
                    }
                }
            }

            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, m, cfg.hidden, 1e-6),
                Binding::Full(&s.hidden),
                &self.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.normed),
                self.logit_head(),
                Binding::Full(&s.logits),
                m,
            )?;
        }
        backend.submit(&mut enc)?;

        let mut outs = Vec::with_capacity(batch.len());
        let mut base = 0u32;
        for chunk in batch {
            let m_s = chunk.len();
            let s = chunk.seq as usize;
            outs.push(ChunkOut {
                logits: rows::read_rows(
                    backend,
                    &self.s.logits,
                    chunk.logit_rows,
                    m_s,
                    cfg.vocab,
                    base,
                )?,
                hidden: rows::read_rows(
                    backend,
                    &self.s.hidden,
                    chunk.hidden_rows,
                    m_s,
                    cfg.hidden,
                    base,
                )?,
            });
            base += m_s;
            self.pos[s] += m_s;
        }
        Ok(outs)
    }

    fn reset(&mut self, backend: &Backend, seq: u32) -> Result<()> {
        self.arena.free_seq(seq);
        self.upload_tables(backend);
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                state.zero(backend, seq)?;
            }
        }
        self.pos[seq as usize] = 0;
        self.saved_pos[seq as usize] = 0;
        self.mtp_pos[seq as usize] = 0;
        Ok(())
    }

    fn pos(&self, seq: u32) -> u32 {
        self.pos[seq as usize]
    }
    fn context_limit(&self, seq: u32) -> u32 {
        self.arena.seq_len(seq)
    }
    fn seq_count(&self) -> u32 {
        self.arena.seqs()
    }
    fn vocab(&self) -> u32 {
        self.cfg.vocab
    }
    fn eos(&self) -> &[u32] {
        &self.cfg.eos
    }

    fn alloc_pages(&mut self, backend: &Backend, seq: u32, tokens: u32) -> Result<()> {
        self.arena.alloc(seq, self.pos[seq as usize], tokens)?;
        self.upload_tables(backend);
        Ok(())
    }

    fn free_pages(&mut self, backend: &Backend, seq: u32) -> Result<()> {
        self.arena.free_seq(seq);
        self.upload_tables(backend);
        Ok(())
    }

    fn truncate_pages(&mut self, backend: &Backend, seq: u32, keep_tokens: u32) -> Result<()> {
        self.arena.truncate(seq, keep_tokens);
        self.upload_tables(backend);
        Ok(())
    }

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        self.mtp.is_some().then_some(self as &mut dyn Speculator)
    }
}

impl Qwen35 {
    fn upload_tables(&self, backend: &Backend) {
        let table = self.arena.table();
        for layer in &self.layers {
            if let Layer::Full { kv, .. } = layer {
                kv.upload(backend, &table);
            }
        }
        if let Some(mtp) = &self.mtp {
            mtp.kv.upload(backend, &table);
        }
    }
}
