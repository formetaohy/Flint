use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use flint_generate::{Engine, GenStats, Grammar, SamplingParams, SessionId};

pub enum Command {
    Generate {
        prompt: String,
        max_tokens: usize,
        stop_extra: Vec<u32>,
        sampling: Option<SamplingParams>,
        grammar: Option<Grammar>,
        tx: Sender<Event>,
    },
    Close(SessionId),
}

pub enum Event {
    Started(SessionId),
    Piece(String),
    Done(GenStats),
    Failed(String),
}

pub struct CloseGuard {
    pub(crate) id: SessionId,
    pub(crate) cmd: Sender<Command>,
}

impl CloseGuard {
    pub fn close_now(self) {
        let _ = self.cmd.send(Command::Close(self.id));
    }
}

impl Drop for CloseGuard {
    fn drop(&mut self) {
        let _ = self.cmd.send(Command::Close(self.id));
    }
}

pub struct Client {
    pub id: SessionId,
    pub rx: Receiver<Event>,
    pub guard: CloseGuard,
}

pub fn spawn(engine: Engine) -> Sender<Command> {
    let (tx, rx) = channel();
    thread::Builder::new()
        .name("flint-engine".into())
        .spawn(move || run(engine, rx))
        .expect("spawn the engine thread");
    tx
}

fn run(mut engine: Engine, rx: Receiver<Command>) {
    let mut clients: HashMap<u32, Sender<Event>> = HashMap::new();
    loop {
        while let Ok(cmd) = rx.try_recv() {
            handle(&mut engine, &mut clients, cmd);
        }
        if clients.is_empty() {
            match rx.recv() {
                Ok(cmd) => handle(&mut engine, &mut clients, cmd),
                Err(_) => break,
            }
            continue;
        }
        if let Err(e) = engine.step() {
            eprintln!("[server] engine step failed: {e}");
            for (_, tx) in clients.drain() {
                let _ = tx.send(Event::Failed(e.to_string()));
            }
            continue;
        }
        let mut done = Vec::new();
        for (&id, tx) in &clients {
            for piece in engine.poll(SessionId(id)) {
                let _ = tx.send(Event::Piece(piece.text));
            }
            if engine.finished(SessionId(id)) {
                done.push(id);
            }
        }
        for id in done {
            let tx = clients.remove(&id).expect("done sessions are registered");
            if let Some(stats) = engine.stats(SessionId(id)) {
                let _ = tx.send(Event::Done(stats));
            }
            let _ = engine.close(SessionId(id));
        }
    }
}

fn handle(engine: &mut Engine, clients: &mut HashMap<u32, Sender<Event>>, cmd: Command) {
    match cmd {
        Command::Generate {
            prompt,
            max_tokens,
            stop_extra,
            sampling,
            grammar,
            tx,
        } => match engine.create(&prompt, max_tokens, grammar, sampling, &stop_extra) {
            Ok(id) => {
                clients.insert(id.0, tx.clone());
                let _ = tx.send(Event::Started(id));
            }
            Err(e) => {
                let _ = tx.send(Event::Failed(e.to_string()));
            }
        },
        Command::Close(id) => {
            let _ = engine.close(id);
            clients.remove(&id.0);
        }
    }
}
