use std::collections::HashMap;
use std::thread;

use thuban_generate::{Engine, GenStats, Grammar, Piece, SamplingParams, SessionId};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

pub enum Command {
    Generate {
        prompt: String,
        max_tokens: usize,
        stop_extra: Vec<u32>,
        sampling: Option<SamplingParams>,
        grammar: Option<Grammar>,
        tx: UnboundedSender<Event>,
    },
    Close(SessionId),
}

pub enum Event {
    Started(SessionId),
    Piece(Piece),
    Done(GenStats),
    Failed(String),
}

#[derive(Clone)]
pub struct EngineHandle {
    cmd: UnboundedSender<Command>,
}

impl EngineHandle {
    pub fn send(&self, cmd: Command) {
        let _ = self.cmd.send(cmd);
    }

    pub fn sender(&self) -> UnboundedSender<Command> {
        self.cmd.clone()
    }
}

pub struct Client {
    pub id: SessionId,
    pub rx: UnboundedReceiver<Event>,
    guard: CloseGuard,
}

impl Client {
    pub(crate) fn new(
        id: SessionId,
        rx: UnboundedReceiver<Event>,
        cmd: UnboundedSender<Command>,
    ) -> Self {
        Self {
            id,
            rx,
            guard: CloseGuard { id, cmd },
        }
    }

    pub fn into_parts(self) -> (UnboundedReceiver<Event>, CloseGuard) {
        (self.rx, self.guard)
    }
}

pub struct CloseGuard {
    id: SessionId,
    cmd: UnboundedSender<Command>,
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

pub fn spawn(engine: Engine) -> EngineHandle {
    let (cmd, rx) = unbounded_channel();
    thread::Builder::new()
        .name("thuban-engine".into())
        .spawn(move || run(engine, rx))
        .expect("spawn the engine thread");
    EngineHandle { cmd }
}

fn run(mut engine: Engine, mut rx: UnboundedReceiver<Command>) {
    let mut clients: HashMap<u32, UnboundedSender<Event>> = HashMap::new();
    let result: Result<(), String> = 'outer: loop {
        while let Ok(cmd) = rx.try_recv() {
            if let Err(e) = handle(&mut engine, &mut clients, cmd) {
                break 'outer Err(e);
            }
        }
        if clients.is_empty() {
            match rx.blocking_recv() {
                Some(cmd) => {
                    if let Err(e) = handle(&mut engine, &mut clients, cmd) {
                        break 'outer Err(e);
                    }
                }
                None => break 'outer Ok(()),
            }
            continue;
        }
        if let Err(e) = engine.step() {
            break 'outer Err(e.to_string());
        }
        let mut done = Vec::new();
        let mut gone = Vec::new();
        for (&id, tx) in &clients {
            for piece in engine.poll(SessionId(id)) {
                if tx.send(Event::Piece(piece)).is_err() {
                    gone.push(id);
                    break;
                }
            }
            if !gone.contains(&id) && engine.finished(SessionId(id)) {
                done.push(id);
            }
        }
        for id in gone {
            clients.remove(&id);
            if let Err(e) = engine.close(SessionId(id)) {
                break 'outer Err(e.to_string());
            }
        }
        for id in done {
            let tx = clients
                .remove(&id)
                .expect("finished sessions are registered");
            if let Some(stats) = engine.stats(SessionId(id)) {
                let _ = tx.send(Event::Done(stats));
            }
            if let Err(e) = engine.close(SessionId(id)) {
                break 'outer Err(e.to_string());
            }
        }
    };
    if let Err(msg) = result {
        eprintln!("[server] engine worker failed: {msg}");
        for (_, tx) in clients.drain() {
            let _ = tx.send(Event::Failed(msg.clone()));
        }
    }
}

fn handle(
    engine: &mut Engine,
    clients: &mut HashMap<u32, UnboundedSender<Event>>,
    cmd: Command,
) -> Result<(), String> {
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
                Ok(())
            }
            Err(e) => {
                let _ = tx.send(Event::Failed(e.to_string()));
                Ok(())
            }
        },
        Command::Close(id) => {
            engine.close(id).map_err(|e| e.to_string())?;
            clients.remove(&id.0);
            Ok(())
        }
    }
}
