use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use thuban_error::{Error, Result};
use thuban_tokenizer::Tokenizer;
use serde_json::{Map, Value};

pub enum Rule {
    Lit(Vec<u8>),
    Class(Vec<(u8, u8)>),
    Alt {
        a: u32,
        b: u32,
    },
    Seq {
        a: u32,
        b: u32,
    },
    Repeat {
        rule: u32,
        min: u32,
        max: Option<u32>,
    },
}

pub struct Grammar {
    rules: Vec<Rule>,
    start: u32,
}

impl Grammar {
    pub fn from_schema(schema: &Value) -> Result<Self> {
        let defs = schema
            .get("$defs")
            .or_else(|| schema.get("definitions"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut b = Builder {
            rules: Vec::new(),
            defs,
        };
        let start = b.build(schema)?;
        Ok(Self {
            rules: b.rules,
            start,
        })
    }
}

struct Builder {
    rules: Vec<Rule>,
    defs: Map<String, Value>,
}

impl Builder {
    fn push(&mut self, rule: Rule) -> u32 {
        self.rules.push(rule);
        (self.rules.len() - 1) as u32
    }

    fn lit(&mut self, text: &str) -> u32 {
        self.push(Rule::Lit(text.as_bytes().to_vec()))
    }

    fn seq(&mut self, a: u32, b: u32) -> u32 {
        self.push(Rule::Seq { a, b })
    }

    fn alt(&mut self, a: u32, b: u32) -> u32 {
        self.push(Rule::Alt { a, b })
    }

    fn rep(&mut self, rule: u32, min: u32, max: Option<u32>) -> u32 {
        self.push(Rule::Repeat { rule, min, max })
    }

    fn opt(&mut self, rule: u32) -> u32 {
        self.rep(rule, 0, Some(1))
    }

    fn class(&mut self, ranges: Vec<(u8, u8)>) -> u32 {
        self.push(Rule::Class(ranges))
    }

    fn digits(&mut self) -> u32 {
        let d = self.class(vec![(b'0', b'9')]);
        self.rep(d, 1, None)
    }

    fn build(&mut self, schema: &Value) -> Result<u32> {
        if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
            let key = r
                .rsplit_once('/')
                .map(|(_, k)| k)
                .ok_or_else(|| Error::Config(format!("unsupported $ref {r:?}")))?;
            let def = self
                .defs
                .get(key)
                .cloned()
                .ok_or_else(|| Error::Config(format!("$ref {r:?} not found in $defs")))?;
            return self.build(&def);
        }
        if let Some(items) = schema.get("anyOf").or_else(|| schema.get("oneOf")) {
            let mut alts = items
                .as_array()
                .ok_or_else(|| Error::Config("anyOf/oneOf must be an array".into()))?
                .iter()
                .map(|s| self.build(s))
                .collect::<Result<Vec<_>>>()?;
            let first = alts.remove(0);
            return Ok(alts.into_iter().fold(first, |acc, r| self.alt(acc, r)));
        }
        if let Some(v) = schema.get("enum").or_else(|| schema.get("const")) {
            let mut consts: Vec<Value> = match v {
                Value::Array(a) => a.clone(),
                other => vec![other.clone()],
            };
            let first = serde_json::to_string(&consts.remove(0))
                .map_err(|e| Error::Config(e.to_string()))?;
            let mut rule = self.lit(&first);
            for c in consts {
                let s = serde_json::to_string(&c).map_err(|e| Error::Config(e.to_string()))?;
                let alt = self.lit(&s);
                rule = self.alt(rule, alt);
            }
            return Ok(rule);
        }
        match schema.get("type").and_then(Value::as_str) {
            Some("object") => self.object(schema),
            Some("array") => self.array(schema),
            Some("string") => {
                let chars = self.class(vec![(0x20, 0x21), (0x23, 0x5B), (0x5D, 0xFF)]);
                let body = self.rep(chars, 0, None);
                let open = self.lit("\"");
                let close = self.lit("\"");
                let inner = self.seq(open, body);
                Ok(self.seq(inner, close))
            }
            Some("integer") => Ok(self.integer()),
            Some("number") => Ok(self.number()),
            Some("boolean") => {
                let t = self.lit("true");
                let f = self.lit("false");
                Ok(self.alt(t, f))
            }
            Some("null") => Ok(self.lit("null")),
            other => Err(Error::Config(format!("unsupported schema type {other:?}"))),
        }
    }

    fn integer(&mut self) -> u32 {
        let minus = self.lit("-");
        let minus = self.opt(minus);
        let zero = self.lit("0");
        let d19 = self.class(vec![(b'1', b'9')]);
        let d09 = self.class(vec![(b'0', b'9')]);
        let rest = self.rep(d09, 0, None);
        let nz = self.seq(d19, rest);
        let int = self.alt(zero, nz);
        self.seq(minus, int)
    }

    fn number(&mut self) -> u32 {
        let int = self.integer();
        let dot = self.lit(".");
        let digits = self.digits();
        let frac = self.seq(dot, digits);
        let frac = self.opt(frac);
        let e = self.class(vec![(b'e', b'e'), (b'E', b'E')]);
        let sign = self.class(vec![(b'+', b'+'), (b'-', b'-')]);
        let sign = self.opt(sign);
        let digits = self.digits();
        let mantissa = self.seq(sign, digits);
        let exp = self.seq(e, mantissa);
        let exp = self.opt(exp);
        let frac_exp = self.seq(frac, exp);
        self.seq(int, frac_exp)
    }

    fn object(&mut self, schema: &Value) -> Result<u32> {
        let props = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let required: HashSet<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let order = schema
            .get("propertyOrder")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .filter(|n| props.contains_key(*n))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut names: Vec<&String> = props
            .keys()
            .filter(|k| !order.contains(&k.as_str()))
            .collect();
        names.splice(
            0..0,
            order.iter().map(|n| {
                props
                    .keys()
                    .find(|k| k.as_str() == *n)
                    .expect("propertyOrder names are validated against the properties")
            }),
        );
        let mut pre: Option<u32> = None;
        let mut opt_any: Option<u32> = None;
        let mut first_required = true;
        for name in names {
            let sub = &props[name];
            let key = self.lit(&format!("\"{name}\":"));
            let value = self.build(sub)?;
            let prop = self.seq(key, value);
            if required.contains(name.as_str()) {
                let element = if first_required {
                    first_required = false;
                    prop
                } else {
                    let comma = self.lit(",");
                    self.seq(comma, prop)
                };
                pre = Some(match pre {
                    None => element,
                    Some(prev) => self.seq(prev, element),
                });
            } else {
                opt_any = Some(match opt_any {
                    None => prop,
                    Some(prev) => self.alt(prev, prop),
                });
            }
        }
        let list = match opt_any {
            None => None,
            Some(any) => {
                let comma = self.lit(",");
                let comma_prop = self.seq(comma, any);
                let tail = self.rep(comma_prop, 0, None);
                Some(self.seq(any, tail))
            }
        };
        let body = match (pre, list) {
            (None, None) => self.lit(""),
            (None, Some(list)) => self.opt(list),
            (Some(pre), None) => pre,
            (Some(pre), Some(list)) => {
                let comma = self.lit(",");
                let rest = self.seq(comma, list);
                let rest = self.opt(rest);
                self.seq(pre, rest)
            }
        };
        let open = self.lit("{");
        let close = self.lit("}");
        let inner = self.seq(open, body);
        Ok(self.seq(inner, close))
    }

    fn array(&mut self, schema: &Value) -> Result<u32> {
        let item = schema
            .get("items")
            .ok_or_else(|| Error::Config("array schema has no items".into()))?;
        let item = self.build(item)?;
        let comma = self.lit(",");
        let elem = self.seq(comma, item);
        let tail = self.rep(elem, 0, None);
        let body = self.seq(item, tail);
        let body = self.opt(body);
        let open = self.lit("[");
        let close = self.lit("]");
        let inner = self.seq(open, body);
        Ok(self.seq(inner, close))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Frame {
    rule: u32,
    pos: u32,
    count: u32,
}

impl Frame {
    fn new(rule: u32) -> Self {
        Self {
            rule,
            pos: 0,
            count: 0,
        }
    }
}

pub struct TokenTrie {
    vocab: usize,
    root: usize,
    next: Vec<Vec<(u8, u32)>>,
    ids: Vec<Vec<u32>>,
    spellings: Vec<Vec<u8>>,
    alphabet: Vec<u8>,
}

impl TokenTrie {
    pub fn from_tokenizer(tokenizer: &Tokenizer, vocab: usize) -> Self {
        let mut tokens = Vec::new();
        for id in 0..tokenizer.vocab_size().min(vocab as u32) {
            if let Some(bytes) = tokenizer.decode_id(id) {
                tokens.push((id, bytes));
            }
        }
        Self::build(vocab, &tokens)
    }

    fn build(vocab: usize, tokens: &[(u32, Vec<u8>)]) -> Self {
        let mut t = Self {
            vocab,
            root: 0,
            next: vec![Vec::new()],
            ids: vec![Vec::new()],
            spellings: vec![Vec::new(); vocab],
            alphabet: Vec::new(),
        };
        for (id, bytes) in tokens {
            let mut node = t.root;
            for &b in bytes.iter() {
                if !t.alphabet.contains(&b) {
                    t.alphabet.push(b);
                }
                let child = match t.next[node].iter().find(|(c, _)| *c == b) {
                    Some((_, c)) => *c,
                    None => {
                        let c = t.next.len() as u32;
                        t.next.push(Vec::new());
                        t.ids.push(Vec::new());
                        t.next[node].push((b, c));
                        c
                    }
                };
                node = child as usize;
            }
            t.ids[node].push(*id);
            t.spellings[*id as usize] = bytes.clone();
        }
        t
    }

    fn transitions(&self, node: usize) -> &[(u8, u32)] {
        &self.next[node]
    }

    fn token_ids(&self, node: usize) -> &[u32] {
        &self.ids[node]
    }
}

pub struct Matcher {
    grammar: Arc<Grammar>,
    states: Vec<Vec<Frame>>,
}

impl Clone for Matcher {
    fn clone(&self) -> Self {
        Self {
            grammar: self.grammar.clone(),
            states: self.states.clone(),
        }
    }
}

impl Matcher {
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let mut m = Self {
            grammar,
            states: Vec::new(),
        };
        let start = vec![vec![Frame::new(m.grammar.start)]];
        m.states = m.epsilon_close(start);
        m
    }

    fn rule(&self, idx: u32) -> &Rule {
        &self.grammar.rules[idx as usize]
    }

    fn epsilon_close(&self, states: Vec<Vec<Frame>>) -> Vec<Vec<Frame>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let mut work = states;
        while let Some(mut stack) = work.pop() {
            if !seen.insert(stack.clone()) {
                continue;
            }
            match stack.last().map(|f| self.rule(f.rule)) {
                Some(Rule::Lit(bytes)) => {
                    if stack.last().unwrap().pos == bytes.len() as u32 {
                        stack.pop();
                        work.push(stack);
                    } else {
                        out.push(stack);
                    }
                }
                Some(Rule::Class(_)) => out.push(stack),
                Some(Rule::Alt { a, b }) => {
                    for branch in [*a, *b] {
                        let mut s = stack.clone();
                        s.pop();
                        s.push(Frame::new(branch));
                        work.push(s);
                    }
                }
                Some(Rule::Seq { a, b }) => {
                    let stage = stack.last().unwrap().pos;
                    if stage == 2 {
                        stack.pop();
                        work.push(stack);
                    } else {
                        stack.last_mut().unwrap().pos += 1;
                        stack.push(Frame::new(if stage == 0 { *a } else { *b }));
                        work.push(stack);
                    }
                }
                Some(Rule::Repeat { rule, min, max }) => {
                    let count = stack.last().unwrap().count;
                    if count >= *min {
                        let mut done = stack.clone();
                        done.pop();
                        work.push(done);
                    }
                    if max.is_none_or(|m| count < m) {
                        stack.last_mut().unwrap().count += 1;
                        stack.push(Frame::new(*rule));
                        work.push(stack);
                    }
                }
                None => out.push(stack),
            }
        }
        out
    }

    fn advance(&self, states: &[Vec<Frame>], byte: u8) -> Vec<Vec<Frame>> {
        let mut next = Vec::new();
        for stack in states {
            let Some(top) = stack.last().copied() else {
                continue;
            };
            match self.rule(top.rule) {
                Rule::Lit(bytes) => {
                    let pos = top.pos as usize;
                    if pos < bytes.len() && bytes[pos] == byte {
                        let mut s = stack.clone();
                        if pos + 1 == bytes.len() {
                            s.pop();
                        } else {
                            s.last_mut().unwrap().pos += 1;
                        }
                        next.push(s);
                    }
                }
                Rule::Class(ranges) => {
                    if ranges.iter().any(|(lo, hi)| *lo <= byte && byte <= *hi) {
                        let mut s = stack.clone();
                        s.pop();
                        next.push(s);
                    }
                }
                _ => unreachable!("unclosed frame on the byte frontier"),
            }
        }
        self.epsilon_close(next)
    }

    pub fn commit(&mut self, trie: &TokenTrie, token: u32) {
        let bytes = trie.spellings[token as usize].clone();
        let mut next = Vec::new();
        for stack in &self.states {
            let mut cur = vec![stack.clone()];
            let mut alive = true;
            for &byte in &bytes {
                cur = self.advance(&cur, byte);
                if cur.is_empty() {
                    alive = false;
                    break;
                }
            }
            if alive {
                next.extend(cur);
            }
        }
        assert!(
            !next.is_empty(),
            "committing token {token} that the grammar forbids"
        );
        self.states = next;
    }

    pub fn mask(&self, trie: &TokenTrie) -> Vec<f32> {
        let mut mask = vec![0.0f32; trie.vocab];
        let mut memo = HashMap::new();
        for stack in &self.states {
            self.walk(stack, trie.root, &mut mask, trie, &mut memo);
        }
        mask
    }

    fn walk(
        &self,
        stack: &[Frame],
        node: usize,
        mask: &mut [f32],
        trie: &TokenTrie,
        memo: &mut HashMap<Vec<Frame>, bool>,
    ) {
        if self.viable(stack, trie, memo) {
            for &id in trie.token_ids(node) {
                mask[id as usize] = 1.0;
            }
        }
        for &(byte, child) in trie.transitions(node) {
            for next in self.advance(&[stack.to_vec()], byte) {
                self.walk(&next, child as usize, mask, trie, memo);
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.states.iter().any(|s| s.is_empty())
    }

    fn viable(
        &self,
        stack: &[Frame],
        trie: &TokenTrie,
        memo: &mut HashMap<Vec<Frame>, bool>,
    ) -> bool {
        if stack.is_empty() {
            return true;
        }
        let key = self.canonicalize(stack);
        if let Some(&r) = memo.get(&key) {
            return r;
        }
        memo.insert(key.clone(), false);
        let result = self.token_continues(stack, trie.root, trie, memo);
        memo.insert(key, result);
        result
    }

    fn token_continues(
        &self,
        stack: &[Frame],
        node: usize,
        trie: &TokenTrie,
        memo: &mut HashMap<Vec<Frame>, bool>,
    ) -> bool {
        if self.viable(stack, trie, memo) && !trie.token_ids(node).is_empty() {
            return true;
        }
        for &(byte, child) in trie.transitions(node) {
            for next in self.advance(&[stack.to_vec()], byte) {
                if self.token_continues(&next, child as usize, trie, memo) {
                    return true;
                }
            }
        }
        false
    }

    fn canonicalize(&self, stack: &[Frame]) -> Vec<Frame> {
        let mut out = Vec::with_capacity(stack.len());
        for f in stack {
            let mut f = *f;
            if let Rule::Repeat { min, max, .. } = self.rule(f.rule) {
                let (min, max) = (*min, *max);
                if f.count >= min {
                    f.count = match max {
                        Some(m) if f.count >= m => m,
                        _ => min,
                    };
                }
            }
            out.push(f);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn trie(tokens: &[(&str, u32)], vocab: usize) -> TokenTrie {
        let v: Vec<(u32, Vec<u8>)> = tokens
            .iter()
            .map(|(s, id)| (*id, s.as_bytes().to_vec()))
            .collect();
        TokenTrie::build(vocab, &v)
    }

    fn matcher(schema: Value, _t: &TokenTrie) -> Matcher {
        let g = Arc::new(Grammar::from_schema(&schema).unwrap());
        Matcher::new(g)
    }

    #[test]
    fn enum_constrains_to_literal_tokens() {
        let t = trie(
            &[
                ("{", 0),
                ("\"status\"", 1),
                (":", 2),
                ("\"ok\"", 3),
                ("\"error\"", 4),
                ("}", 5),
            ],
            6,
        );
        let mut m = matcher(
            json!({"type": "object", "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]}
            }}),
            &t,
        );
        m.commit(&t, 0);
        m.commit(&t, 1);
        m.commit(&t, 2);
        let mask = m.mask(&t);
        assert_eq!(mask[3], 1.0);
        assert_eq!(mask[4], 1.0);
        assert_eq!(mask[5], 0.0);
        m.commit(&t, 3);
        let mask = m.mask(&t);
        assert_eq!(mask[5], 1.0);
        assert!(!m.is_complete(), "the closing brace is still pending");
        m.commit(&t, 5);
        assert!(m.is_complete());
    }

    #[test]
    fn token_splitting_across_literals() {
        let t = trie(&[("\"h", 0), ("i\"", 1), ("\"", 2), ("x", 3)], 4);
        let mut m = matcher(json!({"type": "string", "enum": ["hi"]}), &t);
        let mask = m.mask(&t);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 0.0);
        assert_eq!(mask[2], 0.0);
        assert_eq!(mask[3], 0.0);
        m.commit(&t, 0);
        let mask = m.mask(&t);
        assert_eq!(mask[1], 1.0);
        m.commit(&t, 1);
        assert!(m.is_complete());
    }

    #[test]
    fn number_rejects_non_numeric_prefixes() {
        let t = trie(
            &[
                ("1", 0),
                ("23", 1),
                (".5", 2),
                ("true", 3),
                ("-", 4),
                ("0", 5),
            ],
            6,
        );
        let m = matcher(json!({"type": "number"}), &t);
        let mask = m.mask(&t);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 1.0, "multi-digit tokens are valid numbers");
        assert_eq!(mask[2], 0.0);
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[4], 1.0);
        assert_eq!(mask[5], 1.0);
    }

    #[test]
    fn optional_properties_branch() {
        let t = trie(
            &[
                ("{", 0),
                ("\"a\":", 1),
                ("\"b\":", 2),
                ("true", 3),
                ("false", 4),
                (",", 5),
                ("}", 6),
            ],
            7,
        );
        let mut m = matcher(
            json!({"type": "object", "properties": {
                "a": {"type": "boolean"},
                "b": {"type": "boolean"}
            }}),
            &t,
        );
        m.commit(&t, 0);
        let mask = m.mask(&t);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 1.0);
        assert_eq!(mask[6], 1.0);
        m.commit(&t, 1);
        m.commit(&t, 3);
        let mask = m.mask(&t);
        assert_eq!(mask[5], 1.0);
        assert_eq!(mask[6], 1.0);
        m.commit(&t, 5);
        m.commit(&t, 2);
        m.commit(&t, 4);
        m.commit(&t, 6);
        assert!(m.is_complete());
    }

    #[test]
    fn required_property_is_mandatory() {
        let t = trie(&[("{", 0), ("\"a\":", 1), ("true", 2), ("}", 3)], 4);
        let mut m = matcher(
            json!({"type": "object", "required": ["a"], "properties": {
                "a": {"type": "boolean"}
            }}),
            &t,
        );
        m.commit(&t, 0);
        let mask = m.mask(&t);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[3], 0.0);
        m.commit(&t, 1);
        m.commit(&t, 2);
        let mask = m.mask(&t);
        assert_eq!(mask[3], 1.0);
    }

    #[test]
    fn property_order_overrides_alphabetical_emission() {
        let t = trie(
            &[
                ("{\"b\":", 0),
                ("\"y\"", 1),
                (",\"a\":", 2),
                ("\"x\"", 3),
                ("}", 4),
            ],
            5,
        );
        let mut m = matcher(
            json!({"type": "object", "propertyOrder": ["b", "a"], "required": ["b", "a"],
                "properties": {"a": {"type": "string", "enum": ["x"]}, "b": {"type": "string", "enum": ["y"]}}}),
            &t,
        );
        let mask = m.mask(&t);
        assert_eq!(mask[0], 1.0, "propertyOrder forces the b property first");
        for (tok, &v) in mask.iter().enumerate().skip(1) {
            assert_eq!(v, 0.0, "token {tok} must not start the object");
        }
        for tok in [0, 1, 2, 3, 4] {
            m.commit(&t, tok);
        }
        assert!(m.is_complete());
    }

    #[test]
    fn any_of_accepts_either_branch() {
        let t = trie(&[("\"a\"", 0), ("\"b\"", 1)], 2);
        let m = matcher(
            json!({"anyOf": [
                {"type": "string", "enum": ["a"]},
                {"type": "string", "enum": ["b"]}
            ]}),
            &t,
        );
        let mask = m.mask(&t);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 1.0);
    }

    #[test]
    fn defs_reference_resolves() {
        let t = trie(&[("{\"a\":", 0), ("1", 1), ("}", 2), ("2", 3), ("]", 4)], 5);
        let mut m = matcher(
            json!({
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/n"}},
                "$defs": {"n": {"type": "integer"}}
            }),
            &t,
        );
        m.commit(&t, 0);
        let mask = m.mask(&t);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[3], 1.0, "any digit continues the integer");
        assert_eq!(mask[4], 0.0);
    }
}
