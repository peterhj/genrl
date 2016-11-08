use env::{Env};

use rand::{Rng};
use std::rc::{Rc};

#[derive(Clone)]
pub struct ReplayEntry<E> where E: Env + Clone {
  pub prev_env: Rc<E>,
  pub action:   E::Action,
  pub res:      Option<E::Response>,
  pub next_env: Rc<E>,
}

pub struct LinearReplayCache<E> where E: Env + Clone {
  capacity: usize,
  head:     usize,
  entries:  Vec<ReplayEntry<E>>,
}

impl<E> LinearReplayCache<E> where E: Env + Clone {
  pub fn new(capacity: usize) -> LinearReplayCache<E> {
    LinearReplayCache{
      capacity: capacity,
      head:     0,
      entries:  Vec::with_capacity(capacity),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.entries.len() == 0
  }

  pub fn is_full(&self) -> bool {
    self.entries.len() >= self.capacity
  }

  pub fn insert(&mut self, prev_env: Rc<E>, action: E::Action, res: Option<E::Response>, next_env: Rc<E>) {
    let entry = ReplayEntry{
      prev_env: prev_env,
      action:   action,
      res:      res,
      next_env: next_env,
    };
    if !self.is_full() {
      self.entries.push(entry);
    } else {
      self.entries[self.head] = entry;
    }
    self.head = (self.head + 1) % self.capacity;
  }

  pub fn sample<R>(&self, rng: &mut R) -> &ReplayEntry<E> where R: Rng {
    assert!(!self.is_empty());
    let idx = rng.gen_range(0, self.entries.len());
    &self.entries[idx]
  }
}
