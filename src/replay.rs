use env::{Env};

use rand::{Rng};

pub struct ReplayEntry<E> where E: Env {
  pub orig_env: E,
  pub action:   E::Action,
  pub res:      E::Response,
  pub next_env: E,
}

pub struct ReplayCache<E> where E: Env {
  capacity: usize,
  head:     usize,
  entries:  Vec<ReplayEntry<E>>,
}

impl<E> ReplayCache<E> where E: Env {
  pub fn new(capacity: usize) -> ReplayCache<E> {
    ReplayCache{
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

  pub fn insert(&mut self, orig_env: E, action: E::Action, res: E::Response, next_env: E) {
    let entry = ReplayEntry{
      orig_env: orig_env,
      action:   action,
      res:      res,
      next_env: next_env,
    };
    if self.entries.len() < self.capacity {
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
