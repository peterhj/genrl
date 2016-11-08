use features::{BeliefState};

use rand::{Rng};
use std::rc::{Rc};

#[derive(Clone)]
pub struct ReplayFrame<F, Action, Res> {
  pub action:   Action,
  pub res:      Option<Res>,
  pub next_obs: Rc<F>,
  pub terminal: bool,
}

#[derive(Clone)]
pub struct ReplayEntry<F, Action, Res> {
  pub prev:     BeliefState<F>,
  pub action:   Action,
  pub res:      Option<Res>,
  pub next:     BeliefState<F>,
  pub terminal: bool,
  pub idx:      usize,
}

pub struct LinearReplayCache<F, Action, Res> {
  history_len:  usize,
  frame_dim:    (usize, usize, usize),
  capacity:     usize,
  head:         usize,
  frames:       Vec<ReplayFrame<F, Action, Res>>,
}

impl<F, Action, Res> LinearReplayCache<F, Action, Res> where Action: Copy, Res: Copy {
  pub fn new(history_len: usize, frame_dim: (usize, usize, usize), capacity: usize) -> Self {
    LinearReplayCache{
      history_len:  history_len,
      frame_dim:    frame_dim,
      capacity:     capacity,
      head:         0,
      frames:       Vec::with_capacity(capacity),
    }
  }

  pub fn insert(&mut self, action: Action, res: Option<Res>, next_obs: Rc<F>, terminal: bool) {
    let new_frame = ReplayFrame{
      action:   action,
      res:      res,
      next_obs: next_obs,
      terminal: terminal,
    };
    assert!(self.head < self.capacity);
    if self.head >= self.frames.len() {
      self.frames.push(new_frame);
    } else {
      self.frames[self.head] = new_frame;
    }
    assert!(self.frames.len() <= self.capacity);
    self.head = (self.head + 1) % self.capacity;
  }

  fn build_state(&self, idx: usize) -> BeliefState<F> {
    let mut state = BeliefState::new(Some(self.history_len), self.frame_dim);
    /*if idx >= self.history_len - 1 {
      for offset in 0 .. self.history_len {
        state.push(self.frames[idx - (self.history_len - 1) + offset].next_obs.clone());
      }
    } else {*/
    for offset in 0 .. self.history_len {
      let offset_idx = (self.frames.len() + idx - (self.history_len - 1) + offset) % self.frames.len();
      state.push(self.frames[offset_idx].next_obs.clone());
    }
    //}
    state
  }

  pub fn sample<R>(&self, rng: &mut R) -> ReplayEntry<F, Action, Res> where R: Rng {
    assert!(self.frames.len() >= self.history_len + 1);
    assert!(self.frames.len() <= self.capacity);
    let mut idx = 0;
    'outer_loop : loop {
      idx = rng.gen_range(0, self.frames.len());
      let last_idx = (self.frames.len() + self.head - 1) % self.frames.len();
      let first_valid_idx = (self.frames.len() + last_idx - (self.frames.len() - 1) + self.history_len) % self.frames.len();
      if idx >= first_valid_idx || idx <= last_idx {
        for offset in 0 .. self.history_len {
          let offset_idx = (self.frames.len() + idx - self.history_len + offset) % self.frames.len();
          if self.frames[offset_idx].terminal {
            continue 'outer_loop;
          }
        }
        break 'outer_loop;
      }
      /*// FIXME(20161029): this index is biased when the replay memory is full,
      // but ignore it for now because the effect should be tiny.
      idx = rng.gen_range(self.history_len, self.frames.len());
      if idx >= self.head && idx < self.head + self.history_len {
        continue;
      }
      for offset in 0 .. self.history_len {
        if self.frames[idx - self.history_len + offset].terminal {
          continue 'outer_loop;
        }
      }
      break;*/
    }
    let prev_state = self.build_state(idx - 1);
    let next_state = self.build_state(idx);
    ReplayEntry{
      prev:     prev_state,
      action:   self.frames[idx].action,
      res:      self.frames[idx].res,
      next:     next_state,
      terminal: self.frames[idx].terminal,
      idx:      idx,
    }
  }
}
