use features::*;

use rand::{Rng};
use std::rc::{Rc};
use std::sync::{Arc};

#[derive(Clone)]
pub struct ReplayEntry<F, Action, Res> {
  pub prev:     BeliefState<F>,
  pub action:   Action,
  pub res:      Option<Res>,
  pub next:     BeliefState<F>,
  pub terminal: bool,
  pub idx:      usize,
}

#[derive(Clone)]
pub struct ReplayFrame<F, Action, Res> {
  pub action:   Action,
  pub res:      Option<Res>,
  pub next_obs: Rc<F>,
  pub terminal: bool,
}

pub struct LinearReplayCache<F, Action, Res> {
  history_len:  usize,
  //frame_dim:    (usize, usize, usize),
  capacity:     usize,
  head:         usize,
  frames:       Vec<ReplayFrame<F, Action, Res>>,
}

impl<F, Action, Res> LinearReplayCache<F, Action, Res> where Action: Copy, Res: Copy {
  pub fn new(history_len: usize, /*frame_dim: (usize, usize, usize),*/ capacity: usize) -> Self {
    LinearReplayCache{
      history_len:  history_len,
      //frame_dim:    frame_dim,
      capacity:     capacity,
      head:         0,
      frames:       Vec::with_capacity(capacity),
    }
  }

  pub fn len(&self) -> usize {
    self.frames.len()
  }

  pub fn insert(&mut self, action: Action, res: Option<Res>, next_obs: Rc<F>, terminal: bool) {
    self.push(action, res, next_obs, terminal);
  }

  pub fn push(&mut self, action: Action, res: Option<Res>, next_obs: Rc<F>, terminal: bool) {
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
    let mut state = BeliefState::new(Some(self.history_len), /*self.frame_dim*/);
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
      if (first_valid_idx < last_idx && idx >= first_valid_idx && idx <= last_idx) ||
          (last_idx < first_valid_idx && (idx >= first_valid_idx || idx <= last_idx)) {
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

#[derive(Clone)]
pub struct SharedReplayEntry<State, Action, Res> {
  pub prev:     State,
  pub action:   Action,
  pub res:      Option<Res>,
  pub next:     State,
  pub terminal: bool,
  pub idx:      usize,
}

#[derive(Clone)]
pub struct SharedReplayFrame<Obs, Action, Res> {
  pub action:   Action,
  pub res:      Option<Res>,
  pub next_obs: Arc<Obs>,
  pub terminal: bool,
}

pub struct SharedLinearReplayCache<Obs, Action, Res> {
  history_len:  usize,
  //frame_dim:    (usize, usize, usize),
  capacity:     usize,
  head:         usize,
  frames:       Vec<SharedReplayFrame<Obs, Action, Res>>,
}

impl<F, Action, Res> SharedLinearReplayCache<F, Action, Res> where F: MultiObs, Action: Copy, Res: Copy {
  pub fn new(history_len: usize, /*frame_dim: (usize, usize, usize),*/ capacity: usize) -> Self {
    SharedLinearReplayCache{
      history_len:  history_len,
      //frame_dim:    frame_dim,
      capacity:     capacity,
      head:         0,
      frames:       Vec::with_capacity(capacity),
    }
  }

  pub fn len(&self) -> usize {
    self.frames.len()
  }

  pub fn insert(&mut self, action: Action, res: Option<Res>, next_obs: Arc<F>, terminal: bool) {
    self.push(action, res, next_obs, terminal);
  }

  pub fn push(&mut self, action: Action, res: Option<Res>, next_obs: Arc<F>, terminal: bool) {
    let new_frame = SharedReplayFrame{
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

  fn build_state<State>(&self, idx: usize) -> State where State: SharedMultiBeliefState<F> {
    //let mut state = BeliefState::new(Some(self.history_len), /*self.frame_dim*/);
    let mut state = State::_build(Some(self.history_len));
    /*if idx >= self.history_len - 1 {
      for offset in 0 .. self.history_len {
        state.push(self.frames[idx - (self.history_len - 1) + offset].next_obs.clone());
      }
    } else {*/
    for offset in 0 .. self.history_len {
      let offset_idx = (self.frames.len() + idx - (self.history_len - 1) + offset) % self.frames.len();
      state.append(self.frames[offset_idx].next_obs.clone());
    }
    //}
    state
  }

  pub fn sample<R, State>(&self, rng: &mut R) -> SharedReplayEntry<State, Action, Res> where R: Rng, State: SharedMultiBeliefState<F> {
    assert!(self.frames.len() >= self.history_len + 1);
    assert!(self.frames.len() <= self.capacity);
    let mut idx = 0;
    'outer_loop : loop {
      idx = rng.gen_range(0, self.frames.len());
      let last_idx = (self.frames.len() + self.head - 1) % self.frames.len();
      let first_valid_idx = (self.frames.len() + last_idx - (self.frames.len() - 1) + self.history_len) % self.frames.len();
      if (first_valid_idx < last_idx && idx >= first_valid_idx && idx <= last_idx) ||
          (last_idx < first_valid_idx && (idx >= first_valid_idx || idx <= last_idx)) {
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
    SharedReplayEntry{
      prev:     prev_state,
      action:   self.frames[idx].action,
      res:      self.frames[idx].res,
      next:     next_state,
      terminal: self.frames[idx].terminal,
      idx:      idx,
    }
  }
}

#[derive(Clone)]
pub struct SharedMultiReplayEntry<State, ActH, Action, Res> {
  pub prev_states:  Vec<State>,
  pub prev_act_h:   ActH,
  pub action:   Vec<Option<Action>>,
  pub act_idx:  Vec<Option<u32>>,
  pub res:      Vec<Option<Res>>,
  pub next_states:  Vec<State>,
  pub next_act_h:   ActH,
  pub terminal: bool,
  pub idx:      usize,
}

#[derive(Clone)]
pub struct SharedMultiReplayFrame<Obs, Action, Res> {
  pub action:   Vec<Option<Action>>,
  pub act_idx:  Vec<Option<u32>>,
  pub res:      Vec<Option<Res>>,
  pub next_obs: Vec<Arc<Obs>>,
  pub terminal: bool,
}

pub struct SharedLinearMultiReplayCache<Obs, Action, Res> {
  num_players:  usize,
  action_dim:   usize,
  history_len:  usize,
  capacity:     usize,
  head:         usize,
  frames:       Vec<SharedMultiReplayFrame<Obs, Action, Res>>,
}

impl<F, Action, Res> SharedLinearMultiReplayCache<F, Action, Res> where F: MultiObs, Action: Copy, Res: Copy {
  pub fn new(num_players: usize, action_dim: usize, history_len: usize, capacity: usize) -> Self {
    SharedLinearMultiReplayCache{
      num_players:  num_players,
      action_dim:   action_dim,
      history_len:  history_len,
      capacity:     capacity,
      head:         0,
      frames:       Vec::with_capacity(capacity),
    }
  }

  pub fn len(&self) -> usize {
    self.frames.len()
  }

  pub fn push(&mut self, action: Vec<Option<Action>>, act_idx: Vec<Option<u32>>, res: Vec<Option<Res>>, next_obs: Vec<Arc<F>>, terminal: bool) {
    let new_frame = SharedMultiReplayFrame{
      action:   action,
      act_idx:  act_idx,
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

  fn build_state<State>(&self, player_rank: usize, idx: usize) -> State where State: SharedMultiBeliefState<F> {
    let mut state = State::_build(Some(self.history_len));
    for offset in 0 .. self.history_len {
      let offset_idx = (self.frames.len() + idx - (self.history_len - 1) + offset) % self.frames.len();
      state.append(self.frames[offset_idx].next_obs[player_rank].clone());
    }
    state
  }

  fn build_action_history<ActH>(&self, idx: usize) -> ActH where ActH: SharedMultiActionHistory {
    let mut action_hist = ActH::_build(self.num_players, self.action_dim, Some(self.history_len));
    for offset in 0 .. self.history_len {
      let offset_idx = (self.frames.len() + idx - (self.history_len - 1) + offset) % self.frames.len();
      action_hist.append(self.frames[offset_idx].act_idx.clone());
    }
    action_hist
  }

  pub fn sample<R, State, ActH>(&self, rng: &mut R) -> SharedMultiReplayEntry<State, ActH, Action, Res> where R: Rng, State: SharedMultiBeliefState<F>, ActH: SharedMultiActionHistory {
    assert!(self.frames.len() >= self.history_len + 1);
    assert!(self.frames.len() <= self.capacity);
    let mut idx = 0;
    'outer_loop : loop {
      idx = rng.gen_range(0, self.frames.len());
      let last_idx = (self.frames.len() + self.head - 1) % self.frames.len();
      let first_valid_idx = (self.frames.len() + last_idx - (self.frames.len() - 1) + self.history_len) % self.frames.len();
      if (first_valid_idx < last_idx && idx >= first_valid_idx && idx <= last_idx) ||
          (last_idx < first_valid_idx && (idx >= first_valid_idx || idx <= last_idx)) {
        for offset in 0 .. self.history_len {
          let offset_idx = (self.frames.len() + idx - self.history_len + offset) % self.frames.len();
          if self.frames[offset_idx].terminal {
            continue 'outer_loop;
          }
        }
        break 'outer_loop;
      }
    }
    let mut prev_states = vec![];
    let mut next_states = vec![];
    for r in 0 .. self.num_players {
      let prev_state = self.build_state(r, idx - 1);
      prev_states.push(prev_state);
      let next_state = self.build_state(r, idx);
      next_states.push(next_state);
    }
    let prev_act_h = self.build_action_history(idx - 1);
    let next_act_h = self.build_action_history(idx);
    SharedMultiReplayEntry{
      prev_states:  prev_states,
      prev_act_h:   prev_act_h,
      action:   self.frames[idx].action.clone(),
      act_idx:  self.frames[idx].act_idx.clone(),
      res:      self.frames[idx].res.clone(),
      next_states:  next_states,
      next_act_h:   next_act_h,
      terminal: self.frames[idx].terminal,
      idx:      idx,
    }
  }
}
