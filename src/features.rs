use env::{Env};

use operator::prelude::*;

use std::collections::{VecDeque};
use std::rc::{Rc};

pub trait EnvObsRepr<F>: Env {
  fn _obs_shape3d() -> (usize, usize, usize);
  fn observe(&self) -> F;
}

pub struct BeliefState<F> {
  pub history_len:  Option<usize>,
  pub frame_dim:    (usize, usize, usize),
  pub obs_reprs:    VecDeque<Rc<F>>,
}

impl<F> Clone for BeliefState<F> {
  fn clone(&self) -> Self {
    BeliefState{
      history_len:  self.history_len,
      frame_dim:    self.frame_dim,
      obs_reprs:    self.obs_reprs.clone(),
    }
  }
}

impl<F> BeliefState<F> {
  pub fn new(history_len: Option<usize>, frame_dim: (usize, usize, usize)) -> BeliefState<F> {
    BeliefState{
      history_len:  history_len,
      frame_dim:    frame_dim,
      obs_reprs:    VecDeque::new(),
    }
  }

  pub fn reset(&mut self) {
    self.obs_reprs.clear();
  }

  pub fn push(&mut self, obs: Rc<F>) {
    if let Some(cap) = self.history_len {
      assert!(self.obs_reprs.len() <= cap);
      if self.obs_reprs.len() == cap {
        let _ = self.obs_reprs.pop_front();
      }
    }
    self.obs_reprs.push_back(obs);
  }

  pub fn _shape3d(&self) -> (usize, usize, usize) {
    (self.frame_dim.0, self.frame_dim.1, self.obs_reprs.len() * self.frame_dim.2)
  }
}

impl<F> SampleExtractInput<[f32]> for BeliefState<F> where F: SampleExtractInput<[f32]> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let mut offset = 0;
    for obs in self.obs_reprs.iter() {
      match obs.extract_input(&mut output[offset .. ]) {
        Err(_) => return Err(()),
        Ok(count) => offset += count,
      }
    }
    Ok(offset)
  }
}
