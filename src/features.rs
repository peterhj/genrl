use env::{Env, MultiEnv};

use densearray::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::collections::{VecDeque};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait EnvObsRepr<F>: Env {
  fn _obs_shape3d() -> (usize, usize, usize) { unimplemented!(); }
  fn observe(&self, rng: &mut Xorshiftplus128Rng) -> F;
}

pub trait EnvObsBuf<F>: Env {
  fn observe_buf(&self, rng: &mut Xorshiftplus128Rng, obs: &F);
}

pub trait MultiEnvObserve<Obs>: MultiEnv {
  fn observe<R>(&self, observer_rank: usize, rng: &mut R) -> Obs where R: Rng + Sized;
}

pub trait MultiObs {
  fn observer_rank(&self) -> usize;
  fn num_observers(&self) -> usize;
}

pub trait MultiBeliefState<Obs>: Clone where Obs: MultiObs {
  fn depth(&self) -> usize;
  fn reset(&mut self);
  fn append(&mut self, obs: Rc<Obs>);
}

pub trait SharedMultiBeliefState<Obs>: Clone where Obs: MultiObs {
  fn _build(max_depth: Option<usize>) -> Self where Self: Sized;
  fn depth(&self) -> usize;
  fn reset(&mut self);
  fn append(&mut self, obs: Arc<Obs>);
}

pub trait SharedMultiActionHistory: Clone {
  fn _build(num_players: usize, action_dim: usize, max_depth: Option<usize>) -> Self where Self: Sized;
  fn depth(&self) -> usize;
  fn reset(&mut self);
  fn append(&mut self, multi_act_idx: Vec<Option<u32>>);
}

impl SharedMultiActionHistory for () {
  fn _build(_num_players: usize, _action_dim: usize, _max_depth: Option<usize>) -> Self {
    ()
  }

  fn depth(&self) -> usize {
    0
  }

  fn reset(&mut self) {
  }

  fn append(&mut self, _multi_act_idx: Vec<Option<u32>>) {
  }
}

pub struct BeliefState<F> {
  pub history_len:  Option<usize>,
  pub obs_reprs:    VecDeque<Rc<F>>,
}

impl<F> Clone for BeliefState<F> {
  fn clone(&self) -> Self {
    BeliefState{
      history_len:  self.history_len,
      obs_reprs:    self.obs_reprs.clone(),
    }
  }
}

impl<F> BeliefState<F> {
  pub fn new(history_len: Option<usize>) -> BeliefState<F> {
    BeliefState{
      history_len:  history_len,
      obs_reprs:    VecDeque::new(),
    }
  }

  pub fn len(&self) -> usize {
    self.obs_reprs.len()
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
}

impl<F> Extract<[u8]> for BeliefState<F> where F: Extract<[u8]> {
  fn extract(&self, output: &mut [u8]) -> Result<usize, ()> {
    let mut offset = 0;
    for obs in self.obs_reprs.iter() {
      match obs.extract(&mut output[offset .. ]) {
        Err(_)    => return Err(()),
        Ok(count) => offset += count,
      }
    }
    Ok(offset)
  }
}

pub struct SharedBeliefState<F> {
  pub history_len:  Option<usize>,
  pub obs_reprs:    VecDeque<Arc<F>>,
}

impl<F> Clone for SharedBeliefState<F> {
  fn clone(&self) -> Self {
    SharedBeliefState{
      history_len:  self.history_len,
      obs_reprs:    self.obs_reprs.clone(),
    }
  }
}

impl<F> SharedBeliefState<F> {
  pub fn new(history_len: Option<usize>) -> SharedBeliefState<F> {
    SharedBeliefState{
      history_len:  history_len,
      obs_reprs:    VecDeque::new(),
    }
  }

  pub fn len(&self) -> usize {
    self.depth()
  }

  pub fn depth(&self) -> usize {
    self.obs_reprs.len()
  }

  pub fn reset(&mut self) {
    self.obs_reprs.clear();
  }

  pub fn push(&mut self, obs: Arc<F>) {
    self.append(obs);
  }

  pub fn append(&mut self, obs: Arc<F>) {
    if let Some(cap) = self.history_len {
      assert!(self.obs_reprs.len() <= cap);
      if self.obs_reprs.len() == cap {
        let _ = self.obs_reprs.pop_front();
      }
    }
    self.obs_reprs.push_back(obs);
  }
}

impl<F> Extract<[u8]> for SharedBeliefState<F> where F: Extract<[u8]> {
  fn extract(&self, output: &mut [u8]) -> Result<usize, ()> {
    let mut offset = 0;
    for obs in self.obs_reprs.iter() {
      match obs.extract(&mut output[offset .. ]) {
        Err(_)    => return Err(()),
        Ok(count) => offset += count,
      }
    }
    Ok(offset)
  }
}

/*pub struct BeliefState<F> {
  pub history_len:  Option<usize>,
  //pub frame_dim:    (usize, usize, usize),
  //pub obs_reprs:    VecDeque<Rc<F>>,
  pub obs_reprs:    Vec<Rc<F>>,
  frame_counter:    usize,
  frame_length:     usize,
}

impl<F> Clone for BeliefState<F> {
  fn clone(&self) -> Self {
    BeliefState{
      history_len:  self.history_len,
      //frame_dim:    self.frame_dim,
      obs_reprs:    self.obs_reprs.clone(),
      frame_counter:    self.frame_counter,
      frame_length:     self.frame_length,
    }
  }
}

impl<F> BeliefState<F> {
  pub fn new(history_len: Option<usize>, /*frame_dim: (usize, usize, usize)*/) -> BeliefState<F> {
    BeliefState{
      history_len:  history_len,
      //frame_dim:    frame_dim,
      //obs_reprs:    VecDeque::new(),
      obs_reprs:    Vec::with_capacity(history_len.unwrap_or(4)),
      frame_counter:    0,
      frame_length:     0,
    }
  }

  pub fn reset(&mut self) {
    self.obs_reprs.clear();
    self.frame_counter = 0;
    self.frame_length = 0;
  }

  pub fn push(&mut self, obs: Rc<F>) {
    if let Some(cap) = self.history_len {
      assert!(self.obs_reprs.len() <= cap);
      /*if self.obs_reprs.len() == cap {
        let _ = self.obs_reprs.pop_front();
      }*/
      if self.obs_reprs.len() < cap {
        self.obs_reprs.push(obs);
        self.frame_counter += 1;
        self.frame_length += 1;
      } else {
        self.obs_reprs[self.frame_counter] = obs;
        self.frame_counter += 1;
      }
      if self.frame_counter < cap {
      } else if self.frame_counter == cap {
        self.frame_counter = 0;
      } else {
        unreachable!();
      }
      assert!(self.frame_length <= cap);
    } else {
      self.obs_reprs.push(obs);
      self.frame_counter += 1;
      self.frame_length += 1;
    }
    //self.obs_reprs.push_back(obs);
  }

  /*pub fn _shape3d(&self) -> (usize, usize, usize) {
    (self.frame_dim.0, self.frame_dim.1, self.obs_reprs.len() * self.frame_dim.2)
  }*/
}

impl<F> Extract<[u8]> for BeliefState<F> where F: Extract<[u8]> {
  fn extract(&self, output: &mut [u8]) -> Result<usize, ()> {
    let mut offset = 0;
    /*for obs in self.obs_reprs.iter() {
      match obs.extract_input(&mut output[offset .. ]) {
        Err(_) => return Err(()),
        Ok(count) => offset += count,
      }
    }*/
    for frame_idx in 0 .. self.frame_length {
      let frame_offset =
          if let Some(cap) = self.history_len {
            (self.frame_counter + cap - self.frame_length + 1) % cap
          } else {
            frame_idx
          };
      match self.obs_reprs[frame_offset].extract(&mut output[offset .. ]) {
        Err(_) => panic!(),
        Ok(count) => offset += count,
      }
    }
    Ok(offset)
  }
}

impl<F> SampleExtractInput<[u8]> for BeliefState<F> where F: SampleExtractInput<[u8]> {
  fn extract_input(&self, output: &mut [u8]) -> Result<usize, ()> {
    let mut offset = 0;
    /*for obs in self.obs_reprs.iter() {
      match obs.extract_input(&mut output[offset .. ]) {
        Err(_) => return Err(()),
        Ok(count) => offset += count,
      }
    }*/
    for frame_idx in 0 .. self.frame_length {
      let frame_offset =
          if let Some(cap) = self.history_len {
            (self.frame_counter + cap - self.frame_length + 1) % cap
          } else {
            frame_idx
          };
      match self.obs_reprs[frame_offset].extract_input(&mut output[offset .. ]) {
        Err(_) => panic!(),
        Ok(count) => offset += count,
      }
    }
    Ok(offset)
  }
}

impl<F> SampleExtractInput<[f32]> for BeliefState<F> where F: SampleExtractInput<[f32]> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    // FIXME(20170126)
    unimplemented!();
    /*let mut offset = 0;
    for obs in self.obs_reprs.iter() {
      match obs.extract_input(&mut output[offset .. ]) {
        Err(_) => return Err(()),
        Ok(count) => offset += count,
      }
    }
    Ok(offset)*/
  }
}

//impl<F, Shape> SampleInputShape<Shape> for BeliefState<F> where F: SampleInputShape<Shape>, Shape: PartialEq + Eq {
impl<F> SampleInputShape<(usize, usize, usize)> for BeliefState<F> where F: SampleInputShape<(usize, usize, usize)> {
  fn input_shape(&self) -> Option<(usize, usize, usize)> {
    /*let mut shape = None;
    for obs in self.obs_reprs.iter() {
      let obs_shape = match obs.input_shape() {
        None => continue,
        Some(shape) => shape,
      };
      match shape {
        None => shape = Some(obs_shape),
        Some(ref prev_shape) => if prev_shape != &obs_shape { panic!(); },
      }
    }
    shape.map(|(w, h, c)| (w, h, c * self.obs_reprs.len()))*/

    /*let (w, h, c) = self.frame_dim;
    Some((w, h, c * self.frame_length))*/
    unimplemented!();
  }
}*/
