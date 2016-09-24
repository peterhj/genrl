//use discrete::{DiscreteSampler32};

use operator::{DiffOperatorOutput};
use sharedmem::{RwSlice};

use bit_set::{BitSet};

use rand::{Rng};
use std::cell::{RefCell};
use std::fmt::{Debug};
use std::io::{Write};
use std::rc::{Rc};

pub trait Action: Clone {
  fn dim() -> usize;
}

impl Action for f32 {
  fn dim() -> usize {
    1
  }
}

pub trait DiscreteAction: Action + Copy {
  fn from_idx(idx: u32) -> Self where Self: Sized;
  fn idx(&self) -> u32;
}

pub trait Response: Copy + Debug {
  fn lreduce(&mut self, prefix: Self);
  fn as_scalar(&self) -> f32;
}

impl Response for bool {
  #[inline]
  fn lreduce(&mut self, _: bool) {
    // Do nothing.
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    match *self {
      false => 0.0,
      true  => 1.0,
    }
  }
}

impl Response for f32 {
  #[inline]
  fn lreduce(&mut self, prefix: f32) {
    //println!("DEBUG: lreduce: {:e} {:e}", prefix, *self);
    *self += prefix;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    *self
  }
}

#[derive(Clone, Copy, Debug)]
pub struct HorizonAveraged<T> where T: Copy {
  pub value:    T,
  pub horizon:  usize,
}

impl Response for HorizonAveraged<f32> {
  #[inline]
  fn lreduce(&mut self, prefix: HorizonAveraged<f32>) {
    assert_eq!(self.horizon, prefix.horizon);
    self.value += prefix.value;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    self.value / self.horizon as f32
  }
}

#[derive(Clone, Copy, Debug)]
pub struct OnlineAveraged<T> where T: Copy {
  pub value:    T,
  pub count:    usize,
}

impl<T> OnlineAveraged<T> where T: Copy {
  pub fn new(value: T) -> OnlineAveraged<T> {
    OnlineAveraged{
      value:    value,
      count:    1,
    }
  }
}

impl Response for OnlineAveraged<f32> {
  #[inline]
  fn lreduce(&mut self, prefix: OnlineAveraged<f32>) {
    assert_eq!(1, prefix.count);
    self.value = self.value + (prefix.value - self.value) / (self.count + 1) as f32;
    self.count += 1;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    self.value
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Discounted<T> where T: Copy {
  pub value:    T,
  pub discount: T,
}

impl Response for Discounted<f32> {
  #[inline]
  fn lreduce(&mut self, prefix: Discounted<f32>) {
    self.value = prefix.value + prefix.discount * self.value;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    self.value
  }
}

pub trait Env: Default {
  type Init;
  type Action: Action;
  type Response: Response;

  /// Reset the environment according to the initial state distribution and
  /// other initial configuration.
  fn reset<R>(&mut self, init: &Self::Init, rng: &mut R) where R: Rng + Sized;

  /// Check if the environment is at a terminal state (no more legal actions).
  fn is_terminal(&mut self) -> bool;

  /// Check if an action is legal. Can be expensive if this involves simulating
  /// the action using `step`.
  fn is_legal_action(&mut self, action: &Self::Action) -> bool;

  /// Try to execute an action, returning an error if the action is illegal.
  fn step(&mut self, action: &Self::Action) -> Result<Option<Self::Response>, ()>;
}

pub trait DiscreteEnv: Env where Self::Action: DiscreteAction {
  /// Extracting discrete actions.
  fn extract_all_actions_buf(&mut self, actions_buf: &mut Vec<Self::Action>);
  fn extract_legal_actions_buf(&mut self, actions_buf: &mut Vec<Self::Action>);
  fn extract_legal_actions_set(&mut self, actions_set: &mut BitSet);
}

pub trait EnvSerialize: Env {
  fn serialize(&mut self, buf: &mut Write);
}

pub trait EnvOpaqueRepr<Obs>: Env {
  fn extract_opaque_observable(&mut self, obs: &mut Obs);
}

pub trait EnvRepr<T>: Env {
  fn observable_len(&self) -> usize;
  fn extract_observable(&mut self, obs: &mut [T]);
}

pub trait EnvConvert<Target>: Env where Target: Env {
  fn clone_from_env(&mut self, other: &Target);

  fn from_env(other: &Target) -> Self where Self: Sized {
    let mut env: Self = Default::default();
    env.clone_from_env(other);
    env
  }
}

//#[derive(Clone)]
pub struct EpisodeStep<E> where E: Env {
  pub action:   E::Action,
  pub res:      Option<E::Response>,
  pub next_env: Rc<RefCell<E>>,
}

impl<E> Clone for EpisodeStep<E> where E: Env {
  fn clone(&self) -> EpisodeStep<E> {
    EpisodeStep{
      action:   self.action.clone(),
      res:      self.res,
      next_env: self.next_env.clone(),
    }
  }
}

//#[derive(Clone)]
pub struct Episode<E> where E: Env {
  pub init_env: Rc<RefCell<E>>,
  pub steps:    Vec<EpisodeStep<E>>,
  pub suffixes: Vec<Option<E::Response>>,
}

impl<E> Clone for Episode<E> where E: Env {
  fn clone(&self) -> Episode<E> {
    Episode{
      init_env: self.init_env.clone(),
      steps:    self.steps.clone(),
      suffixes: self.suffixes.clone(),
    }
  }
}

impl<E> Episode<E> where E: Env {
  pub fn new() -> Episode<E> {
    Episode{
      init_env: Rc::new(RefCell::new(Default::default())),
      steps:    vec![],
      suffixes: vec![],
    }
  }

  pub fn reset<R>(&mut self, init_cfg: &E::Init, rng: &mut R) where R: Rng + Sized {
    self.init_env.borrow_mut().reset(init_cfg, rng);
    self.steps.clear();
    self.suffixes.clear();
  }

  pub fn terminated(&self) -> bool {
    match self.steps.len() {
      0   => self.init_env.borrow_mut().is_terminal(),
      len => self.steps[len-1].next_env.borrow_mut().is_terminal(),
    }
  }

  pub fn fill_suffixes(&mut self) {
    let horizon = self.steps.len();
    self.suffixes.clear();
    for k in 0 .. horizon {
      self.suffixes.push(self.steps[k].res);
    }
    let mut suffix = self.steps[horizon-1].res;
    for k in (0 .. horizon-1).rev() {
      if suffix.is_none() {
        suffix = self.steps[k].res;
      } else if let Some(prefix) = self.steps[k].res {
        suffix.as_mut().unwrap().lreduce(prefix);
      }
      self.suffixes[k] = suffix;
    }
  }

  pub fn value(&self) -> Option<f32> {
    if !self.suffixes.is_empty() {
      assert_eq!(self.steps.len(), self.suffixes.len());
      self.suffixes[0].map(|r| r.as_scalar())
    } else {
      None
    }
  }
}
