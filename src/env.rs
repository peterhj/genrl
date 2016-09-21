use bit_set::{BitSet};

pub trait Action {
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

pub trait Response: Copy {
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
    *self += prefix;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    *self
  }
}

#[derive(Clone, Copy)]
pub struct Averaged<T> where T: Copy {
  pub value:    T,
  pub horizon:  usize,
}

impl Response for Averaged<f32> {
  #[inline]
  fn lreduce(&mut self, prefix: Averaged<f32>) {
    assert_eq!(1, prefix.horizon);
    self.value = self.value + (prefix.value - self.value) / (self.horizon + 1) as f32;
    self.horizon += 1;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    self.value
  }
}

#[derive(Clone, Copy)]
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

pub trait Env: Clone + Default {
  type Init;
  type Action: Action;
  type Response: Response;

  /// Reset the environment according to the initial state distribution and
  /// other initial configuration.
  fn reset(&mut self, init: &Self::Init);

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

pub trait EnvRawRepr: Env {
  fn extract_raw_observable(&mut self, buf: &mut [u8]);
}

pub trait EnvOpaqueRepr<Obs>: Env {
  fn extract_opaque_observable(&mut self, obs: &mut Obs);
}

pub trait EnvRepr<T>: Env {
  fn extract_observable(&mut self, obs: &mut [T]);
}

pub trait EnvConvert: Env {
  type Target: Env;

  fn clone_from_env(&mut self, other: &Self::Target);

  fn from_env(other: &Self::Target) -> Self where Self: Sized {
    let mut env: Self = Default::default();
    env.clone_from_env(other);
    env
  }
}

pub struct EpisodeStep<E> where E: Env {
  pub action:   E::Action,
  pub res:      Option<E::Response>,
  pub next_env: E,
}

pub struct EpisodeTraj<E> where E: Env {
  pub init_env: E,
  pub steps:    Vec<EpisodeStep<E>>,
}
