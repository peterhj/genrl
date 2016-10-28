//use discrete::{DiscreteSampler32};

use operator::{DiffOperatorOutput};
use sharedmem::{RwSlice};

use bit_set::{BitSet};

use rand::{Rng};
use std::cell::{RefCell};
use std::fmt::{Debug};
use std::io::{Write};
use std::path::{Path};
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

impl<T> Discounted<T> where T: Copy {
  pub fn new(value: T, discount: T) -> Discounted<T> {
    Discounted{
      value:    value,
      discount: discount,
    }
  }
}

impl Response for Discounted<f32> {
  #[inline]
  fn lreduce(&mut self, prefix: Discounted<f32>) {
    self.value = prefix.value + prefix.discount * self.value;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    self.value * (1.0 - self.discount)
  }
}

#[derive(Clone, Copy, Debug)]
pub struct NormalizeDiscounted<T> where T: Copy {
  pub value:    T,
  pub discount: T,
  pub count:    usize,
}

impl<T> NormalizeDiscounted<T> where T: Copy {
  pub fn new(value: T, discount: T) -> NormalizeDiscounted<T> {
    NormalizeDiscounted{
      value:    value,
      discount: discount,
      count:    1,
    }
  }
}

impl Response for NormalizeDiscounted<f32> {
  #[inline]
  fn lreduce(&mut self, prefix: NormalizeDiscounted<f32>) {
    assert_eq!(1, prefix.count);
    self.value = prefix.value + prefix.discount * self.value;
    self.count += 1;
  }

  #[inline]
  fn as_scalar(&self) -> f32 {
    self.value * (1.0 - self.discount) / (1.0 - self.discount.powi(self.count as i32))
  }
}

pub trait Value: Copy + Debug {
  type Cfg: Copy;
  type Res: Response;

  fn from_res(res: Self::Res, cfg: Self::Cfg) -> Self where Self: Sized;
  fn from_scalar(scalar_value: f32, cfg: Self::Cfg) -> Self where Self: Sized;
  fn to_scalar(&self) -> f32;
  fn lreduce(&mut self, prefix: Self::Res);
}

#[derive(Clone, Copy, Debug)]
pub struct SumValue<T> where T: Copy {
  pub value:    T,
}

impl Value for SumValue<f32> {
  type Cfg = ();
  type Res = f32;

  fn from_res(res: f32, cfg: ()) -> SumValue<f32> {
    SumValue{
      value:    res,
    }
  }

  fn from_scalar(scalar_value: f32, cfg: ()) -> SumValue<f32> {
    SumValue{
      value:    scalar_value,
    }
  }

  fn to_scalar(&self) -> f32 {
    self.value
  }

  fn lreduce(&mut self, prefix: f32) {
    self.value = prefix + self.value;
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Discount<T>(pub T) where T: Copy;

#[derive(Clone, Copy, Debug)]
pub struct DiscountedValue<T> where T: Copy {
  pub value:    T,
  pub discount: T,
}

impl Value for DiscountedValue<f32> {
  type Cfg = Discount<f32>;
  type Res = f32;

  fn from_res(res: f32, cfg: Discount<f32>) -> DiscountedValue<f32> {
    DiscountedValue{
      value:    res,
      discount: cfg.0,
    }
  }

  fn from_scalar(scalar_value: f32, cfg: Discount<f32>) -> DiscountedValue<f32> {
    DiscountedValue{
      value:    scalar_value,
      discount: cfg.0,
    }
  }

  fn to_scalar(&self) -> f32 {
    self.value
  }

  fn lreduce(&mut self, prefix: f32) {
    self.value = prefix + self.discount * self.value;
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ClipDiscountedValue<T> where T: Copy {
  pub value:    T,
  pub discount: T,
}

impl Value for ClipDiscountedValue<f32> {
  type Cfg = Discount<f32>;
  type Res = f32;

  fn from_res(res: f32, cfg: Discount<f32>) -> ClipDiscountedValue<f32> {
    let clipped_res = if res > 0.0 {
      1.0
    } else if res < 0.0 {
      -1.0
    } else {
      0.0
    };
    ClipDiscountedValue{
      value:    clipped_res,
      discount: cfg.0,
    }
  }

  fn from_scalar(scalar_value: f32, cfg: Discount<f32>) -> ClipDiscountedValue<f32> {
    ClipDiscountedValue{
      value:    scalar_value,
      discount: cfg.0,
    }
  }

  fn to_scalar(&self) -> f32 {
    self.value
  }

  fn lreduce(&mut self, prefix: f32) {
    let clipped_prefix = if prefix > 0.0 {
      1.0
    } else if prefix < 0.0 {
      -1.0
    } else {
      0.0
    };
    self.value = clipped_prefix + self.discount * self.value;
  }
}

#[derive(Clone, Copy, Debug)]
pub struct NormDiscountedValue<T> where T: Copy {
  pub value:    T,
  pub discount: T,
}

impl Value for NormDiscountedValue<f32> {
  type Cfg = Discount<f32>;
  type Res = f32;

  fn from_res(res: f32, cfg: Discount<f32>) -> NormDiscountedValue<f32> {
    NormDiscountedValue{
      value:    res,
      discount: cfg.0,
    }
  }

  fn from_scalar(scalar_value: f32, cfg: Discount<f32>) -> NormDiscountedValue<f32> {
    NormDiscountedValue{
      value:    scalar_value,
      discount: cfg.0,
    }
  }

  fn to_scalar(&self) -> f32 {
    self.value * (1.0 - self.discount)
  }

  fn lreduce(&mut self, prefix: f32) {
    self.value = prefix + self.discount * self.value;
  }
}

pub trait Env: Default {
  type Init;
  type Action: Action;
  type Response: Response;

  /// Reset the environment according to the initial state distribution and
  /// other initial configuration.
  fn reset<R>(&self, init: &Self::Init, rng: &mut R) where R: Rng + Sized;

  /// Check if the environment is at a terminal state (no more legal actions).
  fn is_terminal(&self) -> bool;

  /// Check if an action is legal. Can be expensive if this involves simulating
  /// the action using `step`.
  fn is_legal_action(&self, action: &Self::Action) -> bool;

  /// Try to execute an action, returning an error if the action is illegal.
  fn step(&self, action: &Self::Action) -> Result<Option<Self::Response>, ()>;

  fn _save_png(&self, path: &Path) {}
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

/*pub trait EnvOpaqueRepr<Obs>: Env {
  fn extract_opaque_observable(&mut self, obs: &mut Obs);
}*/

pub trait EnvInputRepr<U: ?Sized>: Env {
  fn _shape3d(&self) -> (usize, usize, usize);
}

pub trait EnvRepr<T>: Env {
  #[deprecated] fn observable_len(&mut self) -> usize { self.observable_sz() }
  fn observable_sz(&self) -> usize;
  fn extract_observable(&self, obs: &mut [T]);
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
  //pub next_env: Rc<RefCell<E>>,
  pub next_env: Rc<E>,
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
  //pub init_env: Rc<RefCell<E>>,
  pub init_env: Rc<E>,
  pub steps:    Vec<EpisodeStep<E>>,
  pub suffixes: Vec<Option<E::Response>>,
  //final_value:  Option<E::Response>,
}

impl<E> Clone for Episode<E> where E: Env {
  fn clone(&self) -> Episode<E> {
    Episode{
      init_env: self.init_env.clone(),
      steps:    self.steps.clone(),
      suffixes: self.suffixes.clone(),
      //final_value:  self.final_value,
    }
  }
}

impl<E> Episode<E> where E: Env {
  pub fn new() -> Episode<E> {
    Episode{
      //init_env: Rc::new(RefCell::new(Default::default())),
      init_env: Rc::new(Default::default()),
      steps:    vec![],
      suffixes: vec![],
      //final_value:  None,
    }
  }

  pub fn horizon(&self) -> usize {
    self.steps.len()
  }

  pub fn value(&self) -> Option<f32> {
    self.suffixes[0].map(|r| r.as_scalar())
  }

  pub fn reset<R>(&mut self, init_cfg: &E::Init, rng: &mut R) where R: Rng + Sized {
    self.init_env.reset(init_cfg, rng);
    self.steps.clear();
    self.suffixes.clear();
  }

  pub fn terminated(&self) -> bool {
    match self.steps.len() {
      0   => self.init_env.is_terminal(),
      len => self.steps[len-1].next_env.is_terminal(),
    }
  }

  /*pub fn set_final_value(&mut self, v: E::Response) {
    self.final_value = Some(v);
  }*/

  pub fn _fill_suffixes(&mut self) {
    let horizon = self.steps.len();
    self.suffixes.clear();
    for k in 0 .. horizon {
      self.suffixes.push(self.steps[k].res);
    }
    /*let mut suffix = if self.terminated() {
      None
    } else if let Some(v) = self.final_value {
      Some(v)
    } else {
      None
    };*/
    let mut suffix = None;
    for k in (0 .. horizon).rev() {
      if suffix.is_none() {
        suffix = self.steps[k].res;
      } else if let Some(prefix) = self.steps[k].res {
        suffix.as_mut().unwrap().lreduce(prefix);
      }
      self.suffixes[k] = suffix;
    }
  }

  pub fn _value(&self) -> Option<f32> {
    if !self.suffixes.is_empty() {
      assert_eq!(self.steps.len(), self.suffixes.len());
      self.suffixes[0].map(|r| r.as_scalar())
    } else {
      None
    }
  }
}
