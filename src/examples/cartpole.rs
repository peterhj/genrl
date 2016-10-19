use env::{Env, EnvConvert, EnvInputRepr, EnvRepr, Action, DiscreteAction, Response, OnlineAveraged, Discounted};

use operator::prelude::*;

use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::f32::consts::{PI};
use std::cell::{Cell, RefCell};

// XXX: This version of the cart-pole is based on the one in rl-gym, which
// itself is based on Sutton's original C code:
// <https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c>

#[derive(Clone, Copy)]
pub enum CartpoleAction {
  Plus,
  Minus,
}

impl CartpoleAction {
  pub fn as_scalar(&self) -> f32 {
    match *self {
      CartpoleAction::Plus  => 1.0,
      CartpoleAction::Minus => -1.0,
    }
  }
}

impl Action for CartpoleAction {
  fn dim() -> usize {
    2
  }
}

impl DiscreteAction for CartpoleAction {
  fn from_idx(idx: u32) -> CartpoleAction {
    match idx {
      0 => CartpoleAction::Plus,
      1 => CartpoleAction::Minus,
      _ => unreachable!(),
    }
  }

  fn idx(&self) -> u32 {
    match *self {
      CartpoleAction::Plus  => 0,
      CartpoleAction::Minus => 1,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct CartpoleConfig {
  pub gravity:      f32,
  pub cart_mass:    f32,
  pub pole_mass:    f32,
  pub pole_length:  f32,
  pub force_mag:    f32,
  pub time_delta:   f32,
  pub x_thresh:     f32,
  pub theta_thresh: f32,
  pub horizon:      usize,
  //pub discount:     f32,
}

impl Default for CartpoleConfig {
  fn default() -> CartpoleConfig {
    CartpoleConfig{
      gravity:        9.8,
      cart_mass:      1.0,
      pole_mass:      0.1,
      pole_length:    1.0,
      force_mag:      10.0,
      time_delta:     0.02,
      x_thresh:       2.4,
      theta_thresh:   (12.0 / 360.0) * 2.0 * PI,
      horizon:        100,
      //discount:       0.99,
    }
  }
}

#[derive(Clone, Copy, Default)]
struct CartpoleState {
  total_mass:       f32,
  pole_mass_length: f32,
  x:            f32,
  x_dot:        f32,
  theta:        f32,
  theta_dot:    f32,
  terminated:   bool,
}

#[derive(Clone, Default)]
pub struct CartpoleEnv {
  cfg:      Cell<CartpoleConfig>,
  state:    RefCell<CartpoleState>,
}

impl Env for CartpoleEnv {
  type Init     = CartpoleConfig;
  type Action   = CartpoleAction;
  //type Response = Discounted<f32>;
  type Response = f32;

  fn reset<R>(&self, init: &CartpoleConfig, rng: &mut R) where R: Rng + Sized {
    self.cfg.set(*init);
    let cfg = self.cfg.get();
    let mut state = self.state.borrow_mut();
    state.total_mass = cfg.cart_mass + cfg.pole_mass;
    state.pole_mass_length = 0.5 * cfg.pole_mass * cfg.pole_length;
    let dist = Range::new(-0.05, 0.05);
    state.x = dist.ind_sample(rng);
    state.x_dot = dist.ind_sample(rng);
    state.theta = dist.ind_sample(rng);
    state.theta_dot = dist.ind_sample(rng);
    state.terminated = false;
  }

  fn is_terminal(&self) -> bool {
    //self.state.terminated || self.state.x.abs() > self.cfg.x_thresh || self.state.theta.abs() > self.cfg.theta_thresh
    false
  }

  fn is_legal_action(&self, action: &CartpoleAction) -> bool {
    true
  }

  //fn step(&self, action: &CartpoleAction) -> Result<Option<Discounted<f32>>, ()> {
  fn step(&self, action: &CartpoleAction) -> Result<Option<f32>, ()> {
    //if self.is_terminal() {
    let cfg = self.cfg.get();
    let mut state = self.state.borrow_mut();
    if state.terminated || state.x.abs() > cfg.x_thresh || state.theta.abs() > cfg.theta_thresh {
      state.terminated = true;
    }
    let force = cfg.force_mag * action.as_scalar();
    let cos_theta = state.theta.cos();
    let sin_theta = state.theta.sin();
    let tmp = (force + state.pole_mass_length * state.theta_dot * state.theta_dot * sin_theta) / state.total_mass;
    let theta_acc = (cfg.gravity * sin_theta - cos_theta * tmp) / (0.5 * cfg.pole_length * (4.0 / 3.0 - cfg.pole_mass * cos_theta / state.total_mass));
    let x_acc = tmp - state.pole_mass_length * theta_acc * cos_theta / state.total_mass;
    let mut next_state: CartpoleState = *state;
    next_state.x = state.x + cfg.time_delta * state.x_dot;
    next_state.x_dot = state.x_dot + cfg.time_delta * x_acc;
    next_state.theta = state.theta + cfg.time_delta * state.theta_dot;
    next_state.theta_dot = state.theta_dot + cfg.time_delta * theta_acc;
    next_state.terminated = state.terminated;
    *state = next_state;
    /*if state.terminated {
      Ok(Some(Discounted{value: 0.0, discount: cfg.discount}))
    } else {
      Ok(Some(Discounted{value: 1.0, discount: cfg.discount}))
    }*/
    if state.terminated {
      Ok(Some(0.0))
    } else {
      Ok(Some(1.0))
    }
  }
}

impl EnvInputRepr<[f32]> for CartpoleEnv {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (1, 1, 4)
  }
}

impl EnvConvert<CartpoleEnv> for CartpoleEnv {
  fn clone_from_env(&mut self, other: &CartpoleEnv) {
    unimplemented!();
    /*self.cfg = other.cfg;
    self.total_mass = other.total_mass;
    self.pole_mass_length = other.pole_mass_length;
    self.state = other.state;*/
  }
}

impl EnvRepr<f32> for CartpoleEnv {
  fn observable_sz(&self) -> usize {
    4
  }

  fn extract_observable(&self, obs: &mut [f32]) {
    let state = self.state.borrow();
    obs[0] = state.x;
    obs[1] = state.x_dot;
    obs[2] = state.theta;
    obs[3] = state.theta_dot;
  }
}

impl SampleExtractInput<[f32]> for CartpoleEnv {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let state = self.state.borrow();
    output[0] = state.x;
    output[1] = state.x_dot;
    output[2] = state.theta;
    output[3] = state.theta_dot;
    Ok(4)
  }
}
