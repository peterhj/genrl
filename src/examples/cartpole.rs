use env::{Env, EnvConvert, EnvRepr, Action, DiscreteAction, Response, OnlineAveraged, Discounted};

use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::f32::consts::{PI};

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
  pub discount:     f32,
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
      discount:       0.99,
    }
  }
}

#[derive(Clone, Copy, Default)]
struct CartpoleState {
  x:            f32,
  x_dot:        f32,
  theta:        f32,
  theta_dot:    f32,
  terminated:   bool,
}

#[derive(Clone, Default)]
pub struct CartpoleEnv {
  cfg:              CartpoleConfig,
  total_mass:       f32,
  pole_mass_length: f32,
  state:            CartpoleState,
}

impl Env for CartpoleEnv {
  type Init     = CartpoleConfig;
  type Action   = CartpoleAction;
  type Response = OnlineAveraged<f32>;

  fn reset<R>(&mut self, init: &CartpoleConfig, rng: &mut R) where R: Rng + Sized {
    self.cfg = *init;
    self.total_mass = self.cfg.cart_mass + self.cfg.pole_mass;
    self.pole_mass_length = 0.5 * self.cfg.pole_mass * self.cfg.pole_length;
    let dist = Range::new(-0.05, 0.05);
    self.state.x = dist.ind_sample(rng);
    self.state.x_dot = dist.ind_sample(rng);
    self.state.theta = dist.ind_sample(rng);
    self.state.theta_dot = dist.ind_sample(rng);
    self.state.terminated = false;
  }

  fn is_terminal(&mut self) -> bool {
    //self.state.terminated || self.state.x.abs() > self.cfg.x_thresh || self.state.theta.abs() > self.cfg.theta_thresh
    false
  }

  fn is_legal_action(&mut self, action: &CartpoleAction) -> bool {
    true
  }

  fn step(&mut self, action: &CartpoleAction) -> Result<Option<OnlineAveraged<f32>>, ()> {
    //if self.is_terminal() {
    if self.state.terminated || self.state.x.abs() > self.cfg.x_thresh || self.state.theta.abs() > self.cfg.theta_thresh {
      self.state.terminated = true;
    }
    let force = self.cfg.force_mag * action.as_scalar();
    let cos_theta = self.state.theta.cos();
    let sin_theta = self.state.theta.sin();
    let tmp = (force + self.pole_mass_length * self.state.theta_dot * self.state.theta_dot * sin_theta) / self.total_mass;
    let theta_acc = (self.cfg.gravity * sin_theta - cos_theta * tmp) / (0.5 * self.cfg.pole_length * (4.0 / 3.0 - self.cfg.pole_mass * cos_theta / self.total_mass));
    let x_acc = tmp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;
    let mut next_state: CartpoleState = Default::default();
    next_state.x = self.state.x + self.cfg.time_delta * self.state.x_dot;
    next_state.x_dot = self.state.x_dot + self.cfg.time_delta * x_acc;
    next_state.theta = self.state.theta + self.cfg.time_delta * self.state.theta_dot;
    next_state.theta_dot = self.state.theta_dot + self.cfg.time_delta * theta_acc;
    next_state.terminated = self.state.terminated;
    self.state = next_state;
    /*if self.state.terminated {
      Ok(Some(Discounted{value: 0.0, discount: self.cfg.discount}))
    } else {
      Ok(Some(Discounted{value: 1.0, discount: self.cfg.discount}))
    }*/
    if self.state.terminated {
      Ok(Some(OnlineAveraged::new(0.0)))
    } else {
      Ok(Some(OnlineAveraged::new(1.0)))
    }
  }
}

impl EnvConvert<CartpoleEnv> for CartpoleEnv {
  fn clone_from_env(&mut self, other: &CartpoleEnv) {
    self.cfg = other.cfg;
    self.total_mass = other.total_mass;
    self.pole_mass_length = other.pole_mass_length;
    self.state = other.state;
  }
}

impl EnvRepr<f32> for CartpoleEnv {
  fn observable_len(&self) -> usize {
    4
  }

  fn extract_observable(&mut self, obs: &mut [f32]) {
    obs[0] = self.state.x;
    obs[1] = self.state.x_dot;
    obs[2] = self.state.theta;
    obs[3] = self.state.theta_dot;
  }
}
