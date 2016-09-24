use env::{Env, EnvConvert, EnvRepr, Action, DiscreteAction, Response, HorizonAveraged};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::f32::consts::{PI};

#[derive(Clone, Copy)]
pub struct BanditAction {
  idx:  u32,
}

impl Action for BanditAction {
  fn dim() -> usize {
    10
  }
}

impl DiscreteAction for BanditAction {
  fn from_idx(idx: u32) -> BanditAction {
    assert!(idx < 10);
    BanditAction{idx: idx}
  }

  fn idx(&self) -> u32 {
    self.idx
  }
}

#[derive(Clone, Copy, Debug)]
pub struct BanditConfig {
}

impl Default for BanditConfig {
  fn default() -> BanditConfig {
    BanditConfig{
    }
  }
}

#[derive(Clone, Copy, Default)]
struct BanditState {
}

//#[derive(Default)]
pub struct BanditEnv {
  //cfg:      BanditConfig,
  //state:    BanditState,
  dist:     Range<u32>,
  rng:      Xorshiftplus128Rng,
}

impl Default for BanditEnv {
  fn default() -> BanditEnv {
    BanditEnv{
      dist:     Range::new(0, 10),
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
    }
  }
}

impl Env for BanditEnv {
  type Init     = BanditConfig;
  type Action   = BanditAction;
  type Response = HorizonAveraged<f32>;

  fn reset<R>(&mut self, init: &BanditConfig, rng: &mut R) where R: Rng + Sized {
    self.rng = Xorshiftplus128Rng::new(rng);
  }

  fn is_terminal(&mut self) -> bool {
    false
  }

  fn is_legal_action(&mut self, action: &BanditAction) -> bool {
    true
  }

  fn step(&mut self, action: &BanditAction) -> Result<Option<HorizonAveraged<f32>>, ()> {
    //self.dist.ind_sample(&mut self.rng);
    if action.idx == 7 {
      Ok(Some(HorizonAveraged{value: 1.0, horizon: 100}))
    } else {
      Ok(Some(HorizonAveraged{value: 0.0, horizon: 100}))
    }
  }
}

impl EnvConvert<BanditEnv> for BanditEnv {
  fn clone_from_env(&mut self, other: &BanditEnv) {
    /*self.cfg = other.cfg;
    self.total_mass = other.total_mass;
    self.pole_mass_length = other.pole_mass_length;
    self.state = other.state;*/
    self.dist = other.dist;
    self.rng = other.rng.clone();
  }
}

impl EnvRepr<f32> for BanditEnv {
  fn observable_len(&self) -> usize {
    10
  }

  fn extract_observable(&mut self, obs: &mut [f32]) {
    /*obs[0] = self.state.x;
    obs[1] = self.state.x_dot;
    obs[2] = self.state.theta;
    obs[3] = self.state.theta_dot;*/
    for i in 0 .. 10 {
      obs[i] = 0.0;
    }
    obs[7] = 1.0;
  }
}
