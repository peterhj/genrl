use env::{Env, Episode};
use replay::{ReplayCache};

use std::marker::{PhantomData};

pub trait DetPolicy {
  type Env: Env;

  fn eval(&mut self, env: &mut Self::Env) -> Option<<Self::Env as Env>::Action>;
  fn accum_diff_param(&mut self, env: &mut Self::Env, action: &<Self::Env as Env>::Action, action_weights: &[f32]);
  fn step_grad(&mut self, step_size: f32);
}

pub trait DetActionValueFunc {
  type Env: Env;

  fn eval(&mut self, env: &mut Self::Env, action: &<Self::Env as Env>::Action) -> f32;
  fn diff_action(&mut self, env: &mut Self::Env, action: &<Self::Env as Env>::Action, action_weights: &mut [f32]);
  fn accum_diff_param(&mut self, env: &mut Self::Env, action: &<Self::Env as Env>::Action);
  fn step_grad(&mut self, step_size: f32);
}

pub struct DetPolicyGradCfg {
  pub clip_value:   Option<f32>,
}

pub struct DetPolicyGrad<E, P, Q> where E: Env, P: DetPolicy<Env=E>, Q: DetActionValueFunc<Env=E> {
  replay_cache: ReplayCache<E>,
  _marker: PhantomData<(E, P, Q)>,
}

impl<E, P, Q> DetPolicyGrad<E, P, Q> where E: Env, P: DetPolicy<Env=E>, Q: DetActionValueFunc<Env=E> {
  pub fn step(&mut self) {
  }
}
