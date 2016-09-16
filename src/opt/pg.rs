use env::{Env, DiscreteAction, EpisodeTraj};

use std::marker::{PhantomData};

pub struct EpisodeStepSample {
  pub obs:  Vec<f32>,
  pub act:  u32,
  pub res:  f32,
}

pub trait DiscretePolicy {
  type Env;

  fn eval(&mut self, env: &mut Self::Env, action_dist: &mut [f32]);
}

pub struct DiscretePolicyGrad<E, P> where E: Env, P: DiscretePolicy<Env=E> {
  _marker: PhantomData<(E, P)>,
}

impl<E, P> DiscretePolicyGrad<E, P> where E: Env, P: DiscretePolicy<Env=E> {
}
