use env::{Env, EnvRepr, EnvConvert, Discounted};

use rand::{Rng};
use std::cell::{Cell};

#[derive(Clone, Copy)]
pub struct DiscountedWrapConfig<E> where E: Env<Response=f32> {
  pub discount: f32,
  pub env_init: E::Init,
}

#[derive(Clone, Default)]
pub struct DiscountedWrapEnv<E> where E: Env<Response=f32> {
  pub discount: Cell<f32>,
  pub env:      E,
}

impl<E> Env for DiscountedWrapEnv<E> where E: Env<Response=f32> {
  //type Init = E::Init;
  type Init = DiscountedWrapConfig<E>;
  type Action = E::Action;
  type Response = Discounted<f32>;

  fn reset<R>(&self, init: &Self::Init, rng: &mut R) where R: Rng + Sized {
    self.discount.set(init.discount);
    self.env.reset(&init.env_init, rng);
  }

  fn is_terminal(&self) -> bool {
    self.env.is_terminal()
  }

  fn is_legal_action(&self, action: &Self::Action) -> bool {
    self.env.is_legal_action(action)
  }

  fn step(&self, action: &Self::Action) -> Result<Option<Self::Response>, ()> {
    self.env.step(action).map(|r| r.map(|r| Discounted::new(r, self.discount.get())))
  }
}

impl<E, T> EnvRepr<T> for DiscountedWrapEnv<E> where E: Env<Response=f32> + EnvRepr<T> {
  fn observable_sz(&self) -> usize {
    self.env.observable_sz()
  }

  fn extract_observable(&self, obs: &mut [T]) {
    self.env.extract_observable(obs);
  }
}

impl<E, Target> EnvConvert<DiscountedWrapEnv<Target>> for DiscountedWrapEnv<E> where E: Env<Response=f32> + EnvConvert<Target>, Target: Env<Response=f32> {
  fn clone_from_env(&mut self, other: &DiscountedWrapEnv<Target>) {
    self.discount.set(other.discount.get());
    self.env.clone_from_env(&other.env);
  }
}
