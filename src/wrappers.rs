use env::{Env, EnvRepr, EnvConvert, Discounted};

use rand::{Rng};

#[derive(Clone, Copy)]
pub struct DiscountedWrapConfig<E> where E: Env<Response=f32> {
  pub discount: f32,
  pub env_init: E::Init,
}

#[derive(Clone, Default)]
pub struct DiscountedWrapEnv<E> where E: Env<Response=f32> {
  pub discount: f32,
  pub env:      E,
}

impl<E> Env for DiscountedWrapEnv<E> where E: Env<Response=f32> {
  //type Init = E::Init;
  type Init = DiscountedWrapConfig<E>;
  type Action = E::Action;
  type Response = Discounted<f32>;

  fn reset<R>(&mut self, init: &Self::Init, rng: &mut R) where R: Rng + Sized {
    self.discount = init.discount;
    self.env.reset(&init.env_init, rng);
  }

  fn is_terminal(&mut self) -> bool {
    self.env.is_terminal()
  }

  fn is_legal_action(&mut self, action: &Self::Action) -> bool {
    self.env.is_legal_action(action)
  }

  fn step(&mut self, action: &Self::Action) -> Result<Option<Self::Response>, ()> {
    self.env.step(action).map(|r| r.map(|r| Discounted::new(r, self.discount)))
  }
}

impl<E, T> EnvRepr<T> for DiscountedWrapEnv<E> where E: Env<Response=f32> + EnvRepr<T> {
  fn observable_sz(&mut self) -> usize {
    self.env.observable_sz()
  }

  fn extract_observable(&mut self, obs: &mut [T]) {
    self.env.extract_observable(obs);
  }
}

impl<E, Target> EnvConvert<DiscountedWrapEnv<Target>> for DiscountedWrapEnv<E> where E: Env<Response=f32> + EnvConvert<Target>, Target: Env<Response=f32> {
  fn clone_from_env(&mut self, other: &DiscountedWrapEnv<Target>) {
    self.discount = other.discount;
    self.env.clone_from_env(&other.env);
  }
}
