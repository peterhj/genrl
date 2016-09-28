use env::{Env};

pub struct DiscountedWrapperEnv<E> where E: Env<Response=f32> {
  inner:    E,
}
