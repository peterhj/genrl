extern crate libc;

use libc::*;

#[link(name = "genrl_kernels", kind = "static")]
extern "C" {
  pub fn genrl_volatile_add_f32(
      n: size_t,
      x: *const f32,
      y: *mut f32,
  );
  pub fn genrl_volatile_average_f32(
      n: size_t,
      alpha: f32,
      x: *const f32,
      y: *mut f32,
  );
}
