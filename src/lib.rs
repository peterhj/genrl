#![feature(core_intrinsics)]
#![feature(specialization)]

extern crate genrl_kernels;

extern crate densearray;
extern crate float;
extern crate iter_utils;
#[macro_use] extern crate matches;
//extern crate operator;
extern crate rng;
extern crate sharedmem;
extern crate stopwatch;

extern crate bit_set;
extern crate rand;
extern crate rustc_serialize;

pub mod discrete;
pub mod env;
//pub mod examples;
pub mod features;
pub mod kernels;
pub mod monitor;
//pub mod opt;
pub mod prelude;
pub mod replay;
//pub mod wrappers;
