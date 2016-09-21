extern crate genrl;
extern crate operator;

use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
use genrl::env::{Episode};
use genrl::opt::pg::{PolicyGradWorker};
use operator::prelude::*;

fn main() {
  let init_cfg = CartpoleConfig::default();

  let minibatch_sz = 32;
  let mut episodes_batch: Vec<Episode<CartpoleEnv>> = Vec::with_capacity(minibatch_sz);

  let mut policy_grad: PolicyGradWorker<CartpoleEnv, _, _> = PolicyGradWorker::new();

  let horizon = 1000;
  let max_iter = 1000;
  for iter_nr in 0 .. max_iter {
    episodes_batch.clear();
    for idx in 0 .. minibatch_sz {
      episodes_batch.push(Episode::new());
      episodes_batch[idx].reset(&init_cfg);
      for _ in 0 .. horizon {
        if episodes_batch[idx].terminated() {
          break;
        }
        episodes_batch[idx].sample_discrete(());
      }
    }
    policy_grad.reset_opt_stats();
    policy_grad.step(&mut episodes_batch.drain( .. ));
    if iter_nr % 10 == 0 {
      println!("DEBUG: iter: {} stats: {:?}", iter_nr, policy_grad.get_opt_stats());
    }
  }
}
