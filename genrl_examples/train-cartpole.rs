extern crate genrl;
extern crate neuralops;
extern crate operator;
extern crate rng;

extern crate rand;

//use genrl::examples::bandit::{BanditConfig, BanditEnv};
use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
use genrl::env::{Episode};
use genrl::opt::pg::{PolicyGradConfig, PolicyGradWorker};
use genrl::wrappers::{DiscountedWrapConfig, DiscountedWrapEnv};
use neuralops::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};

fn main() {
  let mut init_cfg = CartpoleConfig::default();
  //init_cfg.horizon = 300;
  let wrap_init_cfg = DiscountedWrapConfig{
    discount:   0.99,
    env_init:   init_cfg,
  };
  let batch_sz = 32;
  let minibatch_sz = 32;
  let max_horizon = 300;
  let max_iter = 1000;

  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     4,
    preprocs:   vec![],
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  /*op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    16,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  /*op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     16,
    out_dim:    16,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));*/
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     16,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));*/
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    2,
  }));
  let op = SeqOperator::new(op_cfg, OpCapability::Backward);

  let pg_cfg = PolicyGradConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    //step_size:      1.0,
    step_size:      0.05,
    max_horizon:    max_horizon,
    baseline:       0.0,
  };
  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  let mut policy_grad: PolicyGradWorker<DiscountedWrapEnv<CartpoleEnv>, _> = PolicyGradWorker::new(pg_cfg, op);
  policy_grad.init_param(&mut rng);

  let mut episodes_batch: Vec<Episode<DiscountedWrapEnv<CartpoleEnv>>> = Vec::with_capacity(minibatch_sz);
  for _ in 0 .. minibatch_sz {
    episodes_batch.push(Episode::new());
  }

  for iter_nr in 0 .. max_iter {
    policy_grad.sample(&mut episodes_batch, &wrap_init_cfg, &mut rng);
    let mut episodes_iter = episodes_batch.clone();
    policy_grad.step(&mut episodes_iter.drain( .. ));
    if iter_nr % 1 == 0 {
      //println!("DEBUG: iter: {} stats: {:?}", iter_nr, policy_grad.get_opt_stats());
      //policy_grad.reset_opt_stats();
      let mut avg_value = 0.0;
      for episode in episodes_batch.iter() {
        avg_value += episode.value().unwrap_or(0.0);
      }
      avg_value /= episodes_batch.len() as f32;
      println!("DEBUG: iter: {} res: {:?}", iter_nr, avg_value);
    }
  }
}
