extern crate genrl;
extern crate neuralops;
extern crate operator;
extern crate rng;

extern crate rand;

use genrl::examples::bandit::{BanditConfig, BanditEnv};
use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
//use genrl::discrete::{DiscreteSampler32};
use genrl::env::{Action, Episode}; //, batch_sample_discrete};
use genrl::opt::pg::{PolicyGradConfig, PolicyGradWorker};
use neuralops::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};

fn main() {
  let init_cfg = CartpoleConfig::default();
  let batch_sz = 32;
  let minibatch_sz = 32;
  let horizon = 100; //init_cfg.horizon;
  let max_iter = 1000;

  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     4,
    scale:      None,
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
    in_dim:     10,
    out_dim:    32,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     32,
    out_dim:    32,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     32,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));*/
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossOperatorConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    num_classes:    2,
  }));
  let op = SeqOperator::new(op_cfg, OpCapability::Backward);

  let mut episodes_batch: Vec<Episode<CartpoleEnv>> = Vec::with_capacity(minibatch_sz);

  let pg_cfg = PolicyGradConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    step_size:      0.0001,
    max_horizon:    horizon,
    baseline:       0.0,
  };
  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  let mut policy_grad: PolicyGradWorker<CartpoleEnv, SeqOperator<f32, _>> = PolicyGradWorker::new(pg_cfg, op);
  policy_grad.init_param(&mut rng);

  for iter_nr in 0 .. max_iter {
    episodes_batch.clear();
    for idx in 0 .. minibatch_sz {
      episodes_batch.push(Episode::new());
    }
    policy_grad.sample(&mut episodes_batch, &init_cfg, &mut rng);
    let mut episodes_iter = episodes_batch.clone();
    policy_grad.step(&mut episodes_iter.drain( .. ));
    if iter_nr % 1 == 0 {
      //println!("DEBUG: iter: {} stats: {:?}", iter_nr, policy_grad.get_opt_stats());
      //policy_grad.reset_opt_stats();
      let mut avg_value = 0.0;
      for episode in episodes_batch.iter() {
        avg_value += episode.response_value().unwrap_or(0.0);
      }
      avg_value /= episodes_batch.len() as f32;
      println!("DEBUG: iter: {} res: {:?}", iter_nr, avg_value);
    }
  }
}
