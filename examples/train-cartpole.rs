extern crate genrl;
extern crate neuralops;
extern crate operator;

use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
use genrl::env::{Episode};
use genrl::opt::pg::{PolicyGradWorker};
use neuralops::prelude::*;
use operator::prelude::*;

fn main() {
  let init_cfg = CartpoleConfig::default();
  let minibatch_sz = 32;

  let mut op_cfg = vec![];
  op_cfg.push(OperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   minibatch_sz,
    stride:     4,
  }));
  op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   minibatch_sz,
    in_dim:     4,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  /*op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   minibatch_sz,
    in_dim:     4,
    out_dim:    16,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   minibatch_sz,
    in_dim:     16,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));*/
  op_cfg.push(OperatorConfig::SoftmaxNLLClassLoss(ClassLossOperatorConfig{
    batch_sz:       minibatch_sz,
    minibatch_sz:   minibatch_sz,
    num_classes:    2,
  }));
  let op = SeqOperator::new(op_cfg, OpCapability::Backward);

  let mut episodes_batch: Vec<Episode<CartpoleEnv>> = Vec::with_capacity(minibatch_sz);

  let mut policy_grad: PolicyGradWorker<CartpoleEnv, SeqOperator<f32, _, _>> = PolicyGradWorker::new(op);

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
        episodes_batch[idx].sample_discrete(&mut policy_grad.operator);
      }
    }
    //policy_grad.reset_opt_stats();
    policy_grad.step(&mut episodes_batch.drain( .. ));
    if iter_nr % 10 == 0 {
      //println!("DEBUG: iter: {} stats: {:?}", iter_nr, policy_grad.get_opt_stats());
    }
  }
}
