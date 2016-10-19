extern crate genrl;
extern crate neuralops;
extern crate operator;
extern crate rng;

extern crate rand;

use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
use genrl::env::{DiscountedValue, Episode};
use genrl::opt::aac_new::{SgdAdvActorCriticConfig, SgdAdvActorCriticWorker};
use genrl::opt::pg_new::{PolicyGradConfig, SgdPolicyGradWorker};
use neuralops::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};

/*impl StochasticPolicy for SoftmaxNLLClassLoss<SampleItem> {
}*/

fn main() {
  let mut init_cfg = CartpoleConfig::default();
  init_cfg.horizon = 300;
  let batch_sz = 32;
  let minibatch_sz = 32;
  let max_horizon = init_cfg.horizon;
  let max_iter = 10000;

  let input = NewVarInputOperator::new(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 4,
    out_dim:    (1, 1, 4),
    preprocs:   vec![
    ],
  }, OpCapability::Backward);
  let affine = NewAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }, OpCapability::Backward, input, 0);
  let loss = SoftmaxNLLClassLoss::new(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    2,
  }, OpCapability::Backward, affine, 0);

  let input = NewVarInputOperator::new(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 4,
    out_dim:    (1, 1, 4),
    preprocs:   vec![
    ],
  }, OpCapability::Backward);
  let affine = NewAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }, OpCapability::Backward, input, 0);
  let v_loss = NormLstSqRegressLoss::new(NormLstSqRegressLossConfig{
    batch_sz:   batch_sz,
    avg_rate:   0.01,
    epsilon:    1.0e-5,
    init_var:   1.0,
  }, OpCapability::Backward, affine, 0);

  let pg_cfg = SgdAdvActorCriticConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    step_size:      0.1,
    v_step_size:    0.1,
    max_horizon:    max_horizon,
    update_steps:   Some(max_horizon),
    init_cfg:       init_cfg,
    value_cfg:      0.99,
  };
  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  let mut policy_grad: SgdAdvActorCriticWorker<CartpoleEnv, DiscountedValue<f32>, _, _> = SgdAdvActorCriticWorker::new(pg_cfg, loss, v_loss);
  policy_grad.init_param(&mut rng);
  for iter_nr in 0 .. max_iter {
    let avg_value = policy_grad.update();
    if iter_nr % 20 == 0 {
      //println!("DEBUG: iter: {} stats: {:?}", iter_nr, policy_grad.get_opt_stats());
      //policy_grad.reset_opt_stats();
      println!("DEBUG: iter: {} res: {:?}", iter_nr, avg_value);
    }
  }
}
