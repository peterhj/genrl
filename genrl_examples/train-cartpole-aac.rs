extern crate genrl;
extern crate neuralops;
extern crate operator;
extern crate rng;

extern crate rand;

use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
use genrl::env::{SumValue, Discount, DiscountedValue, Episode};
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
  //init_cfg.horizon = 300;
  let batch_sz = 32;
  let minibatch_sz = 32;
  let max_horizon = 300;
  let max_iter = 10000;

  let input = NewVarInputOperator::new(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 4,
    out_dim:    (1, 1, 4),
    preprocs:   vec![
    ],
  }, OpCapability::Backward);
  let affine1 = NewAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    8,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, input, 0);
  let affine2 = NewAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     8,
    out_dim:    2,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, affine1, 0);
  let loss = SoftmaxNLLClassLoss::new(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    2,
  }, OpCapability::Backward, affine2, 0);

  let input = NewVarInputOperator::new(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 4,
    out_dim:    (1, 1, 4),
    preprocs:   vec![
    ],
  }, OpCapability::Backward);
  /*let affine1 = NewAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    8,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, input, 0);*/
  let affine2 = NewAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4,
    out_dim:    1,
    act_kind:   ActivationKind::Logistic,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, input, 0);
  //}, OpCapability::Backward, affine1, 0);
  let v_loss = LstSqRegressLoss::new(RegressLossConfig{
    batch_sz:   batch_sz,
  }, OpCapability::Backward, affine2, 0);
  /*let v_loss = NormLstSqRegressLoss::new(NormLstSqRegressLossConfig{
    batch_sz:   batch_sz,
    avg_rate:   0.01,
    epsilon:    1.0e-6,
    init_var:   1.0,
  }, OpCapability::Backward, affine2, 0);*/

  let pg_cfg = SgdAdvActorCriticConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    step_size:      0.01,
    v_step_size:    0.001,
    max_horizon:    max_horizon,
    update_steps:   Some(100),
    init_cfg:       init_cfg,
    value_cfg:      Discount(0.99),
    eval_vcfg:      (),
  };
  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  let mut policy_grad: SgdAdvActorCriticWorker<CartpoleEnv, DiscountedValue<f32>, SumValue<f32>, _, _> = SgdAdvActorCriticWorker::new(pg_cfg, loss, v_loss);
  policy_grad.init_param(&mut rng);
  for iter_nr in 0 .. max_iter {
    let avg_value = policy_grad.update();
    if iter_nr % 100 == 0 {
      //println!("DEBUG: iter: {} stats: {:?}", iter_nr, policy_grad.get_opt_stats());
      //policy_grad.reset_opt_stats();
      println!("DEBUG: iter: {} res: {:?}", iter_nr, avg_value);
    }
  }
}
