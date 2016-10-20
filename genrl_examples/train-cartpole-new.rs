extern crate genrl;
extern crate neuralops;
extern crate operator;
extern crate rng;

extern crate rand;

//use genrl::examples::bandit::{BanditConfig, BanditEnv};
use genrl::examples::cartpole::{CartpoleConfig, CartpoleEnv};
use genrl::env::{Discount, DiscountedValue, Episode};
//use genrl::opt::pg::{PolicyGradConfig, PolicyGradWorker};
use genrl::opt::pg_new::{PolicyGradConfig, SgdPolicyGradWorker};
//use genrl::wrappers::{DiscountedWrapConfig, DiscountedWrapEnv};
use neuralops::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};

/*impl StochasticPolicy for SoftmaxNLLClassLoss<SampleItem> {
}*/

fn main() {
  let mut init_cfg = CartpoleConfig::default();
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

  let pg_cfg = PolicyGradConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    //step_size:      0.1,
    step_size:      0.05,
    max_horizon:    max_horizon,
    //update_steps:   Some(50),
    update_steps:   Some(max_horizon),
    baseline:       0.0,
    init_cfg:       init_cfg,
    value_cfg:      Discount(0.99),
  };
  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  let mut policy_grad: SgdPolicyGradWorker<CartpoleEnv, DiscountedValue<f32>, _> = SgdPolicyGradWorker::new(pg_cfg, loss);
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
