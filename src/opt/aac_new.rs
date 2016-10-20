use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};
use opt::pg_new::{BasePolicyGrad};

use densearray::prelude::*;
use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::ops::{Deref};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct SgdAdvActorCriticConfig<E, V> where E: 'static + Env, V: Value<Res=E::Response> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub v_step_size:  f32,
  pub max_horizon:  usize,
  pub update_steps: Option<usize>,
  pub init_cfg:     E::Init,
  pub value_cfg:    V::Cfg,
}

pub struct SgdAdvActorCriticWorker<E, V, Policy, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  cfg:      SgdAdvActorCriticConfig<E, V>,
  grad_sz:  usize,
  vgrad_sz: usize,
  iter_counter: usize,
  rng:      Xorshiftplus128Rng,
  base_pg:  BasePolicyGrad<E, V, Policy>,
  policy:   Rc<RefCell<Policy>>,
  value_fn: Rc<RefCell<ValueFn>>,
  cache:    Vec<SampleItem>,
  vcache:   Vec<SampleItem>,
  param:    Vec<f32>,
  grad:     Vec<f32>,
  vparam:   Vec<f32>,
  vgrad:    Vec<f32>,
}

impl<E, V, Policy, ValueFn> SgdAdvActorCriticWorker<E, V, Policy, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  pub fn new(cfg: SgdAdvActorCriticConfig<E, V>, policy: Rc<RefCell<Policy>>, value_fn: Rc<RefCell<ValueFn>>) -> SgdAdvActorCriticWorker<E, V, Policy, ValueFn> {
    let batch_sz = cfg.batch_sz;
    let minibatch_sz = cfg.minibatch_sz;
    let max_horizon = cfg.max_horizon;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let base_pg = BasePolicyGrad::new(minibatch_sz, max_horizon, &cfg.init_cfg, &mut rng);
    let pgrad_sz = policy.borrow_mut().diff_param_sz();
    let vgrad_sz = value_fn.borrow_mut().diff_param_sz();
    //println!("DEBUG: grad sz: {}", grad_sz);
    let mut param = Vec::with_capacity(pgrad_sz);
    param.resize(pgrad_sz, 0.0);
    let mut grad = Vec::with_capacity(pgrad_sz);
    grad.resize(pgrad_sz, 0.0);
    let mut vparam = Vec::with_capacity(vgrad_sz);
    vparam.resize(vgrad_sz, 0.0);
    let mut vgrad = Vec::with_capacity(vgrad_sz);
    vgrad.resize(vgrad_sz, 0.0);
    SgdAdvActorCriticWorker{
      cfg:      cfg,
      grad_sz:  pgrad_sz,
      vgrad_sz: vgrad_sz,
      iter_counter: 0,
      rng:      rng,
      base_pg:  base_pg,
      policy:   policy,
      value_fn: value_fn,
      cache:    Vec::with_capacity(batch_sz),
      vcache:   Vec::with_capacity(batch_sz),
      param:    param,
      grad:     grad,
      vparam:   vparam,
      vgrad:    vgrad,
    }
  }

  pub fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut policy = self.policy.borrow_mut();
    policy.init_param(rng);
    policy.store_diff_param(&mut self.param);
    let mut value_fn = self.value_fn.borrow_mut();
    value_fn.init_param(rng);
    value_fn.store_diff_param(&mut self.vparam);
  }

  pub fn update(&mut self) -> f32 {
    let mut policy = self.policy.borrow_mut();
    let mut value_fn = self.value_fn.borrow_mut();
    self.base_pg.sample_steps(self.cfg.update_steps, &self.cfg.init_cfg, &mut policy, &mut self.rng);
    self.base_pg.impute_step_values(self.cfg.update_steps, &mut *value_fn);
    self.base_pg.impute_final_values(&mut *value_fn);
    self.base_pg.fill_step_values(&self.cfg.value_cfg);
    policy.reset_loss();
    policy.reset_grad();
    policy.next_iteration();
    value_fn.reset_loss();
    value_fn.reset_grad();
    value_fn.next_iteration();
    self.cache.clear();
    self.vcache.clear();
    let mut steps_count = 0;
    let mut tmp_baselines = vec![];
    for (idx, episode) in self.base_pg.episodes.iter().enumerate() {
      for k in self.base_pg.ep_k_offsets[idx] .. episode.horizon() {
        let mut policy_item = SampleItem::new();
        let mut value_fn_item = SampleItem::new();
        let env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let env_repr_dim = env._shape3d();
        let action_value = self.base_pg.eval_actvals[idx][k];
        let smoothed_action_value = self.base_pg.smooth_avals[idx][k];
        let baseline_value = self.base_pg.baseline_val[idx][k];
        if k == 0 {
          tmp_baselines.push(baseline_value);
        }
        policy_item.kvs.insert::<SampleExtractInputKey<[f32]>>(env.clone());
        policy_item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        policy_item.kvs.insert::<SampleClassLabelKey>(episode.steps[k].action.idx());
        policy_item.kvs.insert::<SampleWeightKey>(action_value - baseline_value);
        //policy_item.kvs.insert::<SampleWeightKey>(smoothed_action_value - baseline_value);
        value_fn_item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
        value_fn_item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        value_fn_item.kvs.insert::<SampleRegressTargetKey>(action_value);
        self.cache.push(policy_item);
        self.vcache.push(value_fn_item);
        steps_count += 1;
        assert_eq!(self.cache.len(), self.vcache.len());
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }
        policy.load_batch(&self.cache);
        policy.forward(OpPhase::Learning);
        policy.backward();
        value_fn.load_batch(&self.vcache);
        value_fn.forward(OpPhase::Learning);
        value_fn.backward();
        self.cache.clear();
        self.vcache.clear();
      }
    }
    //println!("DEBUG: baselines[0]: {:?}", &tmp_baselines);
    //println!("DEBUG: baselines[H]: {:?}", &self.base_pg.final_values);
    if !self.cache.is_empty() {
      assert!(!self.vcache.is_empty());
      policy.load_batch(&self.cache);
      policy.forward(OpPhase::Learning);
      policy.backward();
      value_fn.load_batch(&self.vcache);
      value_fn.forward(OpPhase::Learning);
      value_fn.backward();
      self.cache.clear();
      self.vcache.clear();
    }
    // FIXME(20161018): only normalize by minibatch size if all episodes in the
    // minibatch are represented in the policy gradient.
    policy.update_nondiff_param(self.iter_counter);
    value_fn.update_nondiff_param(self.iter_counter);
    policy.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.grad.reshape(self.grad_sz));
    policy.load_diff_param(&mut self.param);
    // FIXME(20161018): only normalize by minibatch size if all episodes in the
    // minibatch are represented in the policy gradient.
    value_fn.store_grad(&mut self.vgrad);
    self.vgrad.reshape_mut(self.vgrad_sz).scale(1.0 / steps_count as f32);
    self.vparam.reshape_mut(self.vgrad_sz).add(-self.cfg.v_step_size, self.vgrad.reshape(self.vgrad_sz));
    value_fn.load_diff_param(&mut self.vparam);
    self.iter_counter += 1;
    let mut avg_value = 0.0;
    for idx in 0 .. self.cfg.minibatch_sz {
      avg_value += self.base_pg.eval_actvals[idx][self.base_pg.ep_k_offsets[idx]];
    }
    avg_value /= self.cfg.minibatch_sz as f32;
    avg_value
  }
}
