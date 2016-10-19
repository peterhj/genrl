use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};

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

pub struct EpisodeStepSample<E> where E: Env {
  //pub env:      Rc<RefCell<E>>,
  pub env:      Rc<E>,
  pub act_idx:  Option<u32>,
  pub suffix_r: Option<E::Response>,
  pub baseline: Option<f32>,
  weight:       Option<f32>,
}

impl<E> EpisodeStepSample<E> where E: Env {
  pub fn new(env: Rc<E>, act_idx: Option<u32>, suffix_r: Option<E::Response>) -> EpisodeStepSample<E> {
    EpisodeStepSample{
      env:          env,
      act_idx:      act_idx,
      suffix_r:     suffix_r,
      baseline:     None,
      weight:       None,
    }
  }

  pub fn set_baseline(&mut self, baseline: f32) {
    self.baseline = Some(baseline);
  }

  pub fn init_weight(&mut self) {
    self.weight = Some(self.suffix_r.map(|r| r.as_scalar()).unwrap() - self.baseline.unwrap_or(0.0));
  }
}

impl<E> SampleDatum<[f32]> for EpisodeStepSample<E> where E: Env + EnvRepr<f32> {
  fn extract_input(&self, output: &mut [f32]) -> Result<(), ()> {
    self.env.extract_observable(output);
    Ok(())
  }
}

impl<E> SampleLabel for EpisodeStepSample<E> where E: Env {
  fn class(&self) -> Option<u32> {
    self.act_idx
  }

  fn target(&self) -> Option<f32> {
    self.suffix_r.map(|r| r.as_scalar())
  }
}

impl<E> SampleLossWeight<ClassLoss> for EpisodeStepSample<E> where E: Env {
  fn weight(&self) -> Option<f32> {
    self.weight
  }

  fn mix_weight(&mut self, w: f32) -> Result<(), ()> {
    self.weight = Some(self.weight.unwrap_or(1.0) * w);
    Ok(())
  }
}

pub trait StochasticPolicy {
  type P: Deref<Target=[f32]>;

  fn policy_probs(&self) -> Self::P;
}

pub struct BasePolicyGrad<E, V, Policy> where E: 'static + Env, V: Value<Res=E::Response> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub max_horizon:  usize,
  pub cache:        Vec<SampleItem>,
  pub cache_idxs:   Vec<usize>,
  pub act_dist:     DiscreteDist32,
  pub episodes:     Vec<Episode<E>>,
  pub ep_k_offsets: Vec<usize>,
  pub ep_is_term:   Vec<bool>,
  pub step_values:  Vec<Vec<f32>>,
  pub impute_vals:  Vec<Vec<f32>>,
  pub final_values: Vec<Option<f32>>,
  _marker:  PhantomData<(E, V, Policy)>,
}

impl<E, V, Policy> BasePolicyGrad<E, V, Policy>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]> //+ StochasticPolicy,
{
  pub fn new<R>(minibatch_sz: usize, max_horizon: usize, init_cfg: &E::Init, rng: &mut R) -> BasePolicyGrad<E, V, Policy> where R: Rng {
    let mut cache = Vec::with_capacity(minibatch_sz);
    let mut cache_idxs = Vec::with_capacity(minibatch_sz);
    cache_idxs.resize(minibatch_sz, 0);
    let mut episodes = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      let mut episode = Episode::new();
      episode.reset(init_cfg, rng);
      episodes.push(episode);
    }
    let mut ep_k_offsets = Vec::with_capacity(minibatch_sz);
    ep_k_offsets.resize(minibatch_sz, 0);
    let mut ep_is_term = Vec::with_capacity(minibatch_sz);
    ep_is_term.resize(minibatch_sz, false);
    let mut step_values = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      step_values.push(vec![]);
    }
    let mut impute_vals = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      impute_vals.push(vec![]);
    }
    let mut final_values = Vec::with_capacity(minibatch_sz);
    final_values.resize(minibatch_sz, None);
    BasePolicyGrad{
      batch_sz:     minibatch_sz, // FIXME(20161018): a hack.
      minibatch_sz: minibatch_sz,
      max_horizon:  max_horizon,
      cache:        cache,
      cache_idxs:   cache_idxs,
      act_dist:     DiscreteDist32::new(<E::Action as Action>::dim()),
      episodes:     episodes,
      ep_k_offsets: ep_k_offsets,
      ep_is_term:   ep_is_term,
      step_values:  step_values,
      impute_vals:  impute_vals,
      final_values: final_values,
      _marker:      PhantomData,
    }
  }

  pub fn sample_steps<R>(&mut self, max_num_steps: Option<usize>, init_cfg: &E::Init, policy: &mut Policy, rng: &mut R) where R: Rng {
    let action_dim = <E::Action as Action>::dim();
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      if episode.terminated() || episode.horizon() >= self.max_horizon {
        episode.reset(init_cfg, rng);
      }
      self.ep_k_offsets[idx] = episode.horizon();
      self.ep_is_term[idx] = false;
    }

    let mut step = 0;
    loop {
      let mut term_count = 0;
      self.cache.clear();
      self.cache_idxs.clear();
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if self.ep_is_term[idx] || episode.terminated() {
          term_count += 1;
          self.ep_is_term[idx] = true;
          continue;
        }
        let k = self.ep_k_offsets[idx] + step;
        assert_eq!(k, episode.horizon());
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let mut item = SampleItem::new();
        let env_repr_dim = prev_env._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(prev_env.clone());
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        self.cache.push(item);
        self.cache_idxs.push(idx);
      }
      // FIXME(20161018): this computes the _minibatch_, but we may want to use
      // a smaller _batch_ here just like during the policy gradient.
      policy.load_batch(&self.cache);
      policy.forward(OpPhase::Learning);
      let mut cache_rank = 0;
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if idx != self.cache_idxs[cache_rank] {
          continue;
        }
        // XXX(20161009): sometimes the policy output contains NaNs because
        // all probabilities were zero, should gracefully handle this case.
        //let output = policy.policy_probs();
        let output = policy._get_pred();
        let act_idx = match self.act_dist.reset(&(*output)[cache_rank * action_dim .. (cache_rank+1) * action_dim]) {
          Ok(_)   => self.act_dist.sample(rng).unwrap(),
          Err(_)  => rng.gen_range(0, <E::Action as Action>::dim()),
        };
        let action = <E::Action as DiscreteAction>::from_idx(act_idx as u32);
        let k = self.ep_k_offsets[idx] + step;
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let next_env = (*prev_env).clone();
        if let Ok(res) = next_env.step(&action) {
          episode.steps.push(EpisodeStep{
            action:   action,
            res:      res,
            next_env: Rc::new(next_env),
          });
        } else {
          panic!();
        }
        cache_rank += 1;
      }
      assert_eq!(cache_rank, self.cache.len());
      step += 1;
      if term_count == self.episodes.len() {
        break;
      } else if let Some(max_num_steps) = max_num_steps {
        if step >= max_num_steps {
          break;
        }
      }
    }
    for (idx, _) in self.episodes.iter().enumerate() {
      self.final_values[idx] = None;
    }
  }

  pub fn fill_step_values(&mut self, value_cfg: &V::Cfg) {
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      let mut suffix_val = if let Some(final_val) = self.final_values[idx] {
        Some(<V as Value>::from_scalar(final_val, *value_cfg))
      } else {
        None
      };
      self.step_values[idx].resize(episode.horizon(), 0.0);
      for k in (self.ep_k_offsets[idx] .. episode.horizon()).rev() {
        if let Some(res) = episode.steps[k].res {
          if let Some(ref mut suffix_val) = suffix_val {
            suffix_val.lreduce(res);
          } else {
            suffix_val = Some(<V as Value>::from_res(res, *value_cfg));
          }
        }
        if let Some(suffix_val) = suffix_val {
          self.step_values[idx][k] = suffix_val.to_scalar();
        } else {
          self.step_values[idx][k] = 0.0;
        }
      }
    }
  }

  pub fn impute_step_values<ValueFn>(&mut self, value_op: &mut ValueFn) where ValueFn: DiffLoss<SampleItem, IoBuf=[f32]> {
    // FIXME(20161019)
    unimplemented!();
  }

  pub fn impute_final_values<ValueFn>(&mut self, value_op: &mut ValueFn) where ValueFn: DiffLoss<SampleItem, IoBuf=[f32]> {
    self.cache.clear();
    self.cache_idxs.clear();
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      if self.ep_is_term[idx] || episode.terminated() {
        continue;
      }
      let k = episode.horizon();
      let prev_env = match k {
        0 => episode.init_env.clone(),
        k => episode.steps[k-1].next_env.clone(),
      };
      let mut item = SampleItem::new();
      let env_repr_dim = prev_env._shape3d();
      item.kvs.insert::<SampleExtractInputKey<[f32]>>(prev_env.clone());
      item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
      self.cache.push(item);
      self.cache_idxs.push(idx);
    }
    value_op.load_batch(&self.cache);
    value_op.forward(OpPhase::Learning);
    let output = value_op._get_pred();
    let mut cache_rank = 0;
    for (idx, _) in self.episodes.iter().enumerate() {
      if idx != self.cache_idxs[cache_rank] {
        continue;
      }
      let impute_val = output[cache_rank];
      self.final_values[idx] = Some(impute_val);
      cache_rank += 1;
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct PolicyGradConfig<E, V> where E: 'static + Env, V: Value<Res=E::Response> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub max_horizon:  usize,
  pub update_steps: Option<usize>,
  pub baseline:     f32,
  pub init_cfg:     E::Init,
  pub value_cfg:    V::Cfg,
}

pub struct SgdPolicyGradWorker<E, V, Policy>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]> //+ StochasticPolicy,
{
  cfg:      PolicyGradConfig<E, V>,
  grad_sz:  usize,
  rng:      Xorshiftplus128Rng,
  base_pg:  BasePolicyGrad<E, V, Policy>,
  policy:   Rc<RefCell<Policy>>,
  cache:    Vec<SampleItem>,
  param:    Vec<f32>,
  grad:     Vec<f32>,
}

impl<E, V, Policy> SgdPolicyGradWorker<E, V, Policy>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]> //+ StochasticPolicy,
{
  pub fn new(cfg: PolicyGradConfig<E, V>, policy: Rc<RefCell<Policy>>) -> SgdPolicyGradWorker<E, V, Policy> {
    let batch_sz = cfg.batch_sz;
    let minibatch_sz = cfg.minibatch_sz;
    let max_horizon = cfg.max_horizon;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let base_pg = BasePolicyGrad::new(minibatch_sz, max_horizon, &cfg.init_cfg, &mut rng);
    let grad_sz = policy.borrow_mut().diff_param_sz();
    //println!("DEBUG: grad sz: {}", grad_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    SgdPolicyGradWorker{
      cfg:      cfg,
      grad_sz:  grad_sz,
      rng:      rng,
      base_pg:  base_pg,
      policy:   policy,
      cache:    Vec::with_capacity(batch_sz),
      param:    param,
      grad:     grad,
    }
  }

  pub fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut policy = self.policy.borrow_mut();
    policy.init_param(rng);
    policy.store_diff_param(&mut self.param);
    //println!("DEBUG: param: {:?}", self.param);
  }

  pub fn update(&mut self) -> f32 {
    let mut policy = self.policy.borrow_mut();
    self.base_pg.sample_steps(self.cfg.update_steps, &self.cfg.init_cfg, &mut policy, &mut self.rng);
    self.base_pg.fill_step_values(&self.cfg.value_cfg);
    policy.reset_loss();
    policy.reset_grad();
    policy.next_iteration();
    self.cache.clear();
    //print!("DEBUG: weights: ");
    for (idx, episode) in self.base_pg.episodes.iter().enumerate() {
      for k in self.base_pg.ep_k_offsets[idx] .. episode.horizon() {
        let mut item = SampleItem::new();
        let env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let env_repr_dim = env._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        item.kvs.insert::<SampleClassLabelKey>(episode.steps[k].action.idx());
        item.kvs.insert::<SampleWeightKey>(self.base_pg.step_values[idx][k] - self.cfg.baseline);
        self.cache.push(item);
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }
        policy.load_batch(&self.cache);
        policy.forward(OpPhase::Learning);
        policy.backward();
        self.cache.clear();
      }
    }
    //println!("");
    if !self.cache.is_empty() {
      policy.load_batch(&self.cache);
      policy.forward(OpPhase::Learning);
      policy.backward();
      self.cache.clear();
    }
    policy.store_grad(&mut self.grad);
    //println!("DEBUG: grad:  {:?}", self.grad);
    // FIXME(20161018): only normalize by minibatch size if all episodes in the
    // minibatch are represented in the policy gradient.
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.grad.reshape(self.grad_sz));
    policy.load_diff_param(&mut self.param);
    //println!("DEBUG: param: {:?}", self.param);
    let mut avg_value = 0.0;
    for idx in 0 .. self.cfg.minibatch_sz {
      avg_value += self.base_pg.step_values[idx][self.base_pg.ep_k_offsets[idx]];
    }
    avg_value /= self.cfg.minibatch_sz as f32;
    avg_value
  }
}

/*impl<E, Op> OptStats<()> for SgdPolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
{
  fn reset_opt_stats(&mut self) {
    unimplemented!();
  }

  fn get_opt_stats(&self) -> &() {
    unimplemented!();
  }
}*/
