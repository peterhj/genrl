use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};

use densearray::prelude::*;
use float::ord::{F32InfNan};
use iter_utils::{argmax};
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

pub struct BasePolicyGrad<E, V> where E: 'static + Env, V: Value<Res=E::Response> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  //pub max_horizon:  usize,
  pub cache:        Vec<SampleItem>,
  pub cache_idxs:   Vec<usize>,
  pub act_dist:     DiscreteDist32,
  pub episodes:     Vec<Episode<E>>,
  pub ep_k_offsets: Vec<usize>,
  pub ep_is_term:   Vec<bool>,
  pub raw_actvals:  Vec<Vec<f32>>,
  pub smooth_avals: Vec<Vec<f32>>,
  pub baseline_val: Vec<Vec<f32>>,
  pub final_values: Vec<Option<f32>>,
  _marker:  PhantomData<(E, V)>,
}

impl<E, V> BasePolicyGrad<E, V>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
{
  pub fn new<R>(minibatch_sz: usize, /*max_horizon: usize,*/ init_cfg: &E::Init, rng: &mut R) -> BasePolicyGrad<E, V> where R: Rng {
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
    let mut raw_actvals = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      raw_actvals.push(vec![]);
    }
    let mut smooth_avals = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      smooth_avals.push(vec![]);
    }
    let mut baseline_val = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      baseline_val.push(vec![]);
    }
    let mut final_values = Vec::with_capacity(minibatch_sz);
    final_values.resize(minibatch_sz, None);
    BasePolicyGrad{
      batch_sz:     minibatch_sz, // FIXME(20161018): a hack.
      minibatch_sz: minibatch_sz,
      //max_horizon:  max_horizon,
      cache:        cache,
      cache_idxs:   cache_idxs,
      act_dist:     DiscreteDist32::new(<E::Action as Action>::dim()),
      episodes:     episodes,
      ep_k_offsets: ep_k_offsets,
      ep_is_term:   ep_is_term,
      raw_actvals:  raw_actvals,
      smooth_avals: smooth_avals,
      baseline_val: baseline_val,
      final_values: final_values,
      _marker:      PhantomData,
    }
  }

  pub fn reset<R>(&mut self, init_cfg: &E::Init, rng: &mut R) where R: Rng {
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      episode.reset(init_cfg, rng);
      assert_eq!(0, episode.horizon());
      self.ep_k_offsets[idx] = episode.horizon();
      self.ep_is_term[idx] = false;
    }
  }

  pub fn sample_epsilon_greedy<ValueFn, R>(&mut self, max_horizon: Option<usize>, max_num_steps: Option<usize>, init_cfg: &E::Init, eps_greedy: f32, value_fn: &mut ValueFn, rng: &mut R)
  where R: Rng,
        ValueFn: DiffLoss<SampleItem, [f32]>,
  {
    let action_dim = <E::Action as Action>::dim();
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      if self.ep_is_term[idx] || episode.terminated() || (max_horizon.is_some() && episode.horizon() >= max_horizon.unwrap()) {
        episode.reset(init_cfg, rng);
        assert_eq!(0, episode.horizon());
      }
      self.ep_k_offsets[idx] = episode.horizon();
      self.ep_is_term[idx] = false;
    }
    let mut step = 0;
    loop {
      let mut max_count = 0;
      let mut term_count = 0;
      self.cache.clear();
      self.cache_idxs.clear();
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if let Some(max_horizon) = max_horizon {
          if episode.horizon() >= max_horizon {
            max_count += 1;
            continue;
          }
        }
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
        let env_repr_dim = prev_env._shape3d();
        let mut item = SampleItem::new();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(prev_env.clone());
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        self.cache.push(item);
        self.cache_idxs.push(idx);
      }
      if max_count + term_count >= self.episodes.len() {
        assert_eq!(max_count + term_count, self.episodes.len());
        break;
      }
      assert!(!self.cache.is_empty());
      assert_eq!(self.cache.len(), self.cache_idxs.len());
      value_fn.load_batch(&self.cache);
      value_fn.forward(OpPhase::Learning);
      let mut cache_rank = 0;
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if cache_rank >= self.cache_idxs.len() {
          break;
        }
        if idx != self.cache_idxs[cache_rank] {
          continue;
        }
        // XXX(20161009): sometimes the policy output contains NaNs because
        // all probabilities were zero, should gracefully handle this case.
        //let output = policy.policy_probs();
        let output = value_fn._get_pred();
        // FIXME(20161025): epsilon greedy sampling.
        let u = rng.gen_range(0.0, 1.0);
        let act_idx = if u < eps_greedy {
          rng.gen_range(0, action_dim)
        } else {
          let max_idx = match argmax(output[cache_rank * action_dim .. (cache_rank+1) * action_dim].iter().map(|&v| F32InfNan(v))) {
            None => panic!("Q function has no argmax!"),
            Some(max_idx) => max_idx,
          };
          max_idx
        };
        /*let act_idx = match self.act_dist.reset(&(*output)[cache_rank * action_dim .. (cache_rank+1) * action_dim]) {
          Ok(_)   => self.act_dist.sample(rng).unwrap(),
          Err(_)  => rng.gen_range(0, <E::Action as Action>::dim()),
        };*/
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
      if let Some(max_num_steps) = max_num_steps {
        if step >= max_num_steps {
          break;
        }
      }
    }
    for (idx, _) in self.episodes.iter().enumerate() {
      self.final_values[idx] = None;
    }
  }

  pub fn sample_steps<Policy, R>(&mut self, max_horizon: Option<usize>, max_num_steps: Option<usize>, init_cfg: &E::Init, policy: &mut Policy, rng: &mut R)
  where R: Rng,
        Policy: DiffLoss<SampleItem, [f32]> //+ StochasticPolicy,
  {
    let action_dim = <E::Action as Action>::dim();
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      if self.ep_is_term[idx] || episode.terminated() || (max_horizon.is_some() && episode.horizon() >= max_horizon.unwrap()) {
        episode.reset(init_cfg, rng);
        assert_eq!(0, episode.horizon());
      }
      self.ep_k_offsets[idx] = episode.horizon();
      self.ep_is_term[idx] = false;
    }
    let mut step = 0;
    loop {
      let mut max_count = 0;
      let mut term_count = 0;
      self.cache.clear();
      self.cache_idxs.clear();
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if let Some(max_horizon) = max_horizon {
          if episode.horizon() >= max_horizon {
            max_count += 1;
            continue;
          }
        }
        if self.ep_is_term[idx] || episode.terminated() {
          term_count += 1;
          self.ep_is_term[idx] = true;
          continue;
        }
        let k = self.ep_k_offsets[idx] + step;
        assert_eq!(k, episode.horizon());
        let mut item = SampleItem::new();
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let env_repr_dim = prev_env._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(prev_env.clone());
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        self.cache.push(item);
        self.cache_idxs.push(idx);
      }
      if max_count + term_count >= self.episodes.len() {
        assert_eq!(max_count + term_count, self.episodes.len());
        break;
      }
      assert!(!self.cache.is_empty());
      assert_eq!(self.cache.len(), self.cache_idxs.len());
      // FIXME(20161018): this computes the _minibatch_, but we may want to use
      // a smaller _batch_ here just like during the policy gradient.
      policy.load_batch(&self.cache);
      policy.forward(OpPhase::Learning);
      let mut cache_rank = 0;
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if cache_rank >= self.cache_idxs.len() {
          break;
        }
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
      if let Some(max_num_steps) = max_num_steps {
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
    for (idx, episode) in self.episodes.iter().enumerate() {
      let mut suffix_val: Option<V> = None;
      let mut smooth_suffix_val = if let Some(final_val) = self.final_values[idx] {
        Some(<V as Value>::from_scalar(final_val, *value_cfg))
      } else {
        None
      };
      assert!(episode.horizon() > 0);
      //self.raw_actvals[idx].clear();
      self.raw_actvals[idx].resize(episode.horizon(), 0.0);
      //self.smooth_avals[idx].clear();
      self.smooth_avals[idx].resize(episode.horizon(), 0.0);
      for k in (self.ep_k_offsets[idx] .. episode.horizon()).rev() {
        if let Some(res) = episode.steps[k].res {
          if let Some(ref mut suffix_val) = suffix_val {
            suffix_val.lreduce(res);
          } else {
            suffix_val = Some(<V as Value>::from_res(res, *value_cfg));
          }
          if let Some(ref mut smooth_suffix_val) = smooth_suffix_val {
            smooth_suffix_val.lreduce(res);
          } else {
            smooth_suffix_val = Some(<V as Value>::from_res(res, *value_cfg));
          }
        }
        if let Some(suffix_val) = suffix_val {
          self.raw_actvals[idx][k] = suffix_val.to_scalar();
        } else {
          self.raw_actvals[idx][k] = 0.0;
        }
        if let Some(smooth_suffix_val) = smooth_suffix_val {
          self.smooth_avals[idx][k] = smooth_suffix_val.to_scalar();
        } else {
          self.smooth_avals[idx][k] = 0.0;
        }
      }
    }
  }

  pub fn impute_baselines<ValueFn>(&mut self, max_num_steps: Option<usize>, value_fn: &mut ValueFn) where ValueFn: DiffLoss<SampleItem, [f32]> {
    for (idx, episode) in self.episodes.iter().enumerate() {
      //self.baseline_val[idx].clear();
      self.baseline_val[idx].resize(episode.horizon(), 0.0);
    }
    let mut step = 0;
    loop {
      let mut term_count = 0;
      self.cache.clear();
      self.cache_idxs.clear();
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        let k = self.ep_k_offsets[idx] + step;
        if k >= episode.horizon() {
          term_count += 1;
          continue;
        }
        let mut item = SampleItem::new();
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let env_repr_dim = prev_env._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(prev_env.clone());
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        self.cache.push(item);
        self.cache_idxs.push(idx);
      }
      if term_count >= self.episodes.len() {
        assert_eq!(term_count, self.episodes.len());
        break;
      }
      value_fn.load_batch(&self.cache);
      value_fn.forward(OpPhase::Learning);
      let mut cache_rank = 0;
      for (idx, episode) in self.episodes.iter_mut().enumerate() {
        if cache_rank >= self.cache_idxs.len() {
          break;
        }
        if idx != self.cache_idxs[cache_rank] {
          continue;
        }
        let output = value_fn._get_pred();
        let k = self.ep_k_offsets[idx] + step;
        self.baseline_val[idx][k] = output[cache_rank];
        cache_rank += 1;
      }
      assert_eq!(cache_rank, self.cache.len());
      step += 1;
      if let Some(max_num_steps) = max_num_steps {
        if step >= max_num_steps {
          break;
        }
      }
    }
  }

  pub fn impute_final_values<ValueFn>(&mut self, value_fn: &mut ValueFn) where ValueFn: DiffLoss<SampleItem, [f32]> {
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
    value_fn.load_batch(&self.cache);
    value_fn.forward(OpPhase::Learning);
    let output = value_fn._get_pred();
    let mut cache_rank = 0;
    for (idx, _) in self.episodes.iter().enumerate() {
      if cache_rank >= self.cache_idxs.len() {
        break;
      }
      if idx != self.cache_idxs[cache_rank] {
        continue;
      }
      let impute_val = output[cache_rank];
      self.final_values[idx] = Some(impute_val);
      cache_rank += 1;
    }
  }

  pub fn impute_final_q_values<ValueFn>(&mut self, value_fn: &mut ValueFn) where ValueFn: DiffLoss<SampleItem, [f32]> {
    let action_dim = <E::Action as Action>::dim();
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
    value_fn.load_batch(&self.cache);
    value_fn.forward(OpPhase::Learning);
    let output = value_fn._get_pred();
    let mut cache_rank = 0;
    for (idx, _) in self.episodes.iter().enumerate() {
      if cache_rank >= self.cache_idxs.len() {
        break;
      }
      if idx != self.cache_idxs[cache_rank] {
        continue;
      }
      let impute_val_idx = argmax(output[cache_rank * action_dim .. (cache_rank+1) * action_dim].iter().map(|&v| F32InfNan(v))).unwrap();
      let impute_val = output[cache_rank * action_dim + impute_val_idx];
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
      Policy: DiffLoss<SampleItem, [f32]> //+ StochasticPolicy,
{
  cfg:      PolicyGradConfig<E, V>,
  grad_sz:  usize,
  iter_counter: usize,
  rng:      Xorshiftplus128Rng,
  base_pg:  BasePolicyGrad<E, V>,
  policy:   Rc<RefCell<Policy>>,
  cache:    Vec<SampleItem>,
  param:    Vec<f32>,
  grad:     Vec<f32>,
}

impl<E, V, Policy> SgdPolicyGradWorker<E, V, Policy>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, [f32]> //+ StochasticPolicy,
{
  pub fn new(cfg: PolicyGradConfig<E, V>, policy: Rc<RefCell<Policy>>) -> SgdPolicyGradWorker<E, V, Policy> {
    let batch_sz = cfg.batch_sz;
    let minibatch_sz = cfg.minibatch_sz;
    let max_horizon = cfg.max_horizon;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let base_pg = BasePolicyGrad::new(minibatch_sz, &cfg.init_cfg, &mut rng);
    let grad_sz = policy.borrow_mut().diff_param_sz();
    //println!("DEBUG: grad sz: {}", grad_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    SgdPolicyGradWorker{
      cfg:      cfg,
      grad_sz:  grad_sz,
      iter_counter: 0,
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
    self.base_pg.sample_steps(Some(self.cfg.max_horizon), self.cfg.update_steps, &self.cfg.init_cfg, &mut *policy, &mut self.rng);
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
        item.kvs.insert::<SampleWeightKey>(self.base_pg.raw_actvals[idx][k] - self.cfg.baseline);
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
    policy.update_nondiff_param(self.iter_counter);
    policy.store_grad(&mut self.grad);
    //println!("DEBUG: grad:  {:?}", self.grad);
    // FIXME(20161018): only normalize by minibatch size if all episodes in the
    // minibatch are represented in the policy gradient.
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.grad.reshape(self.grad_sz));
    policy.load_diff_param(&mut self.param);
    //println!("DEBUG: param: {:?}", self.param);
    self.iter_counter += 1;
    let mut avg_value = 0.0;
    for idx in 0 .. self.cfg.minibatch_sz {
      avg_value += self.base_pg.raw_actvals[idx][self.base_pg.ep_k_offsets[idx]];
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
