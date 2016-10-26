use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};
use opt::pg_new::{BasePolicyGrad};

use densearray::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice, SharedMem, SharedSlice};
use sharedmem::sync::{SpinBarrier};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::intrinsics::{volatile_copy_memory};
use std::marker::{PhantomData};
use std::ops::{Deref};
use std::rc::{Rc};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct AdamAsyncDiffQConfig<Init, VCfg, EvalVCfg> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub epoch_sz:     usize,
  pub step_size:    f32,
  pub grad_clip:    Option<f32>,
  pub eps_init:     f32,
  pub eps_anneal:   f32,
  pub eps_eval:     f32,
  pub gamma1:       f32,
  pub gamma2:       f32,
  pub epsilon:      f32,
  pub max_horizon:  usize,
  pub update_steps: usize,
  pub target_steps: usize,
  pub init_cfg:     Init,
  pub value_cfg:    VCfg,
  pub eval_cfg:     EvalVCfg,
}

#[derive(Clone)]
pub struct AdamAsyncDiffQBuilder {
  num_workers:  usize,
  step_count:   Arc<AtomicUsize>,
  iter_count:   Arc<AtomicUsize>,
  shared_bar:   Arc<SpinBarrier>,
  async_target: Arc<Mutex<Option<SharedMem<f32>>>>,
  async_param:  Arc<Mutex<Option<SharedMem<f32>>>>,
  async_gmean:  Arc<Mutex<Option<SharedMem<f32>>>>,
  async_gvar:   Arc<Mutex<Option<SharedMem<f32>>>>,
}

impl AdamAsyncDiffQBuilder {
  pub fn new(num_workers: usize) -> Self {
    AdamAsyncDiffQBuilder{
      num_workers:  num_workers,
      step_count:   Arc::new(AtomicUsize::new(0)),
      iter_count:   Arc::new(AtomicUsize::new(0)),
      shared_bar:   Arc::new(SpinBarrier::new(num_workers)),
      async_target: Arc::new(Mutex::new(None)),
      async_param:  Arc::new(Mutex::new(None)),
      async_gmean:  Arc::new(Mutex::new(None)),
      async_gvar:   Arc::new(Mutex::new(None)),
    }
  }

  pub fn into_worker<E, V, EvalV, ValueFn>(self, cfg: AdamAsyncDiffQConfig<E::Init, V::Cfg, EvalV::Cfg>, worker_rank: usize, value_fn: Rc<RefCell<ValueFn>>) -> AdamAsyncDiffQWorker<E, V, EvalV, ValueFn>
  where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
        E::Action: DiscreteAction,
        V: Value<Res=E::Response>,
        EvalV: Value<Res=E::Response>,
        ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
  {
    let batch_sz = cfg.batch_sz;
    let minibatch_sz = cfg.minibatch_sz;
    let max_horizon = cfg.max_horizon;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let base_pg = BasePolicyGrad::new(minibatch_sz, &cfg.init_cfg, &mut rng);
    let eval_pg = BasePolicyGrad::new(minibatch_sz, &cfg.init_cfg, &mut rng);
    let grad_sz = value_fn.borrow_mut().diff_param_sz();
    let mut target = Vec::with_capacity(grad_sz);
    target.resize(grad_sz, 0.0);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut gmean = Vec::with_capacity(grad_sz);
    gmean.resize(grad_sz, 0.0);
    let mut gvar = Vec::with_capacity(grad_sz);
    gvar.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    if worker_rank == 0 {
      let mut target = Vec::with_capacity(grad_sz);
      target.resize(grad_sz, 0.0);
      let mut param = Vec::with_capacity(grad_sz);
      param.resize(grad_sz, 0.0);
      let mut gmean = Vec::with_capacity(grad_sz);
      gmean.resize(grad_sz, 0.0);
      let mut gvar = Vec::with_capacity(grad_sz);
      gvar.resize(grad_sz, 0.0);
      let mut async_target = self.async_target.lock().unwrap();
      let mut async_param = self.async_param.lock().unwrap();
      let mut async_gmean = self.async_gmean.lock().unwrap();
      let mut async_gvar = self.async_gvar.lock().unwrap();
      *async_target = Some(SharedMem::new(target));
      *async_param = Some(SharedMem::new(param));
      *async_gmean = Some(SharedMem::new(gmean));
      *async_gvar = Some(SharedMem::new(gvar));
    }
    self.shared_bar.wait();
    let worker = AdamAsyncDiffQWorker{
      cfg:      cfg,
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      grad_sz:  grad_sz,
      step_count:   self.step_count,
      iter_count:   self.iter_count,
      shared_bar:   self.shared_bar,
      iter_counter: 0,
      rng:      rng,
      base_pg:  base_pg,
      eval_pg:  eval_pg,
      value_fn: value_fn,
      cache:    Vec::with_capacity(batch_sz),
      async_target: self.async_target.lock().unwrap().as_ref().unwrap().as_slice(),
      async_param:  self.async_param.lock().unwrap().as_ref().unwrap().as_slice(),
      async_gmean:  self.async_gmean.lock().unwrap().as_ref().unwrap().as_slice(),
      async_gvar:   self.async_gvar.lock().unwrap().as_ref().unwrap().as_slice(),
      target:   target,
      param:    param,
      grad:     grad,
      gmean:    gmean,
      gvar:     gvar,
      tmp_buf:  tmp_buf,
    };
    worker.shared_bar.wait();
    worker
  }
}

pub struct AdamAsyncDiffQWorker<E, V, EvalV, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  cfg:          AdamAsyncDiffQConfig<E::Init, V::Cfg, EvalV::Cfg>,
  worker_rank:  usize,
  num_workers:  usize,
  grad_sz:      usize,
  step_count:   Arc<AtomicUsize>,
  iter_count:   Arc<AtomicUsize>,
  shared_bar:   Arc<SpinBarrier>,
  iter_counter: usize,
  rng:          Xorshiftplus128Rng,
  base_pg:      BasePolicyGrad<E, V>,
  eval_pg:      BasePolicyGrad<E, EvalV>,
  value_fn:     Rc<RefCell<ValueFn>>,
  cache:        Vec<SampleItem>,
  async_target: SharedSlice<f32>,
  async_param:  SharedSlice<f32>,
  async_gmean:  SharedSlice<f32>,
  async_gvar:   SharedSlice<f32>,
  target:       Vec<f32>,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  gmean:        Vec<f32>,
  gvar:         Vec<f32>,
  tmp_buf:      Vec<f32>,
}

impl<E, V, EvalV, ValueFn> AdamAsyncDiffQWorker<E, V, EvalV, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  pub fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    if self.worker_rank == 0 {
      let mut value_fn = self.value_fn.borrow_mut();
      value_fn.init_param(rng);
      value_fn.store_diff_param(&mut self.param);
      self.target.copy_from_slice(&self.param);
      unsafe { volatile_copy_memory(self.async_target.as_ptr() as *mut _, self.target.as_ptr(), self.grad_sz) };
      unsafe { volatile_copy_memory(self.async_param.as_ptr() as *mut _, self.param.as_ptr(), self.grad_sz) };
      println!("DEBUG: rank: {} |param|: {} {:.6e}", self.worker_rank, self.grad_sz, self.param.reshape(self.grad_sz).l2_norm());
    }
    self.shared_bar.wait();
    if self.worker_rank != 0 {
      let mut value_fn = self.value_fn.borrow_mut();
      unsafe { volatile_copy_memory(self.target.as_mut_ptr(), self.async_target.as_ptr(), self.grad_sz) };
      unsafe { volatile_copy_memory(self.param.as_mut_ptr(), self.async_param.as_ptr(), self.grad_sz) };
      value_fn.load_diff_param(&mut self.param);
      println!("DEBUG: rank: {} |param|: {} {:.6e}", self.worker_rank, self.grad_sz, self.param.reshape(self.grad_sz).l2_norm());
    }
    self.shared_bar.wait();
  }

  pub fn step_count(&self) -> usize {
    self.step_count.load(Ordering::Acquire)
  }

  pub fn wait(&self) {
    self.shared_bar.wait();
  }

  pub fn update(&mut self) -> (f32, f32) {
    if self.num_workers > 1 {
      unsafe {
        volatile_copy_memory(self.target.as_mut_ptr(),  self.async_target.as_ptr(), self.grad_sz);
        volatile_copy_memory(self.param.as_mut_ptr(),   self.async_param.as_ptr(),  self.grad_sz);
        if self.cfg.gamma1 < 1.0 {
          volatile_copy_memory(self.gmean.as_mut_ptr(), self.async_gmean.as_ptr(),  self.grad_sz);
        }
        if self.cfg.gamma2 < 1.0 {
          volatile_copy_memory(self.gvar.as_mut_ptr(),  self.async_gvar.as_ptr(),   self.grad_sz);
        }
      }
    }

    let mut value_fn = self.value_fn.borrow_mut();

    let init_step_count = self.step_count();
    let eps_greedy = self.cfg.eps_anneal.max(self.cfg.eps_init - self.cfg.eps_anneal * (init_step_count as f32 / self.cfg.epoch_sz as f32));
    value_fn.load_diff_param(&mut self.param);
    self.base_pg.sample_epsilon_greedy(Some(self.cfg.max_horizon), Some(self.cfg.update_steps), &self.cfg.init_cfg, eps_greedy, &mut *value_fn, &mut self.rng);
    value_fn.load_diff_param(&mut self.target);
    self.base_pg.impute_final_q_values(&mut *value_fn);
    self.base_pg.fill_step_values(&self.cfg.value_cfg);

    value_fn.load_diff_param(&mut self.param);
    value_fn.reset_loss();
    value_fn.reset_grad();
    value_fn.next_iteration();
    self.cache.clear();
    let mut should_update_target = false;
    let mut iter_step_count: usize = 0;
    for (idx, episode) in self.base_pg.episodes.iter().enumerate() {
      for k in self.base_pg.ep_k_offsets[idx] .. episode.horizon() {
        let env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let env_repr_dim = env._shape3d();
        let action_value = self.base_pg.smooth_avals[idx][k];
        let act_idx = episode.steps[k].action.idx();
        let mut value_fn_item = SampleItem::new();
        value_fn_item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
        value_fn_item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        value_fn_item.kvs.insert::<SampleRegressTargetKey>(action_value);
        value_fn_item.kvs.insert::<SampleClassLabelKey>(act_idx);
        self.cache.push(value_fn_item);

        iter_step_count += 1;
        let prev_step_count = self.step_count.fetch_add(1, Ordering::AcqRel);
        if (prev_step_count + 1) % self.cfg.target_steps == 0 {
          should_update_target = true;
        }
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }

        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);
        value_fn.backward();
        self.cache.clear();

      }
    }
    if !self.cache.is_empty() {
      value_fn.load_batch(&self.cache);
      value_fn.forward(OpPhase::Learning);
      value_fn.backward();
      self.cache.clear();
    }
    value_fn.update_nondiff_param(self.iter_counter);

    let value_fn_loss = value_fn.store_loss() / iter_step_count as f32;
    value_fn.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / iter_step_count as f32);
    if let Some(grad_clip) = self.cfg.grad_clip {
      let grad_norm = self.grad.reshape(self.grad_sz).l2_norm();
      if grad_norm > grad_clip {
        self.grad.reshape_mut(self.grad_sz).scale(grad_clip / grad_norm);
      }
    }
    if self.cfg.gamma1 == 1.0 || self.iter_counter == 0 {
      self.gmean.copy_from_slice(&self.grad);
    } else {
      self.gmean.reshape_mut(self.grad_sz).average(self.cfg.gamma1, self.grad.reshape(self.grad_sz));
    }
    self.tmp_buf[ .. self.grad_sz].copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    if self.cfg.gamma2 == 1.0 || self.iter_counter == 0 {
      self.gvar.copy_from_slice(&self.tmp_buf[ .. self.grad_sz]);
    } else {
      self.gvar.reshape_mut(self.grad_sz).average(self.cfg.gamma2, self.tmp_buf.reshape(self.grad_sz));
    }
    self.tmp_buf[ .. self.grad_sz].copy_from_slice(&self.gvar);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon * self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.grad_sz).reciprocal();
    self.tmp_buf.reshape_mut(self.grad_sz).elem_mult(1.0, self.gmean.reshape(self.grad_sz));
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.tmp_buf.reshape(self.grad_sz));

    if self.num_workers > 1 {
      unsafe {
        if should_update_target {
          println!("DEBUG: updating target... rank: {} steps: {}", self.worker_rank, self.step_count());
          volatile_copy_memory(self.async_target.as_ptr() as *mut _,  self.param.as_ptr(),  self.grad_sz);
        }
        volatile_copy_memory(self.async_param.as_ptr() as *mut _,     self.param.as_ptr(),  self.grad_sz);
        if self.cfg.gamma1 < 1.0 {
          volatile_copy_memory(self.async_gmean.as_ptr() as *mut _,   self.gmean.as_ptr(),  self.grad_sz);
        }
        if self.cfg.gamma2 < 1.0 {
          volatile_copy_memory(self.async_gvar.as_ptr() as *mut _,    self.gvar.as_ptr(),   self.grad_sz);
        }
      }
    }

    self.iter_counter += 1;
    self.iter_count.fetch_add(1, Ordering::AcqRel);

    let mut avg_value = 0.0;
    //let mut avg_final_value = 0.0;
    for idx in 0 .. self.cfg.minibatch_sz {
      avg_value += self.base_pg.raw_actvals[idx][self.base_pg.ep_k_offsets[idx]];
      //avg_final_value += self.base_pg.final_values[idx].unwrap_or(0.0);
    }
    avg_value /= self.cfg.minibatch_sz as f32;
    //avg_final_value /= self.cfg.minibatch_sz as f32;
    (avg_value, value_fn_loss)
  }

  pub fn eval(&mut self, num_trials: usize) -> f32 {
    let num_minibatches = (num_trials + self.cfg.minibatch_sz - 1) / self.cfg.minibatch_sz;
    let mut value_fn = self.value_fn.borrow_mut();
    if self.num_workers > 1 {
      unsafe {
        volatile_copy_memory(self.tmp_buf.as_mut_ptr(), self.async_param.as_ptr(), self.grad_sz);
      }
    } else {
      self.tmp_buf.copy_from_slice(&self.param);
    }
    value_fn.load_diff_param(&mut self.tmp_buf);
    let mut avg_value = 0.0;
    for minibatch in 0 .. num_minibatches {
      self.eval_pg.reset(&self.cfg.init_cfg, &mut self.rng);
      self.eval_pg.sample_epsilon_greedy(Some(self.cfg.max_horizon), None, &self.cfg.init_cfg, self.cfg.eps_eval, &mut *value_fn, &mut self.rng);
      self.eval_pg.fill_step_values(&self.cfg.eval_cfg);
      for idx in 0 .. self.cfg.minibatch_sz {
        assert_eq!(0, self.eval_pg.ep_k_offsets[idx]);
        avg_value += self.eval_pg.raw_actvals[idx][0];
      }
    }
    avg_value /= (num_minibatches * self.cfg.minibatch_sz) as f32;
    avg_value
  }
}
