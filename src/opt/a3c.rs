use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};
use opt::pg_new::{BasePolicyGrad};

use densearray::prelude::*;
use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
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

#[derive(RustcEncodable)]
pub struct A3CTraceRecord {
  pub iter:         usize,
  pub step_count:   usize,
  pub avg_value:    f32,
  pub value_fn_avg_loss:    f32,
  pub elapsed:      f64,
}

#[derive(Clone, Copy, Debug)]
//pub struct AdamA3CConfig<E, V, EvalV> where E: 'static + Env, V: Value<Res=E::Response>, EvalV: Value<Res=E::Response> {
pub struct AdamA3CConfig<Init, VCfg, EvalVCfg> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub grad_clip:    Option<f32>,
  pub v_step_size:  f32,
  pub v_grad_clip:  Option<f32>,
  pub gamma1:       f32,
  pub gamma2:       f32,
  pub epsilon:      f32,
  pub max_horizon:  usize,
  pub update_steps: Option<usize>,
  pub impute_final: bool,
  pub normal_adv:   bool,
  pub init_cfg:     Init,
  pub value_cfg:    VCfg,
  pub eval_cfg:     EvalVCfg,
}

#[derive(Clone)]
pub struct AdamA3CBuilder
/*pub struct AdamA3CBuilder<E, V, EvalV>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Init: Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,*/
{
  //cfg:          AdamA3CConfig<E, V, EvalV>,
  num_workers:  usize,
  step_count:   Arc<AtomicUsize>,
  shared_iters: Arc<AtomicUsize>,
  shared_bar:   Arc<SpinBarrier>,
  async_param:  Arc<Mutex<Option<SharedMem<f32>>>>,
  async_gmean:  Arc<Mutex<Option<SharedMem<f32>>>>,
  async_gvar:   Arc<Mutex<Option<SharedMem<f32>>>>,
  async_vparam: Arc<Mutex<Option<SharedMem<f32>>>>,
  async_vgmean: Arc<Mutex<Option<SharedMem<f32>>>>,
  async_vgvar:  Arc<Mutex<Option<SharedMem<f32>>>>,
}

impl AdamA3CBuilder
/*impl<E, V, EvalV> AdamA3CBuilder<E, V, EvalV>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Init: Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,*/
{
  pub fn new(num_workers: usize) -> Self {
  //pub fn new(cfg: AdamA3CConfig<E, V, EvalV>, num_workers: usize) -> Self {
    //assert!(cfg.epsilon * cfg.epsilon > 0.0);
    AdamA3CBuilder{
      //cfg:          cfg,
      num_workers:  num_workers,
      step_count:   Arc::new(AtomicUsize::new(0)),
      shared_iters: Arc::new(AtomicUsize::new(0)),
      shared_bar:   Arc::new(SpinBarrier::new(num_workers)),
      async_param:  Arc::new(Mutex::new(None)),
      async_gmean:  Arc::new(Mutex::new(None)),
      async_gvar:   Arc::new(Mutex::new(None)),
      async_vparam: Arc::new(Mutex::new(None)),
      async_vgmean: Arc::new(Mutex::new(None)),
      async_vgvar:  Arc::new(Mutex::new(None)),
    }
  }

  //pub fn into_worker<E, V, EvalV, Policy, ValueFn>(self, cfg: AdamA3CConfig<E, V, EvalV>, worker_rank: usize, policy: Rc<RefCell<Policy>>, value_fn: Rc<RefCell<ValueFn>>) -> AdamA3CWorker<E, V, EvalV, Policy, ValueFn>
  pub fn into_worker<E, V, EvalV, Policy, ValueFn>(self, cfg: AdamA3CConfig<E::Init, V::Cfg, EvalV::Cfg>, worker_rank: usize, policy: Rc<RefCell<Policy>>, value_fn: Rc<RefCell<ValueFn>>) -> AdamA3CWorker<E, V, EvalV, Policy, ValueFn>
  where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
        E::Action: DiscreteAction,
        V: Value<Res=E::Response>,
        EvalV: Value<Res=E::Response>,
        Policy: DiffLoss<SampleItem, IoBuf=[f32]>,
        ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
  {
    let batch_sz = cfg.batch_sz;
    let minibatch_sz = cfg.minibatch_sz;
    let max_horizon = cfg.max_horizon;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let base_pg = BasePolicyGrad::new(minibatch_sz, &cfg.init_cfg, &mut rng);
    let eval_pg = BasePolicyGrad::new(minibatch_sz, &cfg.init_cfg, &mut rng);
    let pgrad_sz = policy.borrow_mut().diff_param_sz();
    let vgrad_sz = value_fn.borrow_mut().diff_param_sz();
    let mut param = Vec::with_capacity(pgrad_sz);
    param.resize(pgrad_sz, 0.0);
    let mut grad = Vec::with_capacity(pgrad_sz);
    grad.resize(pgrad_sz, 0.0);
    let mut gmean = Vec::with_capacity(pgrad_sz);
    gmean.resize(pgrad_sz, 0.0);
    let mut gvar = Vec::with_capacity(pgrad_sz);
    gvar.resize(pgrad_sz, 0.0);
    let mut vparam = Vec::with_capacity(vgrad_sz);
    vparam.resize(vgrad_sz, 0.0);
    let mut vgrad = Vec::with_capacity(vgrad_sz);
    vgrad.resize(vgrad_sz, 0.0);
    let mut vgmean = Vec::with_capacity(vgrad_sz);
    vgmean.resize(vgrad_sz, 0.0);
    let mut vgvar = Vec::with_capacity(vgrad_sz);
    vgvar.resize(vgrad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(max(pgrad_sz, vgrad_sz));
    tmp_buf.resize(max(pgrad_sz, vgrad_sz), 0.0);
    if worker_rank == 0 {
      let mut param = Vec::with_capacity(pgrad_sz);
      param.resize(pgrad_sz, 0.0);
      let mut gmean = Vec::with_capacity(pgrad_sz);
      gmean.resize(pgrad_sz, 0.0);
      let mut gvar = Vec::with_capacity(pgrad_sz);
      gvar.resize(pgrad_sz, 0.0);
      let mut vparam = Vec::with_capacity(vgrad_sz);
      vparam.resize(vgrad_sz, 0.0);
      let mut vgmean = Vec::with_capacity(vgrad_sz);
      vgmean.resize(vgrad_sz, 0.0);
      let mut vgvar = Vec::with_capacity(vgrad_sz);
      vgvar.resize(vgrad_sz, 0.0);
      let mut async_param = self.async_param.lock().unwrap();
      let mut async_gmean = self.async_gmean.lock().unwrap();
      let mut async_gvar = self.async_gvar.lock().unwrap();
      let mut async_vparam = self.async_vparam.lock().unwrap();
      let mut async_vgmean = self.async_vgmean.lock().unwrap();
      let mut async_vgvar = self.async_vgvar.lock().unwrap();
      *async_param = Some(SharedMem::new(param));
      *async_gmean = Some(SharedMem::new(gmean));
      *async_gvar = Some(SharedMem::new(gvar));
      *async_vparam = Some(SharedMem::new(vparam));
      *async_vgmean = Some(SharedMem::new(vgmean));
      *async_vgvar = Some(SharedMem::new(vgvar));
    }
    self.shared_bar.wait();
    let worker = AdamA3CWorker{
      cfg:      cfg,
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      grad_sz:  pgrad_sz,
      vgrad_sz: vgrad_sz,
      step_count:   self.step_count,
      shared_iters: self.shared_iters,
      shared_bar:   self.shared_bar,
      iter_counter: 0,
      rng:      rng,
      base_pg:  base_pg,
      eval_pg:  eval_pg,
      policy:   policy,
      value_fn: value_fn,
      cache:    Vec::with_capacity(batch_sz),
      vcache:   Vec::with_capacity(batch_sz),
      async_param:  self.async_param.lock().unwrap().as_ref().unwrap().as_slice(),
      async_gmean:  self.async_gmean.lock().unwrap().as_ref().unwrap().as_slice(),
      async_gvar:   self.async_gvar.lock().unwrap().as_ref().unwrap().as_slice(),
      param:    param,
      grad:     grad,
      gmean:    gmean,
      gvar:     gvar,
      async_vparam: self.async_vparam.lock().unwrap().as_ref().unwrap().as_slice(),
      async_vgmean: self.async_vgmean.lock().unwrap().as_ref().unwrap().as_slice(),
      async_vgvar:  self.async_vgvar.lock().unwrap().as_ref().unwrap().as_slice(),
      vparam:   vparam,
      vgrad:    vgrad,
      vgmean:   vgmean,
      vgvar:    vgvar,
      tmp_buf:  tmp_buf,
    };
    worker.shared_bar.wait();
    worker
  }
}

pub struct AdamA3CWorker<E, V, EvalV, Policy, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  cfg:      AdamA3CConfig<E::Init, V::Cfg, EvalV::Cfg>,
  worker_rank:  usize,
  num_workers:  usize,
  grad_sz:  usize,
  vgrad_sz: usize,
  step_count:   Arc<AtomicUsize>,
  shared_iters: Arc<AtomicUsize>,
  shared_bar:   Arc<SpinBarrier>,
  iter_counter: usize,
  rng:      Xorshiftplus128Rng,
  base_pg:  BasePolicyGrad<E, V>,
  eval_pg:  BasePolicyGrad<E, EvalV>,
  policy:   Rc<RefCell<Policy>>,
  value_fn: Rc<RefCell<ValueFn>>,
  cache:    Vec<SampleItem>,
  vcache:   Vec<SampleItem>,
  async_param:  SharedSlice<f32>,
  async_gmean:  SharedSlice<f32>,
  async_gvar:   SharedSlice<f32>,
  param:    Vec<f32>,
  grad:     Vec<f32>,
  gmean:    Vec<f32>,
  gvar:     Vec<f32>,
  async_vparam: SharedSlice<f32>,
  async_vgmean: SharedSlice<f32>,
  async_vgvar:  SharedSlice<f32>,
  vparam:   Vec<f32>,
  vgrad:    Vec<f32>,
  vgmean:   Vec<f32>,
  vgvar:    Vec<f32>,
  tmp_buf:  Vec<f32>,
}

impl<E, V, EvalV, Policy, ValueFn> AdamA3CWorker<E, V, EvalV, Policy, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      Policy: DiffLoss<SampleItem, IoBuf=[f32]>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  pub fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    if self.worker_rank == 0 {
      let mut policy = self.policy.borrow_mut();
      policy.init_param(rng);
      policy.store_diff_param(&mut self.param);
      unsafe { volatile_copy_memory(self.async_param.as_ptr() as *mut _, self.param.as_ptr(), self.grad_sz) };
      println!("DEBUG: |param|: {} {:.6e}", self.grad_sz, self.param.reshape(self.grad_sz).l2_norm());

      let mut value_fn = self.value_fn.borrow_mut();
      value_fn.init_param(rng);
      value_fn.store_diff_param(&mut self.vparam);
      unsafe { volatile_copy_memory(self.async_vparam.as_ptr() as *mut _, self.vparam.as_ptr(), self.vgrad_sz) };
      println!("DEBUG: |vparam|: {} {:.6e}", self.vgrad_sz, self.vparam.reshape(self.vgrad_sz).l2_norm());
    }
    self.shared_bar.wait();
    if self.worker_rank != 0 {
      let mut policy = self.policy.borrow_mut();
      //self.param.copy_from_slice(&self.async_param);
      unsafe { volatile_copy_memory(self.param.as_mut_ptr(), self.async_param.as_ptr(), self.grad_sz) };
      policy.load_diff_param(&mut self.param);

      let mut value_fn = self.value_fn.borrow_mut();
      //self.vparam.copy_from_slice(&self.async_vparam);
      unsafe { volatile_copy_memory(self.vparam.as_mut_ptr(), self.async_vparam.as_ptr(), self.vgrad_sz) };
      value_fn.load_diff_param(&mut self.vparam);
    }
    self.shared_bar.wait();
  }

  pub fn step_count(&self) -> usize {
    self.step_count.load(Ordering::Acquire)
  }

  pub fn update(&mut self) -> (f32, f32, f32, f32) {
    if self.num_workers > 1 {
      unsafe {
        volatile_copy_memory(self.param.as_mut_ptr(),       self.async_param.as_ptr(),  self.grad_sz);
        if self.cfg.gamma1 < 1.0 {
          volatile_copy_memory(self.gmean.as_mut_ptr(),     self.async_gmean.as_ptr(),  self.grad_sz);
        }
        if self.cfg.gamma2 < 1.0 {
          volatile_copy_memory(self.gvar.as_mut_ptr(),      self.async_gvar.as_ptr(),   self.grad_sz);
        }
        volatile_copy_memory(self.vparam.as_mut_ptr(),      self.async_vparam.as_ptr(), self.vgrad_sz);
        if self.cfg.gamma1 < 1.0 {
          volatile_copy_memory(self.vgmean.as_mut_ptr(),    self.async_vgmean.as_ptr(), self.vgrad_sz);
        }
        if self.cfg.gamma2 < 1.0 {
          volatile_copy_memory(self.vgvar.as_mut_ptr(),     self.async_vgvar.as_ptr(),  self.vgrad_sz);
        }
      }
    }
    let mut policy = self.policy.borrow_mut();
    let mut value_fn = self.value_fn.borrow_mut();
    policy.load_diff_param(&mut self.param);
    value_fn.load_diff_param(&mut self.vparam);

    self.base_pg.sample_steps(Some(self.cfg.max_horizon), self.cfg.update_steps, &self.cfg.init_cfg, &mut *policy, &mut self.rng);
    self.base_pg.impute_baselines(self.cfg.update_steps, &mut *value_fn);
    if self.cfg.impute_final {
      self.base_pg.impute_final_values(&mut *value_fn);
    }
    self.base_pg.fill_step_values(&self.cfg.value_cfg);
    if self.cfg.normal_adv {
      // FIXME(20161024): normalize the per-episode advantages.
      unimplemented!();
    }

    policy.reset_loss();
    policy.reset_grad();
    policy.next_iteration();
    value_fn.reset_loss();
    value_fn.reset_grad();
    value_fn.next_iteration();
    self.cache.clear();
    self.vcache.clear();
    let mut steps_count = 0;
    //let mut tmp_baselines = vec![];
    for (idx, episode) in self.base_pg.episodes.iter().enumerate() {
      for k in self.base_pg.ep_k_offsets[idx] .. episode.horizon() {
        let mut policy_item = SampleItem::new();
        let mut value_fn_item = SampleItem::new();
        let env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let env_repr_dim = env._shape3d();
        let action_value = self.base_pg.raw_actvals[idx][k];
        let smoothed_action_value = self.base_pg.smooth_avals[idx][k];
        let baseline_value = self.base_pg.baseline_val[idx][k];
        /*if k == 0 {
          tmp_baselines.push(baseline_value);
        }*/
        policy_item.kvs.insert::<SampleExtractInputKey<[f32]>>(env.clone());
        policy_item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        policy_item.kvs.insert::<SampleClassLabelKey>(episode.steps[k].action.idx());
        // FIXME(20161023): should use the normalized smoothed action values,
        // as long as it is safe/stable to do so.
        //policy_item.kvs.insert::<SampleWeightKey>(action_value - baseline_value);
        policy_item.kvs.insert::<SampleWeightKey>(smoothed_action_value - baseline_value);
        value_fn_item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
        value_fn_item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        value_fn_item.kvs.insert::<SampleRegressTargetKey>(action_value);
        self.cache.push(policy_item);
        self.vcache.push(value_fn_item);
        steps_count += 1;
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }
        assert_eq!(self.cache.len(), self.vcache.len());
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
    self.step_count.fetch_add(steps_count, Ordering::AcqRel);
    //println!("DEBUG: baselines[0]: {:?}", &tmp_baselines);
    //println!("DEBUG: baselines[H]: {:?}", &self.base_pg.final_values);
    if !self.cache.is_empty() {
      assert_eq!(self.cache.len(), self.vcache.len());
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

    let gamma1_scale = if self.cfg.gamma1 < 1.0 {
      1.0 / (1.0 - (1.0 - self.cfg.gamma1).powi((self.iter_counter + 1) as i32))
    } else {
      1.0
    };
    let gamma2_scale = if self.cfg.gamma2 < 1.0 {
      1.0 / (1.0 - (1.0 - self.cfg.gamma2).powi((self.iter_counter + 1) as i32))
    } else {
      1.0
    };

    let policy_loss = policy.store_loss() / self.cfg.minibatch_sz as f32;
    policy.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    if let Some(grad_clip) = self.cfg.grad_clip {
      let grad_norm = self.grad.reshape(self.grad_sz).l2_norm();
      if grad_norm > grad_clip {
        self.grad.reshape_mut(self.grad_sz).scale(grad_clip / grad_norm);
      }
    }
    if self.cfg.gamma1 < 1.0 {
      self.gmean.reshape_mut(self.grad_sz).average(self.cfg.gamma1, self.grad.reshape(self.grad_sz));
    } else {
      self.gmean.copy_from_slice(&self.grad);
    }
    self.tmp_buf[ .. self.grad_sz].copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    if self.cfg.gamma2 < 1.0 {
      self.gvar.reshape_mut(self.grad_sz).average(self.cfg.gamma2, self.tmp_buf.reshape(self.grad_sz));
    } else {
      self.gvar.copy_from_slice(&self.tmp_buf[ .. self.grad_sz]);
    }
    self.tmp_buf[ .. self.grad_sz].copy_from_slice(&self.gvar);
    self.tmp_buf.reshape_mut(self.grad_sz).scale(gamma2_scale);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon * self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.grad_sz).reciprocal();
    self.tmp_buf.reshape_mut(self.grad_sz).elem_mult(self.gmean.reshape(self.grad_sz));
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size * gamma1_scale, self.tmp_buf.reshape(self.grad_sz));

    let value_fn_loss = value_fn.store_loss() / steps_count as f32;
    value_fn.store_grad(&mut self.vgrad);
    self.vgrad.reshape_mut(self.vgrad_sz).scale(1.0 / steps_count as f32);
    if let Some(vgrad_clip) = self.cfg.v_grad_clip {
      let vgrad_norm = self.vgrad.reshape(self.vgrad_sz).l2_norm();
      if vgrad_norm > vgrad_clip {
        self.vgrad.reshape_mut(self.vgrad_sz).scale(vgrad_clip / vgrad_norm);
      }
    }
    if self.cfg.gamma1 < 1.0 {
      self.vgmean.reshape_mut(self.vgrad_sz).average(self.cfg.gamma1, self.vgrad.reshape(self.vgrad_sz));
    } else {
      self.vgmean.copy_from_slice(&self.vgrad);
    }
    self.tmp_buf[ .. self.vgrad_sz].copy_from_slice(&self.vgrad);
    self.tmp_buf.reshape_mut(self.vgrad_sz).square();
    if self.cfg.gamma2 < 1.0 {
      self.vgvar.reshape_mut(self.vgrad_sz).average(self.cfg.gamma2, self.tmp_buf.reshape(self.vgrad_sz));
    } else {
      self.vgvar.copy_from_slice(&self.tmp_buf[ .. self.vgrad_sz]);
    }
    self.tmp_buf[ .. self.vgrad_sz].copy_from_slice(&self.vgvar);
    self.tmp_buf.reshape_mut(self.vgrad_sz).scale(gamma2_scale);
    self.tmp_buf.reshape_mut(self.vgrad_sz).add_scalar(self.cfg.epsilon * self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.vgrad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.vgrad_sz).reciprocal();
    self.tmp_buf.reshape_mut(self.vgrad_sz).elem_mult(self.vgmean.reshape(self.vgrad_sz));
    self.vparam.reshape_mut(self.vgrad_sz).add(-self.cfg.v_step_size * gamma1_scale, self.tmp_buf.reshape(self.vgrad_sz));

    if self.num_workers > 1 {
      unsafe {
        volatile_copy_memory(self.async_param.as_ptr() as *mut _,       self.param.as_ptr(),    self.grad_sz);
        if self.cfg.gamma1 < 1.0 {
          volatile_copy_memory(self.async_gmean.as_ptr() as *mut _,     self.gmean.as_ptr(),    self.grad_sz);
        }
        if self.cfg.gamma2 < 1.0 {
          volatile_copy_memory(self.async_gvar.as_ptr() as *mut _,      self.gvar.as_ptr(),     self.grad_sz);
        }
        volatile_copy_memory(self.async_vparam.as_ptr() as *mut _,      self.vparam.as_ptr(),   self.vgrad_sz);
        if self.cfg.gamma1 < 1.0 {
          volatile_copy_memory(self.async_vgmean.as_ptr() as *mut _,    self.vgmean.as_ptr(),   self.vgrad_sz);
        }
        if self.cfg.gamma2 < 1.0 {
          volatile_copy_memory(self.async_vgvar.as_ptr() as *mut _,     self.vgvar.as_ptr(),    self.vgrad_sz);
        }
      }
    }

    self.shared_iters.fetch_add(1, Ordering::SeqCst);
    self.iter_counter += 1;

    let mut avg_value = 0.0;
    let mut avg_final_value = 0.0;
    for idx in 0 .. self.cfg.minibatch_sz {
      avg_value += self.base_pg.raw_actvals[idx][self.base_pg.ep_k_offsets[idx]];
      avg_final_value += self.base_pg.final_values[idx].unwrap_or(0.0);
    }
    avg_value /= self.cfg.minibatch_sz as f32;
    avg_final_value /= self.cfg.minibatch_sz as f32;
    (avg_value, avg_final_value, policy_loss, value_fn_loss)
  }

  pub fn eval(&mut self, num_trials: usize) -> f32 {
    let num_minibatches = (num_trials + self.cfg.minibatch_sz - 1) / self.cfg.minibatch_sz;
    let mut policy = self.policy.borrow_mut();
    /*if self.num_workers > 1 {
      unsafe {
        volatile_copy_memory(self.tmp_buf.as_mut_ptr(), self.async_param.as_ptr(), self.grad_sz);
      }
    } else {
      self.tmp_buf.copy_from_slice(&self.param);
    }*/
    policy.load_diff_param(&mut self.param);
    let mut avg_value = 0.0;
    for minibatch in 0 .. num_minibatches {
      self.eval_pg.reset(&self.cfg.init_cfg, &mut self.rng);
      self.eval_pg.sample_steps(Some(self.cfg.max_horizon), None, &self.cfg.init_cfg, &mut *policy, &mut self.rng);
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
