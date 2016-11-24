use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};
use features::{EnvObsRepr, BeliefState};
use kernels::*;
use opt::pg_new::{BasePolicyGrad};
use replay::{ReplayEntry, LinearReplayCache};

use densearray::prelude::*;
use float::ord::{F32InfNan};
use iter_utils::{argmax};
use operator::prelude::*;
use rng::{RngState};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice, SharedMem, SharedSlice};
use sharedmem::sync::{SpinBarrier};

use rand::{Rng, thread_rng};
use std::f32;
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::fs::{File, create_dir_all};
use std::intrinsics::{volatile_copy_memory};
use std::io::{Read, Write, BufWriter};
use std::marker::{PhantomData};
use std::ops::{Deref};
use std::path::{PathBuf};
use std::rc::{Rc};
use std::slice::{from_raw_parts_mut};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant};

fn jenkins_hash_obs<F>(key: &F) -> u64 where F: Deref<Target=[u8]> {
  let mut h = 0;
  let buf: &[u8] = &*key;
  for &u in buf {
    let x = u as u64;
    h += x;
    h += (h << 10);
    h ^= (h >> 6);
  }
  h += (h << 3);
  h ^= (h >> 11);
  h += (h << 15);
  h
}

fn jenkins_hash_state<F>(key: &BeliefState<F>) -> u64 where F: Deref<Target=[u8]> {
  let mut h = 0;
  for obs in key.obs_reprs.iter() {
    let buf: &[u8] = &*(*obs);
    for &u in buf {
      let x = u as u64;
      h += x;
      h += (h << 10);
      h ^= (h >> 6);
    }
  }
  h += (h << 3);
  h ^= (h >> 11);
  h += (h << 15);
  h
}

fn jenkins_hash_u64(key: &[u64]) -> u64 {
  let mut h = 0;
  for &x in key {
    h += x;
    h += (h << 10);
    h ^= (h >> 6);
  }
  h += (h << 3);
  h ^= (h >> 11);
  h += (h << 15);
  h
}

#[derive(Clone, Copy, Debug, RustcEncodable)]
pub struct DiffQRecord {
  pub iter:         usize,
  pub step:         usize,
  pub avg_episodes: usize,
  pub avg_value:    f32,
  pub min_value:    f32,
  pub max_value:    f32,
  pub elapsed:      f64,
}

#[derive(Debug)]
pub struct RmspropDiffQConfig<Init, VCfg, EvalVCfg> {
  //pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub history_len:  usize,
  pub repeat_noop:  Option<(usize, u32)>,
  pub replay_sz:    usize,
  pub replay_init:  usize,
  pub update_steps: usize,
  pub target_steps: usize,
  pub eval_horizon: usize,
  pub exp_init:     f64,
  pub exp_anneal:   f64,
  pub exp_eval:     f64,
  pub step_size:    f32,
  pub momentum:     f32,
  pub rms_decay:    f32,
  pub epsilon:      f32,
  pub init_cfg:     Init,
  pub value_cfg:    VCfg,
  pub eval_cfg:     EvalVCfg,
}

pub struct RmspropDiffQWorker<E, F, V, EvalV, ValueFn>
where E: 'static + Env + EnvObsRepr<F>,
      F: 'static + SampleExtractInput<[f32]> + SampleInputShape<(usize, usize, usize)> + Deref<Target=[u8]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      ValueFn: DiffLoss<SampleItem, [f32]>,
{
  cfg:          RmspropDiffQConfig<E::Init, V::Cfg, EvalV::Cfg>,
  grad_sz:      usize,
  iter_count:   usize,
  epoch_count:  usize,
  ep_count:     usize,
  step_count:   usize,
  sim_res:      Vec<E::Response>,
  epoch_ep_cnt: usize,
  avg_value:    f32,
  min_value:    f32,
  max_value:    f32,
  rng:          Xorshiftplus128Rng,
  replay_cache: LinearReplayCache<F, E::Action, E::Response>,
  env:          Rc<E>,
  belief_state: BeliefState<F>,
  eval_env:     Rc<E>,
  eval_belief:  BeliefState<F>,
  value_fn:     Rc<RefCell<ValueFn>>,
  target_fn:    Rc<RefCell<ValueFn>>,
  samples:      Vec<ReplayEntry<F, E::Action, E::Response>>,
  target_maxs:  Vec<usize>,
  target_vals:  Vec<f32>,
  cache:        Vec<SampleItem>,
  target:       Vec<f32>,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  update_acc:   Vec<f32>,
  grad_var_acc: Vec<f32>,
  tmp_buf:      Vec<f32>,
  trace_file:   BufWriter<File>,
}

impl<E, F, V, EvalV, ValueFn> RmspropDiffQWorker<E, F, V, EvalV, ValueFn>
//where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
where E: 'static + Env + EnvObsRepr<F>,
      F: 'static + SampleExtractInput<[f32]> + SampleInputShape<(usize, usize, usize)> + Deref<Target=[u8]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      ValueFn: DiffLoss<SampleItem, [f32]>,
{
  pub fn new(cfg: RmspropDiffQConfig<E::Init, V::Cfg, EvalV::Cfg>, value_fn: Rc<RefCell<ValueFn>>, target_fn: Rc<RefCell<ValueFn>>) -> Self {
    let grad_sz = value_fn.borrow_mut().diff_param_sz();
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    //rng.set_state(&[1234_5678, 1234_5678]); // FIXME(20161029): for debugging.
    let replay_cache = LinearReplayCache::new(cfg.history_len, E::_obs_shape3d(), cfg.replay_sz);
    let env = Rc::new(E::default());
    let belief_state = BeliefState::new(Some(cfg.history_len), E::_obs_shape3d());
    let eval_env = Rc::new(E::default());
    let eval_belief = BeliefState::new(Some(cfg.history_len), E::_obs_shape3d());
    let samples = Vec::with_capacity(cfg.minibatch_sz);
    let target_maxs = Vec::with_capacity(cfg.minibatch_sz);
    let target_vals = Vec::with_capacity(cfg.minibatch_sz);
    let cache = Vec::with_capacity(cfg.minibatch_sz);
    let mut target = Vec::with_capacity(grad_sz);
    target.resize(grad_sz, 0.0);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut update_acc = Vec::with_capacity(grad_sz);
    update_acc.resize(grad_sz, 0.0);
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    RmspropDiffQWorker{
      cfg:          cfg,
      grad_sz:      grad_sz,
      iter_count:   0,
      epoch_count:  0,
      ep_count:     0,
      step_count:   0,
      sim_res:      vec![],
      epoch_ep_cnt: 0,
      avg_value:    0.0,
      min_value:    f32::INFINITY,
      max_value:    -f32::INFINITY,
      rng:          rng,
      replay_cache: replay_cache,
      env:          env,
      belief_state: belief_state,
      eval_env:     eval_env,
      eval_belief:  eval_belief,
      value_fn:     value_fn,
      target_fn:    target_fn,
      samples:      samples,
      target_maxs:  target_maxs,
      target_vals:  target_vals,
      cache:        cache,
      target:       target,
      param:        param,
      grad:         grad,
      update_acc:   update_acc,
      grad_var_acc: grad_var_acc,
      tmp_buf:      tmp_buf,
      trace_file:   BufWriter::new(File::create("trace.log").unwrap()),
    }
  }

  fn reset_env(&mut self) {
    self.env.reset(&self.cfg.init_cfg, &mut self.rng);
    //self.belief_state.reset();
    if let Some((max_reps, noop_idx)) = self.cfg.repeat_noop {
      let nreps = self.rng.gen_range(self.cfg.history_len + 1, max(self.cfg.history_len + 1, max_reps) + 1);
      //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
      //writeln!(&mut self.trace_file, "restart,{}", nreps).unwrap();
      self.trace_file.flush().unwrap();
      for _ in 0 .. nreps {
        let action = <E::Action as DiscreteAction>::from_idx(noop_idx);
        let _ = self.env.step(&action).unwrap();
        let next_obs = self.env.observe(&mut self.rng);
        self.belief_state.push(Rc::new(next_obs.clone()));
      }
    }
  }

  pub fn deterministic_test(&mut self, rng: &mut Xorshiftplus128Rng) {
    {
      let mut value_fn = self.value_fn.borrow_mut();
      let mut target_fn = self.target_fn.borrow_mut();

      let mut param_file = File::open("param.bin").unwrap();
      let mut raw_param_buf = vec![];
      param_file.read_to_end(&mut raw_param_buf).unwrap();
      let mut param_buf = unsafe { from_raw_parts_mut(raw_param_buf.as_mut_ptr() as *mut _, raw_param_buf.len() / 4) };
      assert_eq!(self.grad_sz, param_buf.len());
      value_fn.load_diff_param(&mut param_buf);
    }

    let action_dim = <E::Action as Action>::dim();
    println!("DEBUG: dq: deterministic testing...");
    let mut trace_actions = Vec::with_capacity(10_000);
    //self.env.reset(&self.cfg.init_cfg, &mut self.rng);
    self.reset_env();
    for _ in 0 .. 10_000 {
      if self.env.is_terminal() {
        self.epoch_ep_cnt += 1;
        //self.env.reset(&self.cfg.init_cfg, &mut self.rng);
        self.reset_env();
        let mut v = <EvalV as Value>::from_scalar(0.0, self.cfg.eval_cfg);
        for &r in self.sim_res.iter().rev() {
          v.lreduce(r);
        }
        //println!("DEBUG: dq: init: {} {:.3}", self.epoch_ep_cnt, v.to_scalar());
        self.sim_res.clear();
        self.avg_value += (v.to_scalar() - self.avg_value) / self.epoch_ep_cnt as f32;
        self.min_value = self.min_value.min(v.to_scalar());
        self.max_value = self.max_value.max(v.to_scalar());
      }
      assert!(!self.env.is_terminal());
      let act_idx = {
        let mut item = SampleItem::new();
        //let env = self.env.clone();
        //let env_repr_dim = env._shape3d();
        let obs = Rc::new(self.belief_state.clone());
        let obs_dim = obs._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(obs.clone());
        item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(obs.clone());
        item.kvs.insert::<SampleInputShape3dKey>(obs_dim);
        self.cache.clear();
        self.cache.push(item);
        let mut value_fn = self.value_fn.borrow_mut();
        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);
        let argmax_k = argmax(value_fn._get_pred()[ .. action_dim].iter().map(|&v| F32InfNan(v))).unwrap();
        //println!("DEBUG: dq: argmax: {} qvalues: {:?}", argmax_k, &value_fn._get_pred()[ .. action_dim]);
        trace_actions.push(argmax_k);
        argmax_k as u32
      };
      //let prev_env = Rc::new((*self.env).clone());
      let action = <E::Action as DiscreteAction>::from_idx(act_idx);
      let res = self.env.step(&action).unwrap();
      //let next_env = Rc::new((*self.env).clone());
      let next_obs = self.env.observe(&mut self.rng);
      let terminal = self.env.is_terminal();
      self.belief_state.push(Rc::new(next_obs.clone()));
      //self.replay_cache.insert(prev_env, action, res, next_env);
      self.replay_cache.insert(action, res, Rc::new(next_obs), terminal);
      self.sim_res.push(res.unwrap());
    }
    // FIXME(20161028): dump actions trace to file.
    {
      let mut trace_file = BufWriter::new(File::create("deterministic_test.log").unwrap());
      for &act_idx in trace_actions.iter() {
        writeln!(&mut trace_file, "{}", act_idx).unwrap();
      }
    }
    println!("DEBUG: dq: test: {} {:.3} {:.3} {:.3}",
        self.epoch_ep_cnt, self.avg_value, self.min_value, self.max_value);
    self.epoch_ep_cnt = 0;
    self.avg_value = 0.0;
    self.min_value = f32::INFINITY;
    self.max_value = -f32::INFINITY;
  }

  pub fn init(&mut self, rng: &mut Xorshiftplus128Rng) {
    let action_dim = <E::Action as Action>::dim();
    println!("DEBUG: dq: initializing replay memory...");
    self.reset_env();
    for step_nr in 0 .. self.cfg.replay_init {
      //assert!(!self.env.is_terminal());
      let exp_rate = 1.0;
      //let _ = self.rng._random();
      let u: f32 = self.rng.gen();
      //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
      //writeln!(&mut self.trace_file, "exp,{:.6},{:.6}", u, exp_rate).unwrap();
      self.trace_file.flush().unwrap();
      //let act_idx = self.rng._randint(0, action_dim - 1) as u32;
      let act_idx = self.rng.gen_range(0, action_dim) as u32;
      let action = <E::Action as DiscreteAction>::from_idx(act_idx);
      let res = self.env.step(&action).unwrap();
      let next_obs = self.env.observe(&mut self.rng);
      let terminal = self.env.is_terminal();
      let obs_hash = jenkins_hash_obs(&next_obs);
      //println!("DEBUG: dq: random step: obs hash: {}", obs_hash);
      //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
      //writeln!(&mut self.trace_file, "step,random,{}", act_idx).unwrap();
      //writeln!(&mut self.trace_file, "step,random,hash,{}", obs_hash).unwrap();
      self.belief_state.push(Rc::new(next_obs.clone()));
      self.replay_cache.insert(action, res, Rc::new(next_obs), terminal);
      self.sim_res.push(res.unwrap());
      self.trace_file.flush().unwrap();

      if self.env.is_terminal() {
        self.epoch_ep_cnt += 1;
        self.reset_env();
        let mut v = <EvalV as Value>::from_scalar(0.0, self.cfg.eval_cfg);
        for &r in self.sim_res.iter().rev() {
          v.lreduce(r);
        }
        self.sim_res.clear();
        self.avg_value += (v.to_scalar() - self.avg_value) / self.epoch_ep_cnt as f32;
        self.min_value = self.min_value.min(v.to_scalar());
        self.max_value = self.max_value.max(v.to_scalar());
        println!("DEBUG: dq: stats: {} {} {:.3} {:.3} {:.3}",
            self.epoch_ep_cnt, step_nr + 1, self.avg_value, self.min_value, self.max_value);
      }
    }
    println!("DEBUG: dq: replay init: {} {:.3} {:.3} {:.3}",
        self.epoch_ep_cnt, self.avg_value, self.min_value, self.max_value);
    self.epoch_ep_cnt = 0;
    self.avg_value = 0.0;
    self.min_value = f32::INFINITY;
    self.max_value = -f32::INFINITY;

    let mut value_fn = self.value_fn.borrow_mut();
    let mut target_fn = self.target_fn.borrow_mut();

    /*//let mut param_file = File::open(&format!("saved_record/pong_init_param.bin")).unwrap();
    let mut param_file = File::open(&format!("init_param.bin")).unwrap();
    let mut raw_param_buf = vec![];
    param_file.read_to_end(&mut raw_param_buf).unwrap();
    let mut param_buf = unsafe { from_raw_parts_mut(raw_param_buf.as_mut_ptr() as *mut _, raw_param_buf.len() / 4) };
    assert_eq!(self.grad_sz, param_buf.len());
    value_fn.load_diff_param(&mut param_buf);
    self.param.copy_from_slice(&param_buf);*/

    value_fn.init_param(rng);
    value_fn.store_diff_param(&mut self.param);

    /*//let mut param_file = File::open(&format!("saved_record/pong_init_target.bin")).unwrap();
    let mut param_file = File::open(&format!("init_target_param.bin")).unwrap();
    let mut raw_param_buf = vec![];
    param_file.read_to_end(&mut raw_param_buf).unwrap();
    let mut param_buf = unsafe { from_raw_parts_mut(raw_param_buf.as_mut_ptr() as *mut _, raw_param_buf.len() / 4) };
    assert_eq!(self.grad_sz, param_buf.len());
    target_fn.load_diff_param(&mut param_buf);
    self.target.copy_from_slice(&param_buf);*/

    target_fn.init_param(rng);
    target_fn.store_diff_param(&mut self.target);

    //self.target.copy_from_slice(&self.param);
    //target_fn.load_diff_param(&mut self.target);
  }

  pub fn step_count(&self) -> usize {
    self.step_count
  }

  pub fn iter_count(&self) -> usize {
    self.iter_count
  }

  pub fn next_epoch(&mut self) {
    //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
    //writeln!(&mut self.trace_file, "train,epoch,{}", self.epoch_count).unwrap();
    self.trace_file.flush().unwrap();
  }

  pub fn update(&mut self) {
    let action_dim = <E::Action as Action>::dim();

    {
      let mut value_fn = self.value_fn.borrow_mut();
      //let mut target_fn = self.target_fn.borrow_mut();

      /*let step_prefix = PathBuf::from(&format!("tmp_debug/steps_{}", self.epoch_count));
      let minibatch_prefix = PathBuf::from(&format!("tmp_debug/minibatches_{}", self.epoch_count));
      if self.epoch_ep_cnt <= 2 {
        create_dir_all(&step_prefix).ok();
        create_dir_all(&minibatch_prefix).ok();
      }*/

      value_fn.load_diff_param(&mut self.param);
    }

    let mut step_res = vec![];
    let mut step_nrand = 0;
    let mut step_argmax = vec![];
    let mut step_exp = 0.0;
    for step_nr in 0 .. self.cfg.update_steps {
      //assert!(!self.env.is_terminal());
      let t = 1.0_f64.min(0.0_f64.max(self.step_count as f64 / self.cfg.replay_sz as f64));
      let exp_rate = self.cfg.exp_init * (1.0 - t) + self.cfg.exp_anneal * t;
      step_exp = exp_rate;
      //let u = self.rng._random();
      let u: f32 = self.rng.gen();
      //println!("DEBUG: dq: step: u: {:.6} exp rate: {:.6}", u, exp_rate);
      //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
      writeln!(&mut self.trace_file, "exp,{:.6},{:.6}", u, exp_rate).unwrap();
      self.trace_file.flush().unwrap();
      let act_idx = if (u as f64) < exp_rate {
        //let uniform_act_idx = self.rng._randint(0, action_dim - 1) as u32;
        let uniform_act_idx = self.rng.gen_range(0, action_dim) as u32;
        step_nrand += 1;
        //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
        //writeln!(&mut self.trace_file, "step,train_random,{}", uniform_act_idx).unwrap();
        self.trace_file.flush().unwrap();
        uniform_act_idx
      } else {
        let mut item = SampleItem::new();
        let obs = Rc::new(self.belief_state.clone());
        let obs_dim = obs._shape3d();
        //println!("DEBUG: update (step): obs dim: {:?}", obs_dim);
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(obs.clone());
        item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(obs.clone());
        item.kvs.insert::<SampleInputShape3dKey>(obs_dim);
        self.cache.clear();
        self.cache.push(item);
        let mut value_fn = self.value_fn.borrow_mut();
        value_fn.reset_loss();
        value_fn.next_iteration();
        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);
        let argmax_k = argmax(value_fn._get_pred()[ .. action_dim].iter().map(|&v| F32InfNan(v))).unwrap();
        step_argmax.push(argmax_k);
        let mut qvalues_s = String::new();
        for k in 0 .. action_dim {
          qvalues_s.push_str(&format!("{:.6}", value_fn._get_pred()[k]));
          if k < action_dim-1 {
            qvalues_s.push_str(",");
          }
        }
        //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
        //writeln!(&mut self.trace_file, "step,train_argmax,{},{}", argmax_k, qvalues_s).unwrap();
        self.trace_file.flush().unwrap();
        argmax_k as u32
      };
      //let prev_env = Rc::new((*self.env).clone());
      let action = <E::Action as DiscreteAction>::from_idx(act_idx);
      let res = self.env.step(&action).unwrap();
      //let next_env = Rc::new((*self.env).clone());
      let next_obs = self.env.observe(&mut self.rng);
      let terminal = self.env.is_terminal();
      /*if self.epoch_ep_cnt <= 2 {
        prev_env._save_png(&step_prefix.join(&format!("step_{}_{}_a_prev.png", self.step_count, step_nr)));
        next_env._save_png(&step_prefix.join(&format!("step_{}_{}_z_next.png", self.step_count, step_nr)));
      }*/
      //self.replay_cache.insert(prev_env, action, res, next_env);
      let obs_hash = jenkins_hash_obs(&next_obs);
      //println!("DEBUG: dq: train step: obs hash: {}", obs_hash);
      //writeln!(&mut self.trace_file, "step,train,hash,{}", obs_hash).unwrap();
      self.belief_state.push(Rc::new(next_obs.clone()));
      self.replay_cache.insert(action, res, Rc::new(next_obs), terminal);

      if self.env.is_terminal() {
        self.ep_count += 1;
        self.reset_env();
        self.epoch_ep_cnt += 1;
        let mut v = <EvalV as Value>::from_scalar(0.0, self.cfg.eval_cfg);
        for &r in self.sim_res.iter().rev() {
          v.lreduce(r);
        }
        self.sim_res.clear();
        self.avg_value += (v.to_scalar() - self.avg_value) / self.epoch_ep_cnt as f32;
        self.min_value = self.min_value.min(v.to_scalar());
        self.max_value = self.max_value.max(v.to_scalar());
        println!("DEBUG: dq: stats: steps: {} episodes: {} {:.3} {:.3} {:.3} {}",
            self.step_count(), self.epoch_ep_cnt, self.avg_value, self.min_value, self.max_value, v.to_scalar());
      }

      if self.step_count % self.cfg.target_steps == 0 {
        //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
        //writeln!(&mut self.trace_file, "train,update_target").unwrap();
        self.trace_file.flush().unwrap();
        println!("DEBUG: dq: updating target param... {} {}", self.target.len(), self.param.len());
        self.target.copy_from_slice(&self.param);
        let mut target_fn = self.target_fn.borrow_mut();
        target_fn.load_diff_param(&mut self.target);
      }

      if self.step_count % self.cfg.update_steps == 0 {
        //writeln!(&mut self.trace_file, "rng,{},{}", self.rng._state()[0], self.rng._state()[1]).unwrap();
        //writeln!(&mut self.trace_file, "train,minibatch").unwrap();
        self.trace_file.flush().unwrap();
        self.samples.clear();
        let mut idxs_str = String::new();
        let mut hashes_str = String::new();
        for idx in 0 .. self.cfg.minibatch_sz {
          let entry = self.replay_cache.sample(&mut self.rng);
          idxs_str.push_str(&format!("{}", entry.idx));
          if idx < self.cfg.minibatch_sz-1 {
            idxs_str.push_str(",");
          }
          hashes_str.push_str(&format!("{},{}", jenkins_hash_state(&entry.prev), jenkins_hash_state(&entry.next)));
          if idx < self.cfg.minibatch_sz-1 {
            hashes_str.push_str(",");
          }
          self.samples.push(entry);
        }
        //let minibatch_hash = jenkins_hash_u64(&hashes);
        //println!("DEBUG: minibatch: hash: {}", minibatch_hash);
        //writeln!(&mut self.trace_file, "train,minibatch,idxs,{}", idxs_str).unwrap();
        //writeln!(&mut self.trace_file, "train,minibatch,hashes,{}", hashes_str).unwrap();

        let mut value_fn = self.value_fn.borrow_mut();
        let mut target_fn = self.target_fn.borrow_mut();

        // XXX: double Q-learning.
        /*value_fn.reset_loss();
        value_fn.reset_grad();
        value_fn.next_iteration();
        self.cache.clear();
        for (idx, entry) in self.samples.iter().enumerate() {
          let mut item = SampleItem::new();
          //let env = entry.next_env.clone();
          //let env_repr_dim = env._shape3d();
          let obs = entry.next.clone();
          let obs_dim = obs._shape3d();
          //println!("DEBUG: update (values): obs dim: {:?}", obs_dim);
          /*if self.epoch_ep_cnt <= 2 {
            env._save_png(&minibatch_prefix.join(&format!("minibatch_{}_{}_next.png", self.step_count, idx)));
          }*/
          item.kvs.insert::<SampleExtractInputKey<[f32]>>(Rc::new(obs));
          item.kvs.insert::<SampleInputShape3dKey>(obs_dim);
          self.cache.push(item);
        }
        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);

        self.target_maxs.clear();
        for (idx, entry) in self.samples.iter().enumerate() {
          let value_output = &value_fn._get_pred()[idx * action_dim .. (idx+1) * action_dim];
          let argmax_k = argmax(value_output.iter().map(|&v| F32InfNan(v))).unwrap();
          self.target_maxs.push(argmax_k);
        }*/

        target_fn.load_diff_param(&mut self.target);
        target_fn.reset_loss();
        target_fn.reset_grad();
        target_fn.next_iteration();
        self.cache.clear();
        for (idx, entry) in self.samples.iter().enumerate() {
          let mut item = SampleItem::new();
          let obs = Rc::new(entry.next.clone());
          let obs_dim = obs._shape3d();
          //println!("DEBUG: update (targets): obs dim: {:?}", obs_dim);
          /*if self.epoch_ep_cnt <= 2 {
            env._save_png(&minibatch_prefix.join(&format!("minibatch_{}_{}_next.png", self.step_count, idx)));
          }*/
          item.kvs.insert::<SampleExtractInputKey<[f32]>>(obs.clone());
          item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(obs.clone());
          item.kvs.insert::<SampleInputShape3dKey>(obs_dim);
          self.cache.push(item);
        }
        target_fn.load_batch(&self.cache);
        target_fn.forward(OpPhase::Learning);

        let mut avg_target_val = 0.0;
        self.target_vals.clear();
        //for idx in 0 .. self.cfg.minibatch_sz {
        for (idx, entry) in self.samples.iter().enumerate() {
          if entry.terminal {
            self.target_vals.push(0.0);
            continue;
          }
          let target_output = &target_fn._get_pred()[idx * action_dim .. (idx+1) * action_dim];
          let argmax_k = argmax(target_output.iter().map(|&v| F32InfNan(v))).unwrap();
          //let argmax_k = self.target_maxs[idx]; // XXX: double Q-learning.
          self.target_vals.push(target_output[argmax_k]);
          avg_target_val += target_output[argmax_k];
        }
        avg_target_val /= self.cfg.minibatch_sz as f32;

        let mut avg_value = 0.0;
        let mut rewards = vec![];
        //value_fn.load_diff_param(&mut self.param);
        value_fn.reset_loss();
        value_fn.reset_grad();
        value_fn.next_iteration();
        self.cache.clear();
        for (idx, entry) in self.samples.iter().enumerate() {
          let mut item = SampleItem::new();
          let obs = Rc::new(entry.prev.clone());
          let obs_dim = obs._shape3d();
          //println!("DEBUG: update (qvalues): obs dim: {:?}", obs_dim);
          /*if self.epoch_ep_cnt <= 2 {
            env._save_png(&minibatch_prefix.join(&format!("minibatch_{}_{}_prev.png", self.step_count, idx)));
          }*/
          let act_idx = entry.action.idx();
          let mut action_value = V::from_scalar(self.target_vals[idx], self.cfg.value_cfg);
          action_value.lreduce(entry.res.unwrap());
          rewards.push(entry.res.unwrap());
          let act_val = action_value.to_scalar();
          avg_value += act_val;
          item.kvs.insert::<SampleExtractInputKey<[f32]>>(obs.clone());
          item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(obs.clone());
          item.kvs.insert::<SampleInputShape3dKey>(obs_dim);
          item.kvs.insert::<SampleRegressTargetKey>(act_val);
          item.kvs.insert::<SampleClassLabelKey>(act_idx);
          self.cache.push(item);
        }
        avg_value /= self.cfg.minibatch_sz as f32;
        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);
        value_fn.backward();
        value_fn.update_nondiff_param(self.iter_count);

        let avg_loss = value_fn.store_loss() / self.cfg.minibatch_sz as f32;
        value_fn.store_grad(&mut self.grad);
        self.grad.reshape_mut(self.grad_sz).div_scalar(self.cfg.minibatch_sz as f32);
        for &g in self.grad.iter() {
          assert!(!g.is_nan());
        }

        /*println!("DEBUG: dq: backprop: loss: {:.6}", avg_loss);
        println!("DEBUG: dq: backprop: rewards: {:?}", rewards);
        println!("DEBUG: dq: backprop: maxpostq: {:?}", self.target_vals);
        println!("DEBUG: dq: backprop: postq: {:?}", target_fn._get_pred());
        println!("DEBUG: dq: backprop: preq: {:?}", value_fn._get_pred());
        println!("DEBUG: dq: backprop: target: {:?}", value_fn._get_target());
        println!("DEBUG: dq: backprop: delta: {:?}", value_fn._get_delta());*/

        self.tmp_buf.copy_from_slice(&self.grad);
        self.tmp_buf.reshape_mut(self.grad_sz).square();
        self.grad_var_acc.reshape_mut(self.grad_sz).average(1.0 - self.cfg.rms_decay, self.tmp_buf.reshape(self.grad_sz));

        let rms_decay_scale = 1.0 / (1.0 - self.cfg.rms_decay.powi((self.iter_count + 1) as i32));
        self.tmp_buf.copy_from_slice(&self.grad_var_acc);
        self.tmp_buf.reshape_mut(self.grad_sz).scale(rms_decay_scale);
        self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
        self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
        self.tmp_buf.reshape_mut(self.grad_sz).elem_ldiv(self.grad.reshape(self.grad_sz));

        self.update_acc.reshape_mut(self.grad_sz).scale(self.cfg.momentum);
        self.update_acc.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.tmp_buf.reshape(self.grad_sz));

        self.param.reshape_mut(self.grad_sz).add(1.0, self.update_acc.reshape(self.grad_sz));

        /*let mut param_file = File::open(&format!("saved_record/pong_epoch_0_bin/param_step_{}.bin", self.step_count)).unwrap();
        let mut raw_param_buf = vec![];
        param_file.read_to_end(&mut raw_param_buf).unwrap();
        let mut param_buf = unsafe { from_raw_parts_mut(raw_param_buf.as_mut_ptr() as *mut _, raw_param_buf.len() / 4) };
        assert_eq!(self.grad_sz, param_buf.len());

        self.tmp_buf.copy_from_slice(&self.param);
        self.tmp_buf.reshape_mut(self.grad_sz).add(-1.0, param_buf.reshape(self.grad_sz));
        let diff_norm = self.tmp_buf.reshape(self.grad_sz).l2_norm();
        println!("DEBUG: dq: step: {} param avg diff norm: {:.6e}", self.step_count, diff_norm / (self.grad_sz as f32).sqrt());
        value_fn.load_diff_param(&mut param_buf);
        self.param.copy_from_slice(&param_buf);*/

        self.iter_count += 1;
        /*if self.iter_count % 10 == 0 {
          println!("DEBUG: iter: {} |x|: {:.6e} |g|: {:.6e}", self.iter_count, self.param.reshape(self.grad_sz).l2_norm(), self.grad.reshape(self.grad_sz).l2_norm());
        }*/
      }

      self.step_count += 1;
      self.sim_res.push(res.unwrap());
      step_res.push(res.unwrap());
    }
    let mut v = <V as Value>::from_scalar(0.0, self.cfg.value_cfg);
    for &r in step_res.iter().rev() {
      v.lreduce(r);
    }
    let step_value = v.to_scalar();

    /*let ANSI_COLOR_RED      = "\x1b[31m";
    let ANSI_COLOR_GREEN    = "\x1b[32m";
    let ANSI_COLOR_RESET    = "\x1b[0m";
    let (color_code, color_code_reset) = if step_value > 0.0 {
      (ANSI_COLOR_GREEN, ANSI_COLOR_RESET)
    } else if step_value < 0.0 {
      (ANSI_COLOR_RED, ANSI_COLOR_RESET)
    } else {
      ("", "")
    };
    println!("DEBUG: dq: train: iter: {} step: {} ep: {} exp: {:.3} nrand: {} argmax: {:?} {}value: {:.4}{} avg q: {:.4} avg target: {:.4} avg loss: {:.6}",
        self.iter_count, self.step_count, self.ep_count, step_exp, step_nrand, step_argmax, color_code, step_value, color_code_reset, avg_value, avg_target_val, avg_loss,
    );*/
  }

  pub fn eval(&mut self, num_trials: usize) -> (DiffQRecord, DiffQRecord) {
    let action_dim = <E::Action as Action>::dim();

    self.epoch_count += 1;
    println!("DEBUG: dq: train: epoch: {} avg value: {:.4} min value: {:.4} max value: {:.4}",
        self.epoch_count, self.avg_value, self.min_value, self.max_value);
    let train_rec = DiffQRecord{
      iter:         self.iter_count,
      step:         self.step_count,
      avg_episodes: self.epoch_ep_cnt,
      avg_value:    self.avg_value,
      min_value:    self.min_value,
      max_value:    self.max_value,
      elapsed:      0.0,
    };
    self.epoch_ep_cnt = 0;
    self.avg_value = 0.0;
    self.min_value = f32::INFINITY;
    self.max_value = -f32::INFINITY;

    self.eval_env.reset(&self.cfg.init_cfg, &mut self.rng);
    self.eval_belief.reset();

    // FIXME(20161102)
    let valid_rec = train_rec;

    (train_rec, valid_rec)

    /*let mut value_fn = self.value_fn.borrow_mut();
    value_fn.load_diff_param(&mut self.param);

    //let mut avg_value = 0.0;
    let mut step_res = vec![];
    for idx in 0 .. num_trials {
      value_fn.reset_loss();
      value_fn.reset_grad();
      value_fn.next_iteration();
      self.eval_env.reset(&self.cfg.init_cfg, &mut self.rng);
      assert!(!self.eval_env.is_terminal());
      step_res.clear();
      for _ in 0 .. self.cfg.eval_horizon {
        if self.eval_env.is_terminal() {
          break;
        }
        assert!(!self.eval_env.is_terminal());
        let u = self.rng.gen_range(0.0, 1.0);
        let uniform_act_idx = self.rng.gen_range(0, action_dim as u32);
        let act_idx = if u < self.cfg.exp_eval {
          uniform_act_idx
        } else {
          let mut item = SampleItem::new();
          let env = self.eval_env.clone();
          let env_repr_dim = env._shape3d();
          item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
          item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
          self.cache.clear();
          self.cache.push(item);
          value_fn.load_batch(&self.cache);
          value_fn.forward(OpPhase::Learning);
          let argmax_k = argmax(value_fn._get_pred()[ .. action_dim].iter().map(|&v| F32InfNan(v))).unwrap();
          argmax_k as u32
        };
        let action = <E::Action as DiscreteAction>::from_idx(act_idx);
        let res = self.eval_env.step(&action).unwrap();
        step_res.push(res.unwrap());
      }
      let mut v = <EvalV as Value>::from_scalar(0.0, self.cfg.eval_cfg);
      for &r in step_res.iter().rev() {
        v.lreduce(r);
      }
      //let step_value = v.to_scalar();
      //avg_value += step_value;
      self.avg_value += (v.to_scalar() - self.avg_value) / (idx + 1) as f32;
      self.min_value = self.min_value.min(v.to_scalar());
      self.max_value = self.max_value.max(v.to_scalar());
    }
    //avg_value /= num_trials as f32;
    println!("DEBUG: dq: valid: epoch: {} avg value: {:.4} min value: {:.4} max value: {:.4}",
        self.epoch_count, self.avg_value, self.min_value, self.max_value);
    let valid_rec = DiffQRecord{
      iter:         self.iter_count,
      step:         self.step_count,
      avg_episodes: num_trials,
      avg_value:    self.avg_value,
      min_value:    self.min_value,
      max_value:    self.max_value,
      elapsed:      0.0,
    };
    self.avg_value = 0.0;
    self.min_value = f32::INFINITY;
    self.max_value = -f32::INFINITY;

    (train_rec, valid_rec)*/
  }
}
