use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};
use kernels::*;
use opt::pg_new::{BasePolicyGrad};
use replay::{ReplayEntry, LinearReplayCache};

use densearray::prelude::*;
use float::ord::{F32InfNan};
use iter_utils::{argmax};
use operator::prelude::*;
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
pub struct AdamDiffQConfig<Init, VCfg, EvalVCfg> {
  //pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub replay_sz:    usize,
  pub replay_init:  usize,
  pub update_steps: usize,
  pub target_steps: usize,
  pub eval_horizon: usize,
  pub exp_init:     f32,
  pub exp_anneal:   f32,
  pub exp_eval:     f32,
  pub step_size:    f32,
  pub gamma1:       f32,
  pub gamma2:       f32,
  pub epsilon:      f32,
  pub init_cfg:     Init,
  pub value_cfg:    VCfg,
  pub eval_cfg:     EvalVCfg,
}

pub struct AdamDiffQWorker<E, V, EvalV, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  cfg:          AdamDiffQConfig<E::Init, V::Cfg, EvalV::Cfg>,
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
  replay_cache: LinearReplayCache<E>,
  sim_env:      Rc<E>,
  //eval_env:     Rc<E>,
  value_fn:     Rc<RefCell<ValueFn>>,
  target_fn:    Rc<RefCell<ValueFn>>,
  samples:      Vec<ReplayEntry<E>>,
  target_vals:  Vec<f32>,
  cache:        Vec<SampleItem>,
  target:       Vec<f32>,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  //grad_acc:     Vec<f32>,
  grad_var_acc: Vec<f32>,
  tmp_buf:      Vec<f32>,
}

impl<E, V, EvalV, ValueFn> AdamDiffQWorker<E, V, EvalV, ValueFn>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      EvalV: Value<Res=E::Response>,
      ValueFn: DiffLoss<SampleItem, IoBuf=[f32]>,
{
  pub fn new(cfg: AdamDiffQConfig<E::Init, V::Cfg, EvalV::Cfg>, value_fn: Rc<RefCell<ValueFn>>, target_fn: Rc<RefCell<ValueFn>>) -> Self {
    let grad_sz = value_fn.borrow_mut().diff_param_sz();
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let replay_cache = LinearReplayCache::new(cfg.replay_sz);
    let sim_env = Rc::new(E::default());
    //sim_env.reset(&cfg.init_cfg, &mut rng);
    /*let eval_env = Rc::new(E::default());
    //eval_env.reset(&cfg.init_cfg, &mut rng);*/
    let samples = Vec::with_capacity(cfg.minibatch_sz);
    let target_vals = Vec::with_capacity(cfg.minibatch_sz);
    let cache = Vec::with_capacity(cfg.minibatch_sz);
    let mut target = Vec::with_capacity(grad_sz);
    target.resize(grad_sz, 0.0);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    /*let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);*/
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    AdamDiffQWorker{
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
      sim_env:      sim_env,
      //eval_env:     eval_env,
      value_fn:     value_fn,
      target_fn:    target_fn,
      samples:      samples,
      target_vals:  target_vals,
      cache:        cache,
      target:       target,
      param:        param,
      grad:         grad,
      //grad_acc:     grad_acc,
      grad_var_acc: grad_var_acc,
      tmp_buf:      tmp_buf,
    }
  }

  pub fn deterministic_test(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut value_fn = self.value_fn.borrow_mut();
    let mut target_fn = self.target_fn.borrow_mut();

    let mut param_file = File::open("param.bin").unwrap();
    let mut raw_param_buf = vec![];
    param_file.read_to_end(&mut raw_param_buf).unwrap();
    let mut param_buf = unsafe { from_raw_parts_mut(raw_param_buf.as_mut_ptr() as *mut _, raw_param_buf.len() / 4) };
    assert_eq!(self.grad_sz, param_buf.len());
    value_fn.load_diff_param(&mut param_buf);

    let action_dim = <E::Action as Action>::dim();
    println!("DEBUG: dq: deterministic testing...");
    let mut trace_actions = Vec::with_capacity(10_000);
    self.sim_env.reset(&self.cfg.init_cfg, &mut self.rng);
    for _ in 0 .. 10_000 {
      if self.sim_env.is_terminal() {
        self.sim_env.reset(&self.cfg.init_cfg, &mut self.rng);
        self.epoch_ep_cnt += 1;
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
      assert!(!self.sim_env.is_terminal());
      let act_idx = {
        let mut item = SampleItem::new();
        let env = self.sim_env.clone();
        let env_repr_dim = env._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        self.cache.clear();
        self.cache.push(item);
        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);
        let argmax_k = argmax(value_fn._get_pred()[ .. action_dim].iter().map(|&v| F32InfNan(v))).unwrap();
        //println!("DEBUG: dq: argmax: {} qvalues: {:?}", argmax_k, &value_fn._get_pred()[ .. action_dim]);
        trace_actions.push(argmax_k);
        argmax_k as u32
      };
      let prev_env = Rc::new((*self.sim_env).clone());
      let action = <E::Action as DiscreteAction>::from_idx(act_idx);
      let res = self.sim_env.step(&action).unwrap();
      let next_env = Rc::new((*self.sim_env).clone());
      self.replay_cache.insert(prev_env, action, res, next_env);
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
    let mut value_fn = self.value_fn.borrow_mut();
    let mut target_fn = self.target_fn.borrow_mut();

    value_fn.init_param(rng);
    value_fn.store_diff_param(&mut self.param);

    target_fn.init_param(rng);
    target_fn.store_diff_param(&mut self.target);

    let action_dim = <E::Action as Action>::dim();
    println!("DEBUG: dq: initializing replay memory...");
    self.sim_env.reset(&self.cfg.init_cfg, &mut self.rng);
    for _ in 0 .. self.cfg.replay_init {
      if self.sim_env.is_terminal() {
        self.sim_env.reset(&self.cfg.init_cfg, &mut self.rng);
        self.epoch_ep_cnt += 1;
        let mut v = <EvalV as Value>::from_scalar(0.0, self.cfg.eval_cfg);
        for &r in self.sim_res.iter().rev() {
          v.lreduce(r);
        }
        self.sim_res.clear();
        self.avg_value += (v.to_scalar() - self.avg_value) / self.epoch_ep_cnt as f32;
        self.min_value = self.min_value.min(v.to_scalar());
        self.max_value = self.max_value.max(v.to_scalar());
      }
      assert!(!self.sim_env.is_terminal());
      let act_idx = self.rng.gen_range(0, action_dim as u32);
      let prev_env = Rc::new((*self.sim_env).clone());
      let action = <E::Action as DiscreteAction>::from_idx(act_idx);
      let res = self.sim_env.step(&action).unwrap();
      let next_env = Rc::new((*self.sim_env).clone());
      self.replay_cache.insert(prev_env, action, res, next_env);
      self.sim_res.push(res.unwrap());
    }
    println!("DEBUG: dq: replay init: {} {:.3} {:.3} {:.3}",
        self.epoch_ep_cnt, self.avg_value, self.min_value, self.max_value);
    self.epoch_ep_cnt = 0;
    self.avg_value = 0.0;
    self.min_value = f32::INFINITY;
    self.max_value = -f32::INFINITY;
  }

  pub fn step_count(&self) -> usize {
    self.step_count
  }

  pub fn iter_count(&self) -> usize {
    self.iter_count
  }

  pub fn update(&mut self) {
    let action_dim = <E::Action as Action>::dim();

    let mut value_fn = self.value_fn.borrow_mut();
    let mut target_fn = self.target_fn.borrow_mut();

    /*let step_prefix = PathBuf::from(&format!("tmp_debug/steps_{}", self.epoch_count));
    let minibatch_prefix = PathBuf::from(&format!("tmp_debug/minibatches_{}", self.epoch_count));
    if self.epoch_ep_cnt <= 2 {
      create_dir_all(&step_prefix).ok();
      create_dir_all(&minibatch_prefix).ok();
    }*/

    value_fn.load_diff_param(&mut self.param);
    value_fn.reset_loss();
    value_fn.reset_grad();
    value_fn.next_iteration();
    let mut step_res = vec![];
    let mut step_nrand = 0;
    let mut step_argmax = vec![];
    let mut step_exp = 0.0;
    for step_nr in 0 .. self.cfg.update_steps {
      if self.sim_env.is_terminal() {
        self.sim_env.reset(&self.cfg.init_cfg, &mut self.rng);
        self.ep_count += 1;
        self.epoch_ep_cnt += 1;
        let mut v = <EvalV as Value>::from_scalar(0.0, self.cfg.eval_cfg);
        for &r in self.sim_res.iter().rev() {
          v.lreduce(r);
        }
        self.sim_res.clear();
        self.avg_value += (v.to_scalar() - self.avg_value) / self.epoch_ep_cnt as f32;
        self.min_value = self.min_value.min(v.to_scalar());
        self.max_value = self.max_value.max(v.to_scalar());
      }
      assert!(!self.sim_env.is_terminal());
      let t = 1.0_f32.min(0.0_f32.max(self.step_count as f32 / self.cfg.replay_sz as f32));
      let exp_rate = self.cfg.exp_init * (1.0 - t) + self.cfg.exp_anneal * t;
      step_exp = exp_rate;
      let u = self.rng.gen_range(0.0, 1.0);
      let uniform_act_idx = self.rng.gen_range(0, action_dim as u32);
      let act_idx = if u < exp_rate {
        step_nrand += 1;
        uniform_act_idx
      } else {
        let mut item = SampleItem::new();
        let env = self.sim_env.clone();
        let env_repr_dim = env._shape3d();
        item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
        item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
        self.cache.clear();
        self.cache.push(item);
        value_fn.load_batch(&self.cache);
        value_fn.forward(OpPhase::Learning);
        let argmax_k = argmax(value_fn._get_pred()[ .. action_dim].iter().map(|&v| F32InfNan(v))).unwrap();
        step_argmax.push(argmax_k);
        argmax_k as u32
      };
      let prev_env = Rc::new((*self.sim_env).clone());
      let action = <E::Action as DiscreteAction>::from_idx(act_idx);
      let res = self.sim_env.step(&action).unwrap();
      let next_env = Rc::new((*self.sim_env).clone());
      /*if self.epoch_ep_cnt <= 2 {
        prev_env._save_png(&step_prefix.join(&format!("step_{}_{}_a_prev.png", self.step_count, step_nr)));
        next_env._save_png(&step_prefix.join(&format!("step_{}_{}_z_next.png", self.step_count, step_nr)));
      }*/
      self.replay_cache.insert(prev_env, action, res, next_env);
      self.step_count += 1;
      self.sim_res.push(res.unwrap());
      step_res.push(res.unwrap());
    }
    let mut v = <V as Value>::from_scalar(0.0, self.cfg.value_cfg);
    for &r in step_res.iter().rev() {
      v.lreduce(r);
    }
    let step_value = v.to_scalar();

    self.samples.clear();
    for _ in 0 .. self.cfg.minibatch_sz {
      let entry = self.replay_cache.sample(&mut self.rng);
      self.samples.push(entry.clone());
    }

    //target_fn.load_diff_param(&mut self.target);
    target_fn.reset_loss();
    //target_fn.reset_grad();
    target_fn.next_iteration();
    self.cache.clear();
    for (idx, entry) in self.samples.iter().enumerate() {
      let mut item = SampleItem::new();
      let env = entry.next_env.clone();
      /*if self.epoch_ep_cnt <= 2 {
        env._save_png(&minibatch_prefix.join(&format!("minibatch_{}_{}_next.png", self.step_count, idx)));
      }*/
      let env_repr_dim = env._shape3d();
      item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
      item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
      self.cache.push(item);
    }
    target_fn.load_batch(&self.cache);
    target_fn.forward(OpPhase::Learning);

    let mut avg_target_val = 0.0;
    self.target_vals.clear();
    //for idx in 0 .. self.cfg.minibatch_sz {
    for (idx, entry) in self.samples.iter().enumerate() {
      if entry.next_env.is_terminal() {
        self.target_vals.push(0.0);
        continue;
      }
      let target_output = &target_fn._get_pred()[idx * action_dim .. (idx+1) * action_dim];
      let argmax_k = argmax(target_output.iter().map(|&v| F32InfNan(v))).unwrap();
      self.target_vals.push(target_output[argmax_k]);
      avg_target_val += target_output[argmax_k];
    }
    avg_target_val /= self.cfg.minibatch_sz as f32;

    let mut avg_value = 0.0;
    //value_fn.load_diff_param(&mut self.param);
    value_fn.reset_loss();
    value_fn.reset_grad();
    value_fn.next_iteration();
    self.cache.clear();
    for (idx, entry) in self.samples.iter().enumerate() {
      let mut item = SampleItem::new();
      let env = entry.prev_env.clone();
      /*if self.epoch_ep_cnt <= 2 {
        env._save_png(&minibatch_prefix.join(&format!("minibatch_{}_{}_prev.png", self.step_count, idx)));
      }*/
      let env_repr_dim = env._shape3d();
      let act_idx = entry.action.idx();
      let mut action_value = V::from_scalar(self.target_vals[idx], self.cfg.value_cfg);
      action_value.lreduce(entry.res.unwrap());
      let act_val = action_value.to_scalar();
      avg_value += act_val;
      item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
      item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
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
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);

    //let gamma1_scale = 1.0 / (1.0 - (1.0 - self.cfg.gamma1).powi((self.iter_count + 1) as i32));
    let gamma2_scale = 1.0 / (1.0 - (1.0 - self.cfg.gamma2).powi((self.iter_count + 1) as i32));

    //self.grad_acc.reshape_mut(self.grad_sz).average(self.cfg.gamma1, self.grad.reshape(self.grad_sz));

    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    self.grad_var_acc.reshape_mut(self.grad_sz).average(self.cfg.gamma2, self.tmp_buf.reshape(self.grad_sz));

    self.tmp_buf.copy_from_slice(&self.grad_var_acc);
    self.tmp_buf.reshape_mut(self.grad_sz).scale(gamma2_scale);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    // XXX(20161027): The following is a Neon quirk.
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).reciprocal();
    //self.tmp_buf.reshape_mut(self.grad_sz).elem_mult(-self.cfg.step_size * gamma1_scale, self.grad_acc.reshape(self.grad_sz));
    self.tmp_buf.reshape_mut(self.grad_sz).elem_mult(1.0, self.grad.reshape(self.grad_sz));
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.tmp_buf.reshape(self.grad_sz));

    self.iter_count += 1;

    let ANSI_COLOR_RED      = "\x1b[31m";
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
    );

    if self.step_count % self.cfg.target_steps == 0 {
      println!("DEBUG: dq: updating target param...");
      self.target.copy_from_slice(&self.param);
      target_fn.load_diff_param(&mut self.target);
    }
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
