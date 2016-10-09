use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};

use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct EpisodeStepSample<E> where E: Env {
  pub env:      Rc<RefCell<E>>,
  pub act_idx:  Option<u32>,
  pub suffix_r: Option<E::Response>,
  pub baseline: Option<f32>,
  weight:       Option<f32>,
}

impl<E> EpisodeStepSample<E> where E: Env {
  pub fn new(env: Rc<RefCell<E>>, act_idx: Option<u32>, suffix_r: Option<E::Response>) -> EpisodeStepSample<E> {
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
    self.env.borrow_mut().extract_observable(output);
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

pub struct BasePgWorker<E, PolicyOp> where E: Env {
  batch_sz:     usize,
  minibatch_sz: usize,
  max_horizon:  usize,
  act_dist:     DiscreteDist32,
  episodes:     Vec<Episode<E>>,
  step_values:  Vec<Vec<f32>>,
  final_values: Vec<Option<f32>>,
  _marker:      PhantomData<(E, PolicyOp)>,
}

impl<E, PolicyOp> BasePgWorker<E, PolicyOp>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      PolicyOp: DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  pub fn new(minibatch_sz: usize, max_horizon: usize) -> BasePgWorker<E, PolicyOp> {
    let mut episodes = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      episodes.push(Episode::new());
    }
    let mut step_values = Vec::with_capacity(minibatch_sz);
    for _ in 0 .. minibatch_sz {
      step_values.push(vec![]);
    }
    let mut final_values = Vec::with_capacity(minibatch_sz);
    final_values.resize(minibatch_sz, None);
    BasePgWorker{
      batch_sz:     1,
      minibatch_sz: minibatch_sz,
      max_horizon:  max_horizon,
      act_dist:     DiscreteDist32::new(<E::Action as Action>::dim()),
      episodes:     episodes,
      step_values:  step_values,
      final_values: final_values,
      _marker:      PhantomData,
    }
  }

  pub fn reset_episodes<R>(&mut self, init_cfg: &E::Init, rng: &mut R) where R: Rng {
    for episode in self.episodes.iter_mut() {
      episode.reset(init_cfg, rng);
    }
  }

  pub fn sample_steps<V, R>(&mut self, max_num_steps: Option<usize>, value_cfg: V::Cfg, policy: &mut PolicyOp, /*value: Option<&mut ValueOp>,*/ cache: &mut Vec<EpisodeStepSample<E>>, init_cfg: &E::Init, rng: &mut R) where V: Value<Res=E::Response>, R: Rng {
    let action_dim = <E::Action as Action>::dim();
    for episode in self.episodes.iter_mut() {
      if episode.terminated() || episode.horizon() >= self.max_horizon {
        episode.reset(init_cfg, rng);
      }
    }
    for (idx, episode) in self.episodes.iter_mut().enumerate() {
      let init_horizon = episode.horizon();
      let horizon_limit = if let Some(max_num_steps) = max_num_steps {
        min(init_horizon + max_num_steps, self.max_horizon)
      } else {
        self.max_horizon
      };
      for k in init_horizon .. horizon_limit {
        if episode.terminated() {
          break;
        }
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let mut next_env: E = prev_env.borrow().clone();
        let sample = EpisodeStepSample::new(prev_env, None, None);
        cache.clear();
        cache.push(sample);
        policy.load_data(&cache);
        policy.forward(OpPhase::Learning);
        let output = policy.get_output();
        // FIXME(20161009): sometimes the policy output contains NaNs because
        // all probabilities were zero, should gracefully handle this case.
        self.act_dist.reset(&output.borrow()[ .. action_dim]);
        let act_idx = self.act_dist.sample(rng).unwrap();
        let action = <E::Action as DiscreteAction>::from_idx(act_idx as u32);
        if let Ok(res) = next_env.step(&action) {
          episode.steps.push(EpisodeStep{
            action:   action,
            res:      res,
            next_env: Rc::new(RefCell::new(next_env)),
          });
        } else {
          panic!();
        }
      }
      if !episode.terminated() {
        if false /*let Some(ref mut value_op) = value*/ {
          let impute_val = 0.0; // FIXME: get from the value op.
          self.final_values[idx] = Some(impute_val);
        } else {
          self.final_values[idx] = None;
        }
      } else {
        self.final_values[idx] = None;
      }
      let mut suffix_val = if let Some(final_val) = self.final_values[idx] {
        Some(<V as Value>::from_scalar(final_val, value_cfg))
      } else {
        None
      };
      self.step_values[idx].resize(episode.horizon(), 0.0);
      for k in (init_horizon .. episode.horizon()).rev() {
        if let Some(res) = episode.steps[k].res {
          if let Some(ref mut suffix_val) = suffix_val {
            suffix_val.lreduce(res);
          } else {
            suffix_val = Some(<V as Value>::from_res(res, value_cfg));
          }
        }
        if let Some(suffix_val) = suffix_val {
          self.step_values[idx][k] = suffix_val.to_scalar();
        }
      }
    }
  }

  pub fn sample<R>(&mut self, policy: &mut PolicyOp, /*value: Option<&mut ValueOp>,*/ cache: &mut Vec<EpisodeStepSample<E>>, episodes: &mut [Episode<E>], init_cfg: &E::Init, rng: &mut R) where R: Rng {
    let action_dim = <E::Action as Action>::dim();
    for episode in episodes {
      episode.reset(init_cfg, rng);
      for k in episode.steps.len() .. self.max_horizon {
        if episode.terminated() {
          break;
        }
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        let mut next_env: E = prev_env.borrow().clone();
        let sample = EpisodeStepSample::new(prev_env, None, None);
        cache.clear();
        cache.push(sample);
        policy.load_data(&cache);
        policy.forward(OpPhase::Learning);
        let output = policy.get_output();
        self.act_dist.reset(&output.borrow()[ .. action_dim]);
        let act_idx = self.act_dist.sample(rng).unwrap();
        let action = <E::Action as DiscreteAction>::from_idx(act_idx as u32);
        if let Ok(res) = next_env.step(&action) {
          episode.steps.push(EpisodeStep{
            action:   action,
            res:      res,
            next_env: Rc::new(RefCell::new(next_env)),
          });
        } else {
          panic!();
        }
      }
      if !episode.terminated() {
        // FIXME(20161008): bootstrap with the value of the last state.
      }
      episode._fill_suffixes();
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct PolicyGradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub max_horizon:  usize,
  pub baseline:     f32,
}

pub struct PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + Clone, //EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  cfg:      PolicyGradConfig,
  operator: Op,
  cache:    Vec<EpisodeStepSample<E>>,
  base_pg:  BasePgWorker<E, Op>,
  grad_acc: Vec<f32>,
}

impl<E, Op> PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + Clone, //EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperator<f32> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  pub fn new(cfg: PolicyGradConfig, op: Op) -> PolicyGradWorker<E, Op> {
    let grad_sz = op.diff_param_sz();
    let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);
    PolicyGradWorker{
      cfg:      cfg,
      operator: op,
      cache:    Vec::with_capacity(cfg.batch_sz),
      base_pg:  BasePgWorker::new(cfg.minibatch_sz, cfg.max_horizon),
      grad_acc: grad_acc,
    }
  }

  pub fn sample<R>(&mut self, episodes: &mut [Episode<E>], init_cfg: &E::Init, rng: &mut R) where R: Rng {
    self.base_pg.sample(&mut self.operator, &mut self.cache, episodes, init_cfg, rng);
  }
}

impl<E, Op> OptWorker<f32, Episode<E>> for PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + Clone, //EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperator<f32> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  type Rng = Op::Rng;

  fn init_param(&mut self, rng: &mut Self::Rng) {
    self.operator.init_param(rng);
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) { unimplemented!(); }
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) { unimplemented!(); }
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) { unimplemented!(); }

  fn step(&mut self, episodes: &mut Iterator<Item=Episode<E>>) {
    self.operator.reset_loss();
    self.operator.reset_grad();
    self.cache.clear();
    for episode in episodes.take(self.cfg.minibatch_sz) {
      for k in 0 .. episode.steps.len() {
        let mut sample = match k {
          0 => EpisodeStepSample::new(
              episode.init_env.clone(),
              Some(episode.steps[0].action.idx()),
              episode.suffixes[0]),
          k => EpisodeStepSample::new(
              episode.steps[k-1].next_env.clone(),
              Some(episode.steps[k].action.idx()),
              episode.suffixes[k]),
        };
        assert!(sample.suffix_r.is_some());
        sample.set_baseline(self.cfg.baseline);
        sample.init_weight();
        sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
        self.cache.push(sample);
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }
        self.operator.load_data(&self.cache);
        self.operator.forward(OpPhase::Learning);
        self.operator.backward();
        self.cache.clear();
      }
    }
    if !self.cache.is_empty() {
      self.operator.load_data(&self.cache);
      self.operator.forward(OpPhase::Learning);
      self.operator.backward();
      self.cache.clear();
    }
    self.operator.accumulate_grad(-self.cfg.step_size, 0.0, &mut self.grad_acc, 0);
    self.operator.update_param(1.0, 1.0, &mut self.grad_acc, 0);
  }

  fn eval(&mut self, epoch_size: usize, samples: &mut Iterator<Item=Episode<E>>) {
  }
}

/*impl<E, Op> OptStats<()> for PolicyGradWorker<E, Op>
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
