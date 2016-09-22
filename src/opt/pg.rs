use env::{Env, DiscreteEnv, EnvRepr, DiscreteAction, Response, Episode};

use operator::prelude::*;
use operator::data::{SampleInput, SampleExtractInput, SampleClass, SampleWeight};
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct EpisodeStepSample<E> where E: Env {
  pub env:      Rc<RefCell<E>>,
  pub act_idx:  u32,
  pub suffix_r: Option<E::Response>,
  weight:       Option<f32>,
}

impl<E> EpisodeStepSample<E> where E: Env {
  pub fn new(env: Rc<RefCell<E>>, act_idx: u32, suffix_r: Option<E::Response>) -> EpisodeStepSample<E> {
    EpisodeStepSample{
      env:          env,
      act_idx:      act_idx,
      suffix_r:     suffix_r,
      weight:       None,
    }
  }

  pub fn init_weight(&mut self, constant_baseline: f32) {
    self.weight = Some(self.suffix_r.map(|r| r.as_scalar()).unwrap() - constant_baseline);
  }
}

impl<E> SampleExtractInput<f32> for EpisodeStepSample<E> where E: Env + EnvRepr<f32> {
  fn extract_input(&self, output: &mut [f32]) {
    self.env.borrow_mut().extract_observable(output);
  }
}

impl<E> SampleClass for EpisodeStepSample<E> where E: Env {
  fn class(&self) -> Option<u32> {
    Some(self.act_idx)
  }
}

impl<E> SampleWeight for EpisodeStepSample<E> where E: Env {
  fn weight(&self) -> Option<f32> {
    self.suffix_r.map(|x| x.as_scalar() * self.weight.unwrap_or(1.0))
  }

  fn mix_weight(&mut self, w: f32) {
    self.weight = Some(self.weight.unwrap_or(1.0) * w);
  }
}

/*pub trait DiscretePolicy {
  type Env;

  fn eval(&mut self, env: &mut Self::Env, action_dist: &mut [f32]);
}

pub struct DiscretePolicyGrad<E, P> where E: Env, P: DiscretePolicy<Env=E> {
  _marker: PhantomData<(E, P)>,
}

impl<E, P> DiscretePolicyGrad<E, P> where E: Env, P: DiscretePolicy<Env=E> {
}*/

/*pub trait DiffPolicyOutput {
  fn action_probabilities(&self) -> &[f32];
}*/

/*pub struct DiffPolicy<E, T, S, Op>
where E: Env + EnvRepr<f32>,
      //Out: DiffPolicyOutput,
      Op: Operator<T, S> + DiffOperatorOutput<T, f32>,
{
  _marker:  PhantomData<(E, T, S)>,
}*/

pub struct PolicyGradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub max_horizon:  usize,
  pub baseline:     f32,
}

pub struct PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32>,
      E::Action: DiscreteAction,
      //Out: DiffPolicyOutput,
      Op: Operator<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, f32>,
{
  //policy:   DiffPolicy<E, T, S, Op>,
  cfg:      PolicyGradConfig,
  pub operator: Op,
  cache:    Vec<EpisodeStepSample<E>>,
  grad_acc: Vec<f32>,
  _marker:  PhantomData<(E, Op)>,
}

impl<E, Op> PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32>,
      E::Action: DiscreteAction,
      //Out: DiffPolicyOutput,
      Op: Operator<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, f32>,
{
  pub fn new(op: Op) -> PolicyGradWorker<E, Op> {
    unimplemented!();
  }
}

impl<E, Op> OptWorker<f32, Episode<E>> for PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32>,
      E::Action: DiscreteAction,
      //Out: DiffPolicyOutput,
      Op: Operator<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, f32>,
{
  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
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
          0 => EpisodeStepSample::new(episode.init_env.clone(),             episode.steps[0].action.idx(),  episode.suffixes[0]),
          k => EpisodeStepSample::new(episode.steps[k-1].next_env.clone(),  episode.steps[k].action.idx(),  episode.suffixes[k]),
        };
        assert!(sample.weight().is_some());
        // FIXME(20160920): baseline.
        sample.init_weight(self.cfg.baseline);
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

impl<E, Op> OptStats<()> for PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32>,
      E::Action: DiscreteAction,
      //Out: DiffPolicyOutput,
      Op: Operator<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, f32>,
{
  fn reset_opt_stats(&mut self) {
    unimplemented!();
  }

  fn get_opt_stats(&self) -> &() {
    unimplemented!();
  }
}
