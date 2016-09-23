use discrete::{DiscreteDist32, DiscreteSampler};
use env::{Env, DiscreteEnv, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Episode, EpisodeStep};

use operator::prelude::*;
use operator::data::{SampleInput, SampleExtractInput, SampleClass, SampleWeight};
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct EpisodeSample<E> where E: Env {
  env:      Rc<RefCell<E>>,
  act_idx:  Option<u32>,
  suffix:   Option<E::Response>,
  weight:   Option<f32>,
}

impl<E> EpisodeSample<E> where E: Env + EnvConvert<E> {
  pub fn initial(init_env: Rc<RefCell<E>>) -> EpisodeSample<E> {
    EpisodeSample{
      env:      init_env,
      act_idx:  None,
      suffix:   None,
      weight:   None,
    }
  }

  pub fn new(env: Rc<RefCell<E>>, act_idx: u32, suffix: Option<E::Response>) -> EpisodeSample<E> {
    EpisodeSample{
      env:      env,
      act_idx:  Some(act_idx),
      suffix:   suffix,
      weight:   None,
    }
  }

  pub fn reset_initial(&mut self, init_env: Rc<RefCell<E>>) {
    self.env.borrow_mut().clone_from_env(&*init_env.borrow());
    self.act_idx = None;
    self.suffix = None;
    self.weight = None;
  }

  pub fn init_weight(&mut self, suffix: Option<E::Response>, constant_baseline: f32) {
    self.suffix = suffix;
    self.weight = Some(self.suffix.map(|r| r.as_scalar()).unwrap() - constant_baseline);
  }
}

impl<E> SampleExtractInput<f32> for EpisodeSample<E> where E: Env + EnvRepr<f32> {
  fn extract_input(&self, output: &mut [f32]) {
    self.env.borrow_mut().extract_observable(output);
  }
}

impl<E> SampleClass for EpisodeSample<E> where E: Env {
  fn class(&self) -> Option<u32> {
    self.act_idx
  }
}

impl<E> SampleWeight for EpisodeSample<E> where E: Env {
  fn weight(&self) -> Option<f32> {
    self.suffix.map(|x| x.as_scalar() * self.weight.unwrap_or(1.0))
  }

  fn mix_weight(&mut self, w: f32) {
    self.weight = Some(self.weight.unwrap_or(1.0) * w);
  }
}

pub struct EpisodeStepSample<E> where E: Env {
  pub env:      Rc<RefCell<E>>,
  pub act_idx:  Option<u32>,
  pub suffix_r: Option<E::Response>,
  weight:       Option<f32>,
}

impl<E> EpisodeStepSample<E> where E: Env {
  pub fn new(env: Rc<RefCell<E>>, act_idx: Option<u32>, suffix_r: Option<E::Response>) -> EpisodeStepSample<E> {
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
    self.act_idx
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

#[derive(Clone, Copy, Debug)]
pub struct PolicyGradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub max_horizon:  usize,
  pub baseline:     f32,
}

pub struct PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
{
  //policy:   DiffPolicy<E, T, S, Op>,
  cfg:      PolicyGradConfig,
  pub operator: Op,
  cache:    Vec<EpisodeStepSample<E>>,
  grad_acc: Vec<f32>,
  act_dist: DiscreteSampler,
  //act_dist: DiscreteDist32,
  //episodes: Vec<EpisodeSample<E>>,
}

impl<E, Op> PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
{
  pub fn new(cfg: PolicyGradConfig, op: Op) -> PolicyGradWorker<E, Op> {
    let param_sz = op.param_len();
    let mut grad_acc = Vec::with_capacity(param_sz);
    unsafe { grad_acc.set_len(param_sz) };
    PolicyGradWorker{
      cfg:      cfg,
      operator: op,
      cache:    Vec::with_capacity(cfg.batch_sz),
      grad_acc: grad_acc,
      act_dist: DiscreteSampler::with_capacity(<E::Action as Action>::dim()),
      //act_dist: DiscreteDist32::new(<E::Action as Action>::dim()),
    }
  }

  pub fn sample<R>(&mut self, episodes: &mut [Episode<E>], init_cfg: &E::Init, rng: &mut R) where R: Rng {
    let action_dim = <E::Action as Action>::dim();
    for episode in episodes {
      self.cache.clear();
      //episode.init_env.borrow_mut().reset(init_cfg, rng);
      episode.reset(init_cfg, rng);
      for k in episode.steps.len() .. self.cfg.max_horizon {
        if episode.terminated() {
          break;
        }
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        //if prev_env.borrow_mut().is_terminal() {
        let mut next_env: E = EnvConvert::from_env(&*prev_env.borrow());
        let sample = EpisodeStepSample::new(prev_env, None, None);
        self.cache.push(sample);

        self.operator.load_data(&self.cache);
        self.operator.forward(OpPhase::Inference);
        self.cache.clear();

        let output = self.operator.get_output();
        //println!("DEBUG: policy output ({}): {:?}", k, (&output.borrow()[ .. action_dim]));
        self.act_dist.reset(&output.borrow()[ .. action_dim]);
        let act_idx = self.act_dist.sample(rng).unwrap();
        let action = <E::Action as DiscreteAction>::from_idx(act_idx as u32);
        if let Ok(res) = next_env.step(&action) {
          //println!("DEBUG: sample step ({}): res: {:?}", k, res);
          episode.steps.push(EpisodeStep{
            action:   action,
            res:      res,
            next_env: Rc::new(RefCell::new(next_env)),
          });
        } else {
          panic!();
        }
      }
      episode.fill_suffixes();
      /*let horizon = episode.steps.len()-1;
      println!("DEBUG: sample: suffix[0]: {} {:?} {:?}",
          horizon,
          episode.suffixes[0],
          episode.suffixes[horizon],
      );*/
    }
  }
}

impl<E, Op> OptWorker<f32, Episode<E>> for PolicyGradWorker<E, Op>
where E: Env + EnvRepr<f32> + EnvConvert<E>,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
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
          0 => EpisodeStepSample::new(
              episode.init_env.clone(),
              Some(episode.steps[0].action.idx()),
              episode.suffixes[0]),
          k => EpisodeStepSample::new(
              episode.steps[k-1].next_env.clone(),
              Some(episode.steps[k].action.idx()),
              episode.suffixes[k]),
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
