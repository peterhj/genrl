use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Episode, EpisodeStep};
use opt::pg::{EpisodeStepSample};

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

#[derive(Clone, Copy, Debug)]
pub struct SgdAacConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub max_horizon:  usize,
  pub step_size:    f32,
  pub momentum:     f32,
}

pub struct SgdAacWorker<E, Op>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
{
  cfg:      SgdAacConfig,
  operator: Op,
  cache:    Vec<EpisodeStepSample<E>>,
  grad_acc: Vec<f32>,
  act_dist: DiscreteDist32,
}

impl<E, Op> SgdAacWorker<E, Op>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
{
  pub fn new(cfg: SgdAacConfig, op: Op) -> SgdAacWorker<E, Op> {
    let param_sz = op.param_len();
    let mut grad_acc = Vec::with_capacity(param_sz);
    unsafe { grad_acc.set_len(param_sz) };
    SgdAacWorker{
      cfg:      cfg,
      operator: op,
      cache:    Vec::with_capacity(cfg.batch_sz),
      grad_acc: grad_acc,
      act_dist: DiscreteDist32::new(<E::Action as Action>::dim()),
    }
  }

  pub fn sample<R>(&mut self, episodes: &mut [Episode<E>], init_cfg: &E::Init, rng: &mut R) where R: Rng {
    let action_dim = <E::Action as Action>::dim();
    for episode in episodes {
      episode.reset(init_cfg, rng);
      for k in episode.steps.len() .. self.cfg.max_horizon {
        if episode.terminated() {
          break;
        }
        let prev_env = match k {
          0 => episode.init_env.clone(),
          k => episode.steps[k-1].next_env.clone(),
        };
        //let mut next_env: E = EnvConvert::from_env(&*prev_env.borrow());
        let mut next_env: E = prev_env.borrow().clone();
        let sample = EpisodeStepSample::new(prev_env, None, None);
        self.cache.clear();
        self.cache.push(sample);

        self.operator.load_data(&self.cache);
        self.operator.forward(OpPhase::Inference);

        let output = self.operator.get_output();
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
      episode.fill_suffixes();
    }
  }
}

impl<E, Op> OptWorker<f32, Episode<E>> for SgdAacWorker<E, Op>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      Op: DiffOperatorIo<f32, EpisodeStepSample<E>, RwSlice<f32>>,
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
        assert!(sample.weight().is_some());
        sample.init_weight(0.0); // FIXME: Q-baseline.
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

pub struct RmspropAacGradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub max_horizon:  usize,
  pub step_size:    f32,
  pub epsilon:      f32,
}
