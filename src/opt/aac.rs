use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Episode, EpisodeStep};
use opt::pg::{EpisodeStepSample, BasePgWorker};

use operator::prelude::*;
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
  pub p_step_size:  f32,
  pub v_step_size:  f32,
  pub momentum:     f32,
}

pub struct SgdAacWorker<E, PolicyOp, ValueOp, R>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      R: Rng,
      PolicyOp: DiffOperator<f32, Rng=R> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
      ValueOp: DiffOperator<f32, Rng=R> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  cfg:      SgdAacConfig,
  policy:   PolicyOp,
  value:    ValueOp,
  cache:    Vec<EpisodeStepSample<E>>,
  base_pg:  BasePgWorker<E, PolicyOp>,
  pg_acc:   Vec<f32>,
  vg_acc:   Vec<f32>,
}

impl<E, PolicyOp, ValueOp, R> SgdAacWorker<E, PolicyOp, ValueOp, R>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      R: Rng,
      PolicyOp: DiffOperator<f32, Rng=R> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
      ValueOp: DiffOperator<f32, Rng=R> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  pub fn new(cfg: SgdAacConfig, /*op: Op,*/ policy: PolicyOp, value: ValueOp) -> SgdAacWorker<E, PolicyOp, ValueOp, R> {
    let pg_sz = policy.diff_param_sz();
    let mut pg_acc = Vec::with_capacity(pg_sz);
    pg_acc.resize(pg_sz, 0.0);
    let vg_sz = value.diff_param_sz();
    let mut vg_acc = Vec::with_capacity(pg_sz);
    vg_acc.resize(pg_sz, 0.0);
    SgdAacWorker{
      cfg:      cfg,
      policy:   policy,
      value:    value,
      cache:    Vec::with_capacity(cfg.batch_sz),
      base_pg:  BasePgWorker::new(cfg.batch_sz, cfg.max_horizon),
      pg_acc:   pg_acc,
      vg_acc:   vg_acc,
    }
  }

  pub fn sample<R2>(&mut self, episodes: &mut [Episode<E>], init_cfg: &E::Init, rng: &mut R2) where R2: Rng {
    self.base_pg.sample(&mut self.policy, &mut self.cache, episodes, init_cfg, rng);
  }
}

impl<E, PolicyOp, ValueOp, R> OptWorker<f32, Episode<E>> for SgdAacWorker<E, PolicyOp, ValueOp, R>
where E: Env + EnvRepr<f32> + Clone,
      E::Action: DiscreteAction,
      R: Rng,
      PolicyOp: DiffOperator<f32, Rng=R> + DiffOperator<f32> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
      ValueOp: DiffOperator<f32, Rng=R> + DiffOperator<f32> + DiffOperatorInput<f32, EpisodeStepSample<E>> + DiffOperatorOutput<f32, RwSlice<f32>>,
{
  type Rng = R;

  fn init_param(&mut self, rng: &mut Self::Rng) {
    self.policy.init_param(rng);
    self.value.init_param(rng);
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) { unimplemented!(); }
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) { unimplemented!(); }
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) { unimplemented!(); }

  fn step(&mut self, episodes: &mut Iterator<Item=Episode<E>>) {
    self.policy.reset_loss();
    self.policy.reset_grad();
    self.value.reset_loss();
    self.value.reset_grad();
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
        self.cache.push(sample);
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }
        self.value.load_data(&self.cache);
        self.value.forward(OpPhase::Learning);
        self.value.backward();
        let values = self.value.get_output();
        for (idx, sample) in self.cache.iter_mut().enumerate() {
          sample.set_baseline(values.borrow()[idx]);
          sample.init_weight();
          sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
        }
        self.policy.load_data(&self.cache);
        self.policy.forward(OpPhase::Learning);
        self.policy.backward();
        self.cache.clear();
      }
    }
    if !self.cache.is_empty() {
      self.value.load_data(&self.cache);
      self.value.forward(OpPhase::Learning);
      self.value.backward();
      let values = self.value.get_output();
      for (idx, sample) in self.cache.iter_mut().enumerate() {
        sample.set_baseline(values.borrow()[idx]);
        sample.init_weight();
        sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
      }
      self.policy.load_data(&self.cache);
      self.policy.forward(OpPhase::Learning);
      self.policy.backward();
      self.cache.clear();
    }
    self.policy.accumulate_grad(-self.cfg.p_step_size, 0.0, &mut self.pg_acc, 0);
    self.policy.update_param(1.0, 1.0, &mut self.pg_acc, 0);
    self.value.accumulate_grad(-self.cfg.v_step_size, 0.0, &mut self.vg_acc, 0);
    self.value.update_param(1.0, 1.0, &mut self.vg_acc, 0);
  }

  fn eval(&mut self, epoch_size: usize, samples: &mut Iterator<Item=Episode<E>>) {
  }
}

pub struct RmspropAacGradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub max_horizon:  usize,
  pub p_step_size:  f32,
  pub v_step_size:  f32,
  pub epsilon:      f32,
}
