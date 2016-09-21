use env::{Env, EnvRepr, DiscreteAction, Response, EpisodeTraj};

use operator::prelude::*;
use operator::data::{SampleInput, SampleClass, SampleWeight};
use operator::rw::{ReadBuffer, WriteBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cmp::{min};
use std::marker::{PhantomData};

pub struct EpisodeStepSample<R> where R: Response {
  pub obs:      Vec<f32>,
  //pub env:      Rc<RefCell<E>>,
  pub act_idx:  u32,
  pub res:      R,
  suffix_r:     Option<R>,
  extra_weight: Option<f32>,
}

impl<R> EpisodeStepSample<R> where R: Response {
  pub fn new(obs: Vec<f32>, act_idx: u32, res: R) -> EpisodeStepSample<R> {
    EpisodeStepSample{
      obs:          obs,
      act_idx:      act_idx,
      res:          res,
      suffix_r:     None,
      extra_weight: None,
    }
  }

  pub fn append_suffix(&mut self, mut suffix_r: R) {
    suffix_r.lreduce(self.res);
    self.suffix_r = Some(suffix_r);
  }
}

impl<R> SampleInput<f32> for EpisodeStepSample<R> where R: Response {
  fn input(&self) -> &[f32] {
    &self.obs
  }
}

impl<R> SampleClass for EpisodeStepSample<R> where R: Response {
  fn class(&self) -> Option<u32> {
    Some(self.act_idx)
  }
}

impl<R> SampleWeight for EpisodeStepSample<R> where R: Response {
  fn weight(&self) -> Option<f32> {
    self.suffix_r.map(|x| x.as_scalar() * self.extra_weight.unwrap_or(1.0))
  }

  fn mix_weight(&mut self, w: f32) {
    self.extra_weight = Some(self.extra_weight.unwrap_or(1.0) * w);
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

pub trait DiffPolicyOutput {
  fn action_probabilities(&self) -> &[f32];
}

pub struct DiffPolicy<E, T, S, Out, Op>
where E: Env + EnvRepr<f32>,
      Out: DiffPolicyOutput,
      Op: Operator<T, S, Output=Out>
{
  op:       Op,
  _marker:  PhantomData<(E, T, S, Out)>,
}

pub struct PolicyGradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub max_horizon:  usize,
}

pub struct PolicyGradWorker<E, T, Out, Op>
where E: Env + EnvRepr<f32>,
      //S: SampleInput<f32> + SampleClass + SampleWeight,
      Out: DiffPolicyOutput,
      Op: Operator<T, EpisodeStepSample<E::Response>, Output=Out>
{
  //policy:   DiffPolicy<E, T, S, Out, Op>,
  cfg:      PolicyGradConfig,
  cache:    Vec<EpisodeStepSample<E::Response>>,
  grad_acc: Vec<f32>,
  _marker:  PhantomData<(E, T, Out, Op)>,
}

impl<E, T, Out, Op> PolicyGradWorker<E, T, Out, Op>
where E: Env + EnvRepr<f32>,
      //S: SampleInput<f32> + SampleClass + SampleWeight,
      Out: DiffPolicyOutput,
      Op: Operator<T, EpisodeStepSample<E::Response>, Output=Out>
{
  pub fn new() -> PolicyGradWorker<E, T, Out, Op> {
    unimplemented!();
  }
}

impl<E, T, Out, Op> OptWorker<T, EpisodeStepSample<E::Response>> for PolicyGradWorker<E, T, Out, Op>
where E: Env + EnvRepr<f32>,
      //S: SampleInput<f32> + SampleClass + SampleWeight,
      Out: DiffPolicyOutput,
      Op: Operator<T, EpisodeStepSample<E::Response>, Output=Out>
{
  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<T>) { unimplemented!(); }
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<T>) { unimplemented!(); }
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<T>) { unimplemented!(); }

  fn step(&mut self, samples: &mut Iterator<Item=EpisodeStepSample<E::Response>>) {
    // FIXME(20160920): one sample = one episode step?

    //self.operator.reset_loss();
    //self.operator.reset_grad();
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    for batch in 0 .. num_batches {
      let actual_batch_sz = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
      self.cache.clear();
      for mut sample in samples.take(actual_batch_sz) {
        sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
        self.cache.push(sample);
      }
      //self.operator.load_data(&self.cache);
      //self.operator.forward(OpPhase::Learning);
      //self.operator.backward();
    }
    //self.operator.accumulate_grad(-self.cfg.step_size, 0.0, &mut self.grad_acc, 0);
    //self.operator.update_param(1.0, 1.0, &mut self.grad_acc, 0);
  }

  fn eval(&mut self, epoch_size: usize, samples: &mut Iterator<Item=EpisodeStepSample<E::Response>>) {
  }
}
