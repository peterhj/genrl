use discrete::{DiscreteDist32};
use env::{Env, DiscreteEnv, EnvInputRepr, EnvRepr, EnvConvert, Action, DiscreteAction, Response, Value, Episode, EpisodeStep};
use opt::pg_new::{BasePolicyGrad};

use densearray::prelude::*;
use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::ops::{Deref};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct SgdAdvActorCriticConfig<E, V> where E: 'static + Env, V: Value<Res=E::Response> {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub max_horizon:  usize,
  pub update_steps: Option<usize>,
  pub baseline:     f32,
  pub init_cfg:     E::Init,
  pub value_cfg:    V::Cfg,
}

pub struct SgdAdvActorCriticWorker<E, V, Op>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Op: DiffLoss<SampleItem, IoBuf=[f32]> //+ StochasticPolicy,
{
  cfg:      SgdAdvActorCriticConfig<E, V>,
  grad_sz:  usize,
  rng:      Xorshiftplus128Rng,
  base_pg:  BasePolicyGrad<E, V, Op>,
  operator: Rc<RefCell<Op>>,
  cache:    Vec<SampleItem>,
  param:    Vec<f32>,
  grad:     Vec<f32>,
}

impl<E, V, Op> SgdAdvActorCriticWorker<E, V, Op>
where E: 'static + Env + EnvInputRepr<[f32]> + SampleExtractInput<[f32]> + Clone,
      E::Action: DiscreteAction,
      V: Value<Res=E::Response>,
      Op: DiffLoss<SampleItem, IoBuf=[f32]> //+ StochasticPolicy,
{
  pub fn new(cfg: SgdAdvActorCriticConfig<E, V>, op: Rc<RefCell<Op>>) -> SgdAdvActorCriticWorker<E, V, Op> {
    let batch_sz = cfg.batch_sz;
    let minibatch_sz = cfg.minibatch_sz;
    let max_horizon = cfg.max_horizon;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let base_pg = BasePolicyGrad::new(minibatch_sz, max_horizon, &cfg.init_cfg, &mut rng);
    let grad_sz = op.borrow_mut().diff_param_sz();
    //println!("DEBUG: grad sz: {}", grad_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    SgdAdvActorCriticWorker{
      cfg:      cfg,
      grad_sz:  grad_sz,
      rng:      rng,
      base_pg:  base_pg,
      operator: op,
      cache:    Vec::with_capacity(batch_sz),
      param:    param,
      grad:     grad,
    }
  }

  pub fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut operator = self.operator.borrow_mut();
    operator.init_param(rng);
    operator.store_diff_param(&mut self.param);
    //println!("DEBUG: param: {:?}", self.param);
  }

  pub fn update(&mut self) -> f32 {
    let mut operator = self.operator.borrow_mut();
    self.base_pg.sample_steps(self.cfg.update_steps, &self.cfg.init_cfg, &mut operator, &mut self.rng);
    self.base_pg.fill_values(&self.cfg.value_cfg);
    operator.reset_loss();
    operator.reset_grad();
    operator.next_iteration();
    self.cache.clear();
    //print!("DEBUG: weights: ");
    for (idx, episode) in self.base_pg.episodes.iter().enumerate() {
      for k in self.base_pg.ep_k_offsets[idx] .. episode.horizon() {
        let mut item = SampleItem::new();
        match k {
          0 => {
            let env = episode.init_env.clone();
            let env_repr_dim = env._shape3d();
            item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
            item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
          }
          k => {
            let env = episode.steps[k-1].next_env.clone();
            let env_repr_dim = env._shape3d();
            item.kvs.insert::<SampleExtractInputKey<[f32]>>(env);
            item.kvs.insert::<SampleInputShape3dKey>(env_repr_dim);
          }
        }
        item.kvs.insert::<SampleClassLabelKey>(episode.steps[k].action.idx());
        let w = self.base_pg.step_values[idx][k];
        /*if k == 0 {
          print!("{:?} ", w);
        }*/
        item.kvs.insert::<SampleWeightKey>(w);
        self.cache.push(item);
        if self.cache.len() < self.cfg.batch_sz {
          continue;
        }
        operator.load_batch(&self.cache);
        operator.forward(OpPhase::Learning);
        operator.backward();
        self.cache.clear();
      }
    }
    //println!("");
    if !self.cache.is_empty() {
      operator.load_batch(&self.cache);
      operator.forward(OpPhase::Learning);
      operator.backward();
      self.cache.clear();
    }
    operator.store_grad(&mut self.grad);
    //println!("DEBUG: grad:  {:?}", self.grad);
    // FIXME(20161018): only normalize by minibatch size if all episodes in the
    // minibatch are represented in the policy gradient.
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    self.param.reshape_mut(self.grad_sz).add(-self.cfg.step_size, self.grad.reshape(self.grad_sz));
    operator.load_diff_param(&mut self.param);
    //println!("DEBUG: param: {:?}", self.param);
    let mut avg_value = 0.0;
    for idx in 0 .. self.cfg.minibatch_sz {
      avg_value += self.base_pg.step_values[idx][self.base_pg.ep_k_offsets[idx]];
    }
    avg_value /= self.cfg.minibatch_sz as f32;
    avg_value
  }
}
