use env::{MultiEnv, MultiEnvDiscrete};
use features::{MultiEnvObserve};

use rng::xorshift::{Xorshiftplus128Rng};
use stopwatch::{Stopwatch};

use rand::{Rng, SeedableRng};
use std::cell::{Cell, RefCell};
use std::marker::{PhantomData};
//use std::rc::{Rc};
//use std::sync::{Arc};
use std::sync::mpsc::{SyncSender, Receiver, RecvTimeoutError, sync_channel};
use std::thread::{JoinHandle, spawn};
use std::time::{Duration};

#[derive(Clone, Default)]
pub struct ObjectEnv<E> {
  env:  E,
}

impl<E> MultiEnv for ObjectEnv<E> where E: MultiEnv {
  type Restart = E::Restart;
  type Action = E::Action;
  type Response = E::Response;

  fn shutdown(&self) {
    self.env.shutdown();
  }

  fn reset<R>(&self, restart: &Self::Restart, rng: &mut R) where R: Rng + Sized {
    self.env.reset(restart, rng);
  }

  fn step(&self, multi_action: &[Option<Self::Action>]) -> Result<Vec<Option<Self::Response>>, ()> {
    self.env.step(multi_action)
  }

  fn is_terminal(&self) -> bool {
    self.env.is_terminal()
  }

  fn num_players(&self) -> usize {
    self.env.num_players()
  }

  fn is_active_player(&self, player_rank: usize) -> bool {
    self.env.is_active_player(player_rank)
  }

  fn is_legal_multi_action(&self, multi_action: &[Option<Self::Action>]) -> bool {
    self.env.is_legal_multi_action(multi_action)
  }
}

impl<E> MultiEnvDiscrete for ObjectEnv<E> where E: MultiEnv {
  default fn num_discrete_actions(&self) -> usize {
    unimplemented!();
  }

  default fn get_discrete_action(&self, player_rank: usize, act_idx: u32) -> Self::Action {
    unimplemented!();
  }

  default fn get_discrete_action_index(&self, player_rank: usize, action: &Self::Action) -> u32 {
    unimplemented!();
  }
}

impl<E> MultiEnvDiscrete for ObjectEnv<E> where E: MultiEnvDiscrete {
  fn num_discrete_actions(&self) -> usize {
    self.env.num_discrete_actions()
  }

  fn get_discrete_action(&self, player_rank: usize, act_idx: u32) -> Self::Action {
    self.env.get_discrete_action(player_rank, act_idx)
  }

  fn get_discrete_action_index(&self, player_rank: usize, action: &Self::Action) -> u32 {
    self.env.get_discrete_action_index(player_rank, action)
  }
}

impl<E, Obs> MultiEnvObserve<Obs> for ObjectEnv<E> where E: MultiEnv {
  default fn observe<R>(&self, observer_rank: usize, rng: &mut R) -> Obs where R: Rng + Sized {
    unimplemented!();
  }
}

impl<E, Obs> MultiEnvObserve<Obs> for ObjectEnv<E> where E: MultiEnvObserve<Obs> {
  fn observe<R>(&self, observer_rank: usize, rng: &mut R) -> Obs where R: Rng + Sized {
    self.env.observe(observer_rank, rng)
  }
}

#[derive(Clone, Copy, Debug)]
pub enum MonitorOp {
  CtrlStop,
  StartEvent(MonitorEvent),
  EndEvent(MonitorEvent),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MonitorEvent {
  Construct,
  Shutdown,
  Reset,
  Step,
  IsTerminal,
  /*GetNumPlayers,
  GetIsActivePlayer,
  GetIsLegalAction,*/
  Observe,
}

pub struct MonitorEnvWorker {
  ctrl_rx:      Receiver<(u64, MonitorOp)>,
  active_event: Option<(u64, MonitorEvent)>,
  stopwatch:    Stopwatch,
  ev_counts:    Vec<i64>,
  ev_timings:   Vec<f64>,
}

impl MonitorEnvWorker {
  pub fn runloop(&mut self) {
    loop {
      match self.ctrl_rx.recv_timeout(Duration::from_millis(5000)) {
        Err(RecvTimeoutError::Timeout) => {
          if let Some((tick, event)) = self.active_event {
            panic!("PANIC: MonitorEnvWorker: timed out during an event: {} {:?}",
                tick, event);
          }
        }
        Err(RecvTimeoutError::Disconnected) => {
          break;
        }
        Ok((tick, op)) => {
          match op {
            MonitorOp::CtrlStop => {
              break;
            }
            MonitorOp::StartEvent(event) => {
              self.stopwatch.lap();
              assert!(self.active_event.is_none());
              self.active_event = Some((tick, event));
              /*if self.ev_counts.len() >= 3 {
                if self.ev_counts[2] % 100 == 0 {
                  for &event in [
                      MonitorEvent::Construct,
                      MonitorEvent::Reset,
                      MonitorEvent::Step,
                      MonitorEvent::IsTerminal,
                      MonitorEvent::Observe,
                  ].iter() {
                    println!("DEBUG: MonitorEnvWorker: stats: event: {:?} count: {} avg time: {:.6}",
                        event, self.ev_counts[event as usize], self.ev_timings[event as usize]);
                  }
                }
              }*/
            }
            MonitorOp::EndEvent(event) => {
              assert!(self.active_event.is_some());
              assert_eq!(self.active_event.unwrap().0, tick);
              assert_eq!(self.active_event.unwrap().1, event);
              self.active_event = None;
              if self.ev_counts.len() < (event as usize) + 1 {
                self.ev_counts.resize((event as usize) + 1, 0);
                self.ev_timings.resize((event as usize) + 1, 0.0);
              }
              assert_eq!(self.ev_counts.len(), self.ev_timings.len());
              self.ev_counts[event as usize] += 1;
              self.ev_timings[event as usize] += 1.0 / self.ev_counts[event as usize] as f64 * (self.stopwatch.lap().elapsed() - self.ev_timings[event as usize]);
            }
          }
        }
      }
    }
  }
}

pub struct MonitorEnv<E> {
  tick: Cell<u64>,
  tx:   SyncSender<(u64, MonitorOp)>,
  h:    Option<JoinHandle<()>>,
  env:  E,
}

impl<E> Drop for MonitorEnv<E> {
  fn drop(&mut self) {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorOp::CtrlStop)).ok();
    self.h.take().unwrap().join().unwrap();
  }
}

impl<E> Default for MonitorEnv<E> where E: MultiEnv + Default {
  fn default() -> Self {
    let (tx, rx) = sync_channel(1024);
    let h = spawn(move || {
      let mut worker = MonitorEnvWorker{
        ctrl_rx:    rx,
        active_event:   None,
        stopwatch:  Stopwatch::new(),
        ev_counts:  vec![],
        ev_timings: vec![],
      };
      worker.runloop();
    });
    let tick = Cell::new(0);
    tick.set(tick.get() + 1);
    let t = tick.get();
    tx.send((t, MonitorOp::StartEvent(MonitorEvent::Construct))).unwrap();
    let env = E::default();
    tx.send((t, MonitorOp::EndEvent(MonitorEvent::Construct))).unwrap();
    MonitorEnv{
      tick: tick,
      tx:   tx,
      h:    Some(h),
      env:  env,
    }
  }
}

impl<E> MultiEnv for MonitorEnv<E> where E: MultiEnv {
  type Restart = E::Restart;
  type Action = E::Action;
  type Response = E::Response;

  fn shutdown(&self) {
    self.env.shutdown();
  }

  fn reset<R>(&self, restart: &Self::Restart, rng: &mut R) where R: Rng + Sized {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorOp::StartEvent(MonitorEvent::Reset))).unwrap();
    self.env.reset(restart, rng);
    self.tx.send((t, MonitorOp::EndEvent(MonitorEvent::Reset))).unwrap();
  }

  fn step(&self, multi_action: &[Option<Self::Action>]) -> Result<Vec<Option<Self::Response>>, ()> {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorOp::StartEvent(MonitorEvent::Step))).unwrap();
    let res = self.env.step(multi_action);
    self.tx.send((t, MonitorOp::EndEvent(MonitorEvent::Step))).unwrap();
    res
  }

  fn is_terminal(&self) -> bool {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorOp::StartEvent(MonitorEvent::IsTerminal))).unwrap();
    let res = self.env.is_terminal();
    self.tx.send((t, MonitorOp::EndEvent(MonitorEvent::IsTerminal))).unwrap();
    res
  }

  fn num_players(&self) -> usize {
    self.env.num_players()
  }

  fn is_active_player(&self, player_rank: usize) -> bool {
    self.env.is_active_player(player_rank)
  }

  fn is_legal_multi_action(&self, multi_action: &[Option<Self::Action>]) -> bool {
    self.env.is_legal_multi_action(multi_action)
  }
}

impl<E> MultiEnvDiscrete for MonitorEnv<E> where E: MultiEnvDiscrete {
  fn num_discrete_actions(&self) -> usize {
    self.env.num_discrete_actions()
  }

  fn get_discrete_action(&self, player_rank: usize, act_idx: u32) -> Self::Action {
    self.env.get_discrete_action(player_rank, act_idx)
  }

  fn get_discrete_action_index(&self, player_rank: usize, action: &Self::Action) -> u32 {
    self.env.get_discrete_action_index(player_rank, action)
  }
}

impl<E, Obs> MultiEnvObserve<Obs> for MonitorEnv<E> where E: MultiEnvObserve<Obs> {
  fn observe<R>(&self, observer_rank: usize, rng: &mut R) -> Obs where R: Rng + Sized {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorOp::StartEvent(MonitorEvent::Observe))).unwrap();
    let res = self.env.observe(observer_rank, rng);
    self.tx.send((t, MonitorOp::EndEvent(MonitorEvent::Observe))).unwrap();
    res
  }
}

pub enum AsyncEnvReq<Restart, Action> {
  Reset{restart_cfg: Restart, rng_seed: [u64; 2]},
  Step{action: Vec<Option<Action>>},
  IsTerminal,
  NumPlayers,
  //IsActivePlayer{player_rank: usize},
  //IsLegalAction{player_rank: usize}, // FIXME
  NumDiscreteActions,
  GetDiscreteAction{player_rank: usize, act_idx: u32},
  GetDiscreteActionIndex{player_rank: usize, action: Action},
  Observe{observer_rank: usize, rng_seed: [u64; 2]},
}

pub enum AsyncEnvReply<Restart, Action, Response, Obs> {
  Reset{restart_cfg: Restart},
  Step{res: Result<Vec<Option<Response>>, ()>},
  IsTerminal{terminal: bool},
  NumPlayers{num_players: usize},
  //IsActivePlayer{active: bool},
  //IsLegalAction{legal: bool},
  NumDiscreteActions{action_dim: usize},
  GetDiscreteAction{action: Action},
  GetDiscreteActionIndex{act_idx: u32},
  Observe{obs: Obs},
}

//#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AsyncEnvState<Restart> {
  Constructed,
  WaitReset,
  Ready{ready_restart: Restart},
  Active,
}

pub struct AsyncResetEnvWorker<E, Obs> where E: MultiEnv + MultiEnvDiscrete + MultiEnvObserve<Obs>, E::Restart: Clone + Eq {
  rank: usize,
  rng:  Xorshiftplus128Rng,
  rx:   Receiver<(usize, AsyncEnvReq<E::Restart, E::Action>)>,
  tx:   SyncSender<(usize, AsyncEnvReply<E::Restart, E::Action, E::Response, Obs>)>,
  env:  E,
}

impl<E, Obs> AsyncResetEnvWorker<E, Obs> where E: MultiEnv + MultiEnvDiscrete + MultiEnvObserve<Obs>, E::Restart: Clone + Eq {
  pub fn runloop(&mut self) {
    loop {
      match self.rx.recv() {
        Err(_) => {
          break;
        }
        Ok((active_rank, req)) => {
          assert_eq!(active_rank, self.rank);
          let reply: AsyncEnvReply<E::Restart, E::Action, E::Response, Obs> = match req {
            AsyncEnvReq::Reset{ref restart_cfg, rng_seed} => {
              self.rng.reseed(rng_seed);
              self.env.reset(restart_cfg, &mut self.rng);
              AsyncEnvReply::Reset{restart_cfg: restart_cfg.clone()}
            }
            AsyncEnvReq::Step{ref action} => {
              let res = self.env.step(action);
              AsyncEnvReply::Step{res: res}
            }
            AsyncEnvReq::IsTerminal => {
              let terminal = self.env.is_terminal();
              AsyncEnvReply::IsTerminal{terminal: terminal}
            }
            AsyncEnvReq::NumPlayers => {
              let num_players = self.env.num_players();
              AsyncEnvReply::NumPlayers{num_players: num_players}
            }
            AsyncEnvReq::NumDiscreteActions => {
              let action_dim = self.env.num_discrete_actions();
              AsyncEnvReply::NumDiscreteActions{action_dim: action_dim}
            }
            AsyncEnvReq::GetDiscreteAction{player_rank, act_idx} => {
              let action = self.env.get_discrete_action(player_rank, act_idx);
              AsyncEnvReply::GetDiscreteAction{action: action}
            }
            AsyncEnvReq::GetDiscreteActionIndex{player_rank, ref action} => {
              let act_idx = self.env.get_discrete_action_index(player_rank, action);
              AsyncEnvReply::GetDiscreteActionIndex{act_idx: act_idx}
            }
            AsyncEnvReq::Observe{observer_rank, rng_seed} => {
              self.rng.reseed(rng_seed);
              let obs = self.env.observe(observer_rank, &mut self.rng);
              AsyncEnvReply::Observe{obs: obs}
            }
          };
          self.tx.send((self.rank, reply)).unwrap();
        }
      }
    }
  }
}

pub struct AsyncResetEnvImpl<E, Obs> where E: MultiEnv, E::Restart: 'static + Send, E::Action: 'static + Send, E::Response: 'static + Send, Obs: 'static + Send {
  env_ct:   usize,
  //rng:      Xorshiftplus128Rng,
  states:   Vec<AsyncEnvState<E::Restart>>,
  active:   Option<usize>,
  order:    Vec<usize>,
  txs:  Vec<SyncSender<(usize, AsyncEnvReq<E::Restart, E::Action>)>>,
  rx:   Receiver<(usize, AsyncEnvReply<E::Restart, E::Action, E::Response, Obs>)>,
  hs:   Vec<JoinHandle<()>>,
}

pub struct AsyncResetEnv<E, Obs> where E: MultiEnv, E::Restart: 'static + Send, E::Action: 'static + Send, E::Response: 'static + Send, Obs: 'static + Send {
  inner:    RefCell<AsyncResetEnvImpl<E, Obs>>,
  //_m:   PhantomData<fn (E)>,
}

impl<E, Obs> AsyncResetEnv<E, Obs> where E: MultiEnv + MultiEnvDiscrete + MultiEnvObserve<Obs> + Default, E::Restart: 'static + Send + Clone + Eq, E::Action: 'static + Send, E::Response: 'static + Send, Obs: 'static + Send {
  pub fn new(env_ct: usize) -> Self {
    let mut states = vec![];
    let mut order = vec![];
    let mut txs = vec![];
    let mut hs = vec![];
    let (reply_tx, reply_rx) = sync_channel(1024);
    for rank in 0 .. env_ct {
      let (req_tx, req_rx) = sync_channel(1024);
      let reply_tx = reply_tx.clone();
      let h = spawn(move || {
        let mut worker = AsyncResetEnvWorker{
          rank: rank,
          rng:  Xorshiftplus128Rng::zeros(),
          rx:   req_rx,
          tx:   reply_tx,
          env:  E::default(),
        };
        worker.runloop();
      });
      states.push(AsyncEnvState::Constructed);
      order.push(rank);
      txs.push(req_tx);
      hs.push(h);
    }
    AsyncResetEnv{
      inner:    RefCell::new(AsyncResetEnvImpl{
        env_ct: env_ct,
        states: states,
        active: None,
        order:  order,
        txs:    txs,
        rx:     reply_rx,
        hs:     hs,
      }),
    }
  }
}

impl<E, Obs> MultiEnv for AsyncResetEnv<E, Obs> where E: MultiEnv, E::Restart: 'static + Send + Clone + Eq, E::Action: 'static + Send, E::Response: 'static + Send, Obs: 'static + Send {
  type Restart = E::Restart;
  type Action = E::Action;
  type Response = E::Response;

  fn shutdown(&self) {
    unimplemented!();
  }

  fn reset<R>(&self, restart: &Self::Restart, seed_rng: &mut R) where R: Rng + Sized {
    let mut inner = self.inner.borrow_mut();
    if let Some(active_rank) = inner.active {
      assert!(matches!(inner.states[active_rank], AsyncEnvState::Active));
      let seed = [seed_rng.next_u64(), seed_rng.next_u64()];
      inner.txs[active_rank].send((active_rank, AsyncEnvReq::Reset{restart_cfg: restart.clone(), rng_seed: seed})).unwrap();
      inner.states[active_rank] = AsyncEnvState::WaitReset;
      inner.active = None;
    }
    'reset_loop: loop {
      for i in 0 .. inner.env_ct {
        let j = seed_rng.gen_range(i, inner.env_ct);
        if i < j {
          inner.order.swap(i, j);
        }
        let rank = inner.order[i];
        let mut do_reset = false;
        let mut is_ready = false;
        match inner.states[rank] {
          AsyncEnvState::Constructed => {
            do_reset = true;
          }
          AsyncEnvState::WaitReset => {}
          AsyncEnvState::Ready{ref ready_restart} => {
            if ready_restart == restart {
              is_ready = true;
            } else {
              do_reset = true;
            }
          }
          _ => unreachable!(),
        }
        if do_reset {
          let seed = [seed_rng.next_u64(), seed_rng.next_u64()];
          inner.txs[rank].send((rank, AsyncEnvReq::Reset{restart_cfg: restart.clone(), rng_seed: seed})).unwrap();
          inner.states[rank] = AsyncEnvState::WaitReset;
        } else if is_ready {
          inner.states[rank] = AsyncEnvState::Active;
          inner.active = Some(rank);
          break 'reset_loop;
        }
      }
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }

  fn step(&self, multi_action: &[Option<Self::Action>]) -> Result<Vec<Option<Self::Response>>, ()> {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::Step{action: multi_action.to_owned()})).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::Step{res} => {
              assert_eq!(active_rank, rank);
              return res;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }

  fn is_terminal(&self) -> bool {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::IsTerminal)).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::IsTerminal{terminal} => {
              assert_eq!(active_rank, rank);
              return terminal;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }

  fn num_players(&self) -> usize {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::NumPlayers)).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::NumPlayers{num_players} => {
              assert_eq!(active_rank, rank);
              return num_players;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }

  fn is_active_player(&self, player_rank: usize) -> bool {
    unimplemented!();
  }

  fn is_legal_multi_action(&self, multi_action: &[Option<Self::Action>]) -> bool {
    unimplemented!();
  }
}

impl<E, Obs> MultiEnvDiscrete for AsyncResetEnv<E, Obs> where E: MultiEnvDiscrete, E::Restart: 'static + Send + Clone + Eq, E::Action: 'static + Send, E::Response: 'static + Send, Obs: 'static + Send {
  fn num_discrete_actions(&self) -> usize {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::NumDiscreteActions)).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::NumDiscreteActions{action_dim} => {
              assert_eq!(active_rank, rank);
              return action_dim;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }

  fn get_discrete_action(&self, player_rank: usize, act_idx: u32) -> Self::Action {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::GetDiscreteAction{player_rank: player_rank, act_idx: act_idx})).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::GetDiscreteAction{action} => {
              assert_eq!(active_rank, rank);
              return action;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }

  fn get_discrete_action_index(&self, player_rank: usize, action: &Self::Action) -> u32 {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::GetDiscreteActionIndex{player_rank: player_rank, action: action.clone()})).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::GetDiscreteActionIndex{act_idx} => {
              assert_eq!(active_rank, rank);
              return act_idx;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }
}

impl<E, Obs> MultiEnvObserve<Obs> for AsyncResetEnv<E, Obs> where E: MultiEnvObserve<Obs>, E::Restart: 'static + Send + Clone + Eq, E::Action: 'static + Send, E::Response: 'static + Send, Obs: 'static + Send {
  fn observe<R>(&self, observer_rank: usize, rng: &mut R) -> Obs where R: Rng + Sized {
    let mut inner = self.inner.borrow_mut();
    assert!(inner.active.is_some());
    let active_rank = inner.active.unwrap();
    let seed = [rng.next_u64(), rng.next_u64()];
    inner.txs[active_rank].send((active_rank, AsyncEnvReq::Observe{observer_rank: observer_rank, rng_seed: seed})).unwrap();
    loop {
      match inner.rx.recv_timeout(Duration::from_millis(10)) {
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
          // TODO
        }
        Ok((rank, reply)) => {
          match reply {
            AsyncEnvReply::Reset{restart_cfg} => {
              assert!(matches!(inner.states[rank], AsyncEnvState::WaitReset));
              inner.states[rank] = AsyncEnvState::Ready{ready_restart: restart_cfg};
            }
            AsyncEnvReply::Observe{obs} => {
              assert_eq!(active_rank, rank);
              return obs;
            }
            _ => unreachable!(),
          }
        }
      }
    }
  }
}
