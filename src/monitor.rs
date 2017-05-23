use env::{MultiEnv, MultiEnvDiscrete};
use features::{MultiEnvObserve};

use stopwatch::{Stopwatch};

use rand::{Rng};
use std::cell::{Cell};
//use std::rc::{Rc};
//use std::sync::{Arc};
use std::sync::mpsc::{SyncSender, Receiver, RecvTimeoutError, sync_channel};
use std::thread::{JoinHandle, spawn};
use std::time::{Duration};

#[derive(Clone, Copy, Debug)]
pub enum MonitorAction {
  CtrlStop,
  StartEvent(MonitorEvent),
  EndEvent(MonitorEvent),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MonitorEvent {
  Shutdown,
  Reset,
  Step,
  IsTerminal,
  GetNumPlayers,
  GetIsActivePlayer,
  GetIsLegalAction,
  Observe,
}

pub struct MonitorWorker {
  //stopwatch:    Stopwatch,
  rx:   Receiver<(u64, MonitorAction)>,
  active_event: Option<(u64, MonitorEvent)>,
}

impl MonitorWorker {
  pub fn runloop(&mut self) {
    loop {
      match self.rx.recv_timeout(Duration::from_millis(5000)) {
        Err(RecvTimeoutError::Timeout) => {
          if let Some((tick, event)) = self.active_event {
            panic!("PANIC: MonitorEnv: timed out during an event: {} {:?}",
                tick, event);
          }
        }
        Err(RecvTimeoutError::Disconnected) => {
          break;
        }
        Ok((tick, action)) => {
          match action {
            MonitorAction::CtrlStop => {
              break;
            }
            MonitorAction::StartEvent(event) => {
              assert!(self.active_event.is_none());
              self.active_event = Some((tick, event));
            }
            MonitorAction::EndEvent(event) => {
              assert!(self.active_event.is_some());
              assert_eq!(self.active_event.unwrap().0, tick);
              assert_eq!(self.active_event.unwrap().1, event);
              self.active_event = None;
            }
          }
        }
      }
    }
  }
}

pub struct MonitorEnv<E> {
  tick: Cell<u64>,
  tx:   SyncSender<(u64, MonitorAction)>,
  h:    Option<JoinHandle<()>>,
  env:  E,
}

impl<E> Drop for MonitorEnv<E> {
  fn drop(&mut self) {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorAction::CtrlStop)).unwrap();
    self.h.take().unwrap().join().unwrap();
  }
}

impl<E> Default for MonitorEnv<E> where E: MultiEnv {
  fn default() -> Self {
    let (tx, rx) = sync_channel(1024);
    let h = spawn(move || {
      let mut worker = MonitorWorker{
        //stopwatch:  Stopwatch::new(),
        rx: rx,
        active_event:   None,
      };
      worker.runloop();
    });
    MonitorEnv{
      tick: Cell::new(0),
      tx:   tx,
      h:    Some(h),
      env:  E::default(),
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
    self.tx.send((t, MonitorAction::StartEvent(MonitorEvent::Reset))).unwrap();
    self.env.reset(restart, rng);
    self.tx.send((t, MonitorAction::EndEvent(MonitorEvent::Reset))).unwrap();
  }

  fn step(&self, multi_action: &[Option<Self::Action>]) -> Result<Vec<Option<Self::Response>>, ()> {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorAction::StartEvent(MonitorEvent::Step))).unwrap();
    let res = self.env.step(multi_action);
    self.tx.send((t, MonitorAction::EndEvent(MonitorEvent::Step))).unwrap();
    res
  }

  fn is_terminal(&self) -> bool {
    self.tick.set(self.tick.get() + 1);
    let t = self.tick.get();
    self.tx.send((t, MonitorAction::StartEvent(MonitorEvent::IsTerminal))).unwrap();
    let res = self.env.is_terminal();
    self.tx.send((t, MonitorAction::EndEvent(MonitorEvent::IsTerminal))).unwrap();
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
    self.tx.send((t, MonitorAction::StartEvent(MonitorEvent::Observe))).unwrap();
    let res = self.env.observe(observer_rank, rng);
    self.tx.send((t, MonitorAction::EndEvent(MonitorEvent::Observe))).unwrap();
    res
  }
}
