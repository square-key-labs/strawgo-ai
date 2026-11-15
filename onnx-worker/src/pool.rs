use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use crate::smart_turn::SmartTurnModel;
use crate::vad::{SileroModel, VadState};

pub struct VadJob {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
    pub state: VadState,
}

pub struct VadOutcome {
    pub result: Result<f32>,
    pub samples: Vec<i16>,
    pub state: VadState,
}

pub struct TurnJob {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
    pub speech_start_ms: u32,
}

pub struct TurnOutcome {
    pub result: Result<f32>,
    pub samples: Vec<i16>,
}

enum Job {
    Vad(VadJob, oneshot::Sender<VadOutcome>),
    Turn(TurnJob, oneshot::Sender<TurnOutcome>),
}

pub struct InferencePool {
    tx: mpsc::Sender<Job>,
}

impl InferencePool {
    pub fn start(vad_model: Arc<[u8]>, turn_model: Arc<[u8]>, threads: usize) -> Result<Arc<Self>> {
        let threads = threads.max(1);
        let (tx, rx) = mpsc::channel::<Job>(threads * 64);
        let rx = Arc::new(Mutex::new(rx));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

        for i in 0..threads {
            let rx = Arc::clone(&rx);
            let vad_model = Arc::clone(&vad_model);
            let turn_model = Arc::clone(&turn_model);
            let ready_tx = ready_tx.clone();
            thread::Builder::new()
                .name(format!("inference-{i}"))
                .spawn(move || {
                    let models = SileroModel::from_memory(&vad_model)
                        .and_then(|v| SmartTurnModel::from_memory(&turn_model).map(|t| (v, t)));
                    let (mut vad, mut turn) = match models {
                        Ok(m) => {
                            let _ = ready_tx.send(Ok(()));
                            m
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    drop(ready_tx);
                    loop {
                        let job = {
                            let mut guard = match rx.lock() {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                            guard.blocking_recv()
                        };
                        match job {
                            None => return,
                            Some(Job::Vad(mut job, reply)) => {
                                let result = vad.run(&mut job.state, &job.samples, job.sample_rate);
                                let _ = reply.send(VadOutcome {
                                    result,
                                    samples: job.samples,
                                    state: job.state,
                                });
                            }
                            Some(Job::Turn(job, reply)) => {
                                let result =
                                    turn.run(&job.samples, job.sample_rate, job.speech_start_ms);
                                let _ = reply.send(TurnOutcome {
                                    result,
                                    samples: job.samples,
                                });
                            }
                        }
                    }
                })?;
        }
        drop(ready_tx);

        for _ in 0..threads {
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error!(error = %e, "inference worker failed to load models");
                    return Err(e);
                }
                Err(_) => return Err(anyhow!("inference worker exited during startup")),
            }
        }
        info!(threads, "inference pool ready");
        Ok(Arc::new(Self { tx }))
    }

    pub async fn vad(&self, job: VadJob) -> Result<VadOutcome> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Vad(job, reply))
            .await
            .map_err(|_| anyhow!("inference pool closed"))?;
        rx.await
            .map_err(|_| anyhow!("inference worker dropped VAD job"))
    }

    pub async fn turn(&self, job: TurnJob) -> Result<TurnOutcome> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Turn(job, reply))
            .await
            .map_err(|_| anyhow!("inference pool closed"))?;
        rx.await
            .map_err(|_| anyhow!("inference worker dropped turn job"))
    }
}
