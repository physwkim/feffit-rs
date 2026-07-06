//! Background jobs: the single owner of the app's long-running batch work.
//!
//! Heavy operations (Multiple AUTOBK, batch μ(E), Feffit runs) used to run on
//! the UI thread, freezing every window until they finished. This module gives
//! the app one job slot: [`JobRunner::start`] moves the computation onto a
//! worker thread with a [`JobHandle`] for progress ticks, and the worker's
//! final answer is an [`Apply`] closure that the *main* thread executes via
//! [`JobRunner::poll`] — so shared state (the session, the Feffit editor, the
//! status log) is only ever mutated on the UI thread, by construction.
//!
//! One slot, one owner: `start` refuses while a job is running (no two jobs can
//! race), and every exit — normal completion or a worker panic — funnels
//! through `poll`, which clears the slot and reports the outcome as a
//! [`Polled`]. A stuck "busy" state cannot outlive its worker.

use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

/// What a finished worker hands back: a closure the main thread runs against
/// the app to apply the results (mutate the session, write files, log).
pub type Apply<T> = Box<dyn FnOnce(&mut T) + Send>;

enum JobMsg<T> {
    /// `done` count for the progress display (out-of-order sends are fine; the
    /// receiver keeps the max).
    Progress(usize),
    Done(Apply<T>),
}

/// The worker's side of the channel: progress ticks. Sync, so a rayon fan-out
/// inside the worker can tick from many threads.
pub struct JobHandle<T> {
    tx: Sender<JobMsg<T>>,
}

impl<T> JobHandle<T> {
    /// Report that `done` items are finished (shown as `done/total`).
    pub fn progress(&self, done: usize) {
        let _ = self.tx.send(JobMsg::Progress(done));
    }
}

struct Running<T> {
    label: String,
    done: usize,
    total: usize,
    rx: Receiver<JobMsg<T>>,
}

/// A polled job outcome — see [`JobRunner::poll`].
pub enum Polled<T> {
    /// The job finished; run this on the owner to apply its results.
    Apply(Apply<T>),
    /// The worker died without an answer (panicked); the label identifies it.
    Failed(String),
}

/// The app's single background-job slot.
pub struct JobRunner<T> {
    job: Option<Running<T>>,
}

impl<T> Default for JobRunner<T> {
    fn default() -> Self {
        Self { job: None }
    }
}

impl<T: 'static> JobRunner<T> {
    /// Whether a job is running (new jobs are refused while one is).
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// The running job's `(label, done, total)` for the progress display.
    pub fn status(&self) -> Option<(&str, usize, usize)> {
        self.job
            .as_ref()
            .map(|j| (j.label.as_str(), j.done, j.total))
    }

    /// Spawn `work` on a worker thread. Returns `false` (and does nothing) if a
    /// job is already running — the caller reports the refusal to the user.
    /// `work` computes off-thread, ticking `handle.progress`, and returns the
    /// [`Apply`] closure that [`poll`](Self::poll) hands back to the main
    /// thread when done.
    pub fn start<F>(&mut self, label: &str, total: usize, work: F) -> bool
    where
        F: FnOnce(&JobHandle<T>) -> Apply<T> + Send + 'static,
    {
        if self.job.is_some() {
            return false;
        }
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let handle = JobHandle { tx };
            let apply = work(&handle);
            let _ = handle.tx.send(JobMsg::Done(apply));
        });
        self.job = Some(Running {
            label: label.to_owned(),
            done: 0,
            total,
            rx,
        });
        true
    }

    /// Drain the worker's messages: progress ticks update the status; a
    /// completion (or a dead worker) clears the slot and returns the outcome.
    /// Call once per frame; run the returned [`Polled::Apply`] on the owner.
    pub fn poll(&mut self) -> Option<Polled<T>> {
        enum Ev<T> {
            Apply(Apply<T>),
            Died,
        }
        let ev = {
            let job = self.job.as_mut()?;
            loop {
                match job.rx.try_recv() {
                    Ok(JobMsg::Progress(d)) => job.done = job.done.max(d),
                    Ok(JobMsg::Done(apply)) => break Ev::Apply(apply),
                    Err(TryRecvError::Empty) => return None,
                    Err(TryRecvError::Disconnected) => break Ev::Died,
                }
            }
        };
        let job = self.job.take().expect("a job was just polled");
        Some(match ev {
            Ev::Apply(apply) => Polled::Apply(apply),
            Ev::Died => Polled::Failed(job.label),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Poll until the job yields an outcome (the worker runs on its own
    /// thread, so completion is not immediate).
    fn poll_until_done(runner: &mut JobRunner<Vec<String>>) -> Polled<Vec<String>> {
        for _ in 0..1000 {
            if let Some(polled) = runner.poll() {
                return polled;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("job never finished");
    }

    #[test]
    fn job_computes_off_thread_and_applies_on_the_owner() {
        let mut runner: JobRunner<Vec<String>> = JobRunner::default();
        assert!(!runner.busy());
        let started = runner.start("count", 2, |job| {
            job.progress(1);
            job.progress(2);
            let result = "computed".to_owned();
            Box::new(move |owner: &mut Vec<String>| owner.push(result))
        });
        assert!(started);
        assert!(runner.busy());
        assert_eq!(runner.status().map(|(l, _, t)| (l.to_owned(), t)), {
            Some(("count".to_owned(), 2))
        });

        let mut owner = Vec::new();
        match poll_until_done(&mut runner) {
            Polled::Apply(apply) => apply(&mut owner),
            Polled::Failed(l) => panic!("job failed: {l}"),
        }
        assert_eq!(owner, ["computed"]);
        assert!(!runner.busy(), "the slot clears on completion");
    }

    #[test]
    fn second_start_is_refused_while_busy() {
        let mut runner: JobRunner<Vec<String>> = JobRunner::default();
        // Hold the first job open until we've tried the second.
        let (block_tx, block_rx) = channel::<()>();
        assert!(runner.start("first", 1, move |_| {
            let _ = block_rx.recv();
            Box::new(|_| {})
        }));
        assert!(
            !runner.start("second", 1, |_| Box::new(|_| {})),
            "one slot: a second job must be refused"
        );
        drop(block_tx);
        match poll_until_done(&mut runner) {
            Polled::Apply(_) => {}
            Polled::Failed(l) => panic!("job failed: {l}"),
        }
        // The slot is free again.
        assert!(runner.start("third", 1, |_| Box::new(|_| {})));
        poll_until_done(&mut runner);
    }

    #[test]
    fn worker_panic_clears_the_slot_and_reports_failure() {
        let mut runner: JobRunner<Vec<String>> = JobRunner::default();
        assert!(runner.start("doomed", 1, |_| panic!("worker crashed")));
        match poll_until_done(&mut runner) {
            Polled::Failed(label) => assert_eq!(label, "doomed"),
            Polled::Apply(_) => panic!("a panicked worker cannot produce a result"),
        }
        assert!(!runner.busy(), "a dead worker must not leave the slot busy");
    }
}
