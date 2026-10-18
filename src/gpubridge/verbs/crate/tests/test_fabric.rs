//! The cross-process test fabric (feature `test-fabric`), here with both ends in one
//! process: each endpoint has its own node, socket and threads, as in separate processes.

#![cfg(feature = "test-fabric")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gpubridge_verbs::{Config, Endpoint, Provider, UdsProvider, WaitError};

const LONG: Duration = Duration::from_secs(20);

/// A short directory: socket paths are limited to about 100 bytes.
fn fabric_dir() -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = PathBuf::from(format!(
        "/tmp/gbvf-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn small() -> Config {
    Config {
        slot_bytes: 512,
        credits: 4,
        ctrl_credits: 2,
        send_slots: 4,
        ..Config::default()
    }
}

fn pair(dir: &Path) -> (Endpoint, Endpoint) {
    let make = |i| {
        let p: Arc<dyn Provider> = Arc::new(UdsProvider::open(dir).unwrap());
        Endpoint::new(p, 2, i, small()).unwrap()
    };
    let (a, b) = (make(0), make(1));
    a.connect(1, &b.local_info()).unwrap();
    b.connect(0, &a.local_info()).unwrap();
    (a, b)
}

#[test]
fn messages_cross_between_nodes_and_telemetry_says_mock() {
    let dir = fabric_dir();
    let (a, b) = pair(&dir);
    let data: Vec<u8> = (0..100_000u32).map(|i| (i % 253) as u8).collect();
    let mut out = vec![0u8; data.len()];
    let mut back = vec![0u8; 7];
    // SAFETY: the buffers outlive the waits below.
    let ids_b = unsafe {
        [
            b.recv(0, out.as_mut_ptr() as usize, data.len() as u64, 1)
                .unwrap(),
            b.send(0, data.as_ptr() as usize, 7, 2).unwrap(),
        ]
    };
    // SAFETY: as above.
    let ids_a = unsafe {
        [
            a.send(1, data.as_ptr() as usize, data.len() as u64, 1)
                .unwrap(),
            a.recv(1, back.as_mut_ptr() as usize, 7, 2).unwrap(),
        ]
    };
    a.wait(&ids_a, LONG).unwrap();
    b.wait(&ids_b, LONG).unwrap();
    assert!(out == data);
    assert_eq!(back, data[..7]);
    let t = a.telemetry();
    assert!(
        t.contains("MOCK: cross-process test fabric") && t.contains("\"mock\":true"),
        "{t}"
    );
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    let left: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert!(
        left.is_empty(),
        "nodes remove their sockets and LIDs: {left:?}"
    );
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn a_peer_that_failed_or_left_fails_sends_to_it() {
    let dir = fabric_dir();
    let (a, b) = pair(&dir);
    // b fails without a goodbye: its queue pair is in the error state.
    b.abort("crashed");
    let data = [1u8; 8];
    // SAFETY: data outlives the wait.
    let s = unsafe { a.send(1, data.as_ptr() as usize, 8, 1) }.unwrap();
    match a.wait(&[s], LONG) {
        Err(WaitError::Failed(e)) => assert!(e.contains("RetryExcErr"), "{e}"),
        other => panic!("{other:?}"),
    }
    a.close(LONG).ok();
    b.close(LONG).ok();
    // A peer that closed said goodbye: a send to it is refused at once.
    let (a, b) = pair(&dir);
    b.close(LONG).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    // SAFETY: refused, never used.
    let e = unsafe { a.send(1, data.as_ptr() as usize, 8, 1) }.unwrap_err();
    assert!(e.to_string().contains("closed its end"), "{e}");
    a.close(LONG).ok();
    let _ = std::fs::remove_dir_all(&dir);
}
