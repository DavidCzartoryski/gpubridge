//! Registered memory on the mock fabric with a mock GPU (docs/VERBS_TRANSPORT.md, "GPU
//! memory"): which direct path is chosen and why the others weren't, dma-buf descriptors
//! and registrations released in the right order, stale registrations dropped when an
//! address is reused, and the rendezvous moving device and host memory in place.
//! SOFTWARE-VERIFIED only: no GPU, no RDMA device.

mod common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

use common::{eventually, ib_fabric, RawMember};
use gpubridge_verbs::protocol::Kind;
use gpubridge_verbs::{
    Access, Allocation, Call, Config, DmabufCloser, Endpoint, Error, Fault, GpuMemory, Method,
    MockDevice, MockFabric, MockGpu, MockPort, NodeId, PeerModules, Progress, Result, Vendor,
    WaitError, WcStatus, CACHED_ACCESS, DEVICE_ROUND,
};

const LONG: Duration = Duration::from_secs(20);
const RNDV: u64 = 4096;

#[derive(Debug)]
struct PreparingGpu {
    gpu: MockGpu,
    gates: Option<(Barrier, Barrier)>,
    calls: AtomicUsize,
    reject_first: bool,
}

impl DmabufCloser for PreparingGpu {
    fn close_dmabuf(&self, fd: i32) -> Result<()> {
        self.gpu.close_dmabuf(fd)
    }
}

impl GpuMemory for PreparingGpu {
    fn vendor(&self) -> Vendor {
        self.gpu.vendor()
    }
    fn describe(&self) -> String {
        self.gpu.describe()
    }
    fn allocation(&self, device: i32, ptr: u64) -> Result<Allocation> {
        self.gpu.allocation(device, ptr)
    }
    fn dmabuf_support(&self, device: i32) -> std::result::Result<(), String> {
        self.gpu.dmabuf_support(device)
    }
    fn export_dmabuf(&self, a: &Allocation, start: u64, len: u64) -> Result<(i32, u64)> {
        self.gpu.export_dmabuf(a, start, len)
    }
    fn peer_memory_support(&self, device: i32) -> std::result::Result<(), String> {
        self.gpu.peer_memory_support(device)
    }
    fn prepare(&self, a: &Allocation, start: u64) -> Result<()> {
        let first = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
        if first {
            if let Some((entered, release)) = &self.gates {
                entered.wait();
                release.wait();
            }
            if self.reject_first {
                return Err(Error::Gpu("injected SYNC_MEMOPS failure".into()));
            }
        }
        self.gpu.prepare(a, start)
    }
}

#[test]
fn a_registration_is_not_published_before_its_gpu_preparation_finishes() {
    let gpu = MockGpu::new();
    let (_fabric, a, _b) = pair(RNDV, &gpu, false);
    let va = gpu.alloc(0, 1 << 20);
    let driver = Arc::new(PreparingGpu {
        gpu: gpu.clone(),
        gates: Some((Barrier::new(2), Barrier::new(2))),
        calls: AtomicUsize::new(0),
        reject_first: false,
    });
    a.set_gpu(driver.clone(), modules("preparation-race", false));
    std::thread::scope(|scope| {
        let first = scope.spawn(|| a.register_device(0, va, 4096));
        let (entered, release) = driver.gates.as_ref().unwrap();
        entered.wait();
        let (tx, rx) = mpsc::channel();
        let endpoint = &a;
        let second = scope.spawn(move || {
            let result = endpoint.register_device(0, va, 4096);
            tx.send(()).unwrap();
            result
        });
        let early = rx.recv_timeout(Duration::from_millis(50));
        // Release before asserting, so a failing regression cannot strand a thread.
        release.wait();
        let r1 = first.join().unwrap().unwrap();
        let r2 = second.join().unwrap().unwrap();
        assert!(
            early.is_err(),
            "a cache hit escaped before preparation finished"
        );
        assert!(r1.region().same(r2.region()));
    });
    assert_eq!(driver.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_gpu_preparation_is_not_cached_and_closes_its_export_once() {
    let gpu = MockGpu::new();
    let (_fabric, a, _b) = pair(RNDV, &gpu, false);
    let va = gpu.alloc(0, 1 << 20);
    let driver = Arc::new(PreparingGpu {
        gpu: gpu.clone(),
        gates: None,
        calls: AtomicUsize::new(0),
        reject_first: true,
    });
    a.set_gpu(driver.clone(), modules("preparation-failure", false));
    assert!(a
        .register_device(0, va, 4096)
        .unwrap_err()
        .contains("SYNC_MEMOPS failure"));
    assert_eq!(gpu.open_fds(), 0);
    let r = a.register_device(0, va, 4096).unwrap();
    assert_eq!(driver.calls.load(Ordering::SeqCst), 2);
    assert_eq!(gpu.exported(), 2);
    assert_eq!(gpu.open_fds(), 1);
    drop(r);
}

fn config(rndv: u64) -> Config {
    Config {
        slot_bytes: 1024,
        credits: 4,
        ctrl_credits: 2,
        send_slots: 4,
        rndv_bytes: rndv,
        ..Config::default()
    }
}

fn endpoints(fabric: &Arc<MockFabric>, nodes: &[NodeId], config: &Config) -> Vec<Endpoint> {
    let n = nodes.len();
    let eps: Vec<Endpoint> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| Endpoint::new(fabric.provider(node), n, i, config.clone()).unwrap())
        .collect();
    let infos: Vec<String> = eps.iter().map(Endpoint::local_info).collect();
    for (i, ep) in eps.iter().enumerate() {
        for (j, info) in infos.iter().enumerate() {
            if i != j {
                ep.connect(j, info).unwrap();
            }
        }
    }
    eps
}

/// A fixture root with (or without) a loaded `nvidia_peermem`.
fn modules(name: &str, loaded: bool) -> PeerModules {
    // One directory per call: tests run in parallel threads of one process.
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "gbv-direct-{name}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("proc")).unwrap();
    std::fs::write(dir.join("proc/modules"), "mlx5_ib 1 0 - Live 0x0\n").unwrap();
    if loaded {
        std::fs::create_dir_all(dir.join("sys/module/nvidia_peermem")).unwrap();
    }
    PeerModules::at(dir)
}

fn pair(rndv: u64, gpu: &MockGpu, peer_module: bool) -> (Arc<MockFabric>, Endpoint, Endpoint) {
    pair_with_progress(rndv, gpu, peer_module, Progress::Economy)
}

fn pair_with_progress(
    rndv: u64,
    gpu: &MockGpu,
    peer_module: bool,
    progress: Progress,
) -> (Arc<MockFabric>, Endpoint, Endpoint) {
    let (fabric, a, b) = ib_fabric();
    let mut c = config(rndv);
    c.progress = progress;
    let mut eps = endpoints(&fabric, &[a, b], &c);
    let eb = eps.pop().unwrap();
    let ea = eps.pop().unwrap();
    for (i, ep) in [&ea, &eb].into_iter().enumerate() {
        ep.set_gpu(
            Arc::new(gpu.clone()),
            modules(&format!("pair{i}-{peer_module}"), peer_module),
        );
    }
    (fabric, ea, eb)
}

fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed))
        .collect()
}

#[test]
fn dmabuf_is_chosen_first_then_peer_memory_and_otherwise_every_reason_is_given() {
    let gpu = MockGpu::new();
    let (_fabric, a, _b) = pair(RNDV, &gpu, true);
    let va = gpu.alloc(0, 8 << 20);
    let r = a.register_device(0, va + 4096, 1 << 20).unwrap();
    assert_eq!(r.method(), Method::DmaBuf);
    assert!(!r.cpu() && r.region().is_dmabuf());
    // Rounded out to 2 MiB inside the allocation: the whole first 2 MiB.
    assert_eq!((r.addr() as u64, r.len() as u64), (va, DEVICE_ROUND));
    let caps = a.gpu_capabilities(0);
    assert!(caps.contains("\"dmabuf\":\"ok\""), "{caps}");
    assert!(caps.contains("ok (nvidia_peermem)"), "{caps}");

    // No dma-buf on the device: peer memory, since the module is loaded.
    let gpu2 = MockGpu::new();
    gpu2.set_dmabuf_unsupported(Some("CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0"));
    let (_f2, a2, _b2) = pair(RNDV, &gpu2, true);
    let va2 = gpu2.alloc(0, 1 << 20);
    let r2 = a2.register_device(0, va2, 4096).unwrap();
    assert_eq!(r2.method(), Method::PeerMemory);
    assert!(!r2.region().is_dmabuf());

    // Neither: no dma-buf and no module. Both reasons, nothing registered.
    let gpu3 = MockGpu::new();
    gpu3.set_dmabuf_unsupported(Some("CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0"));
    let (_f3, a3, _b3) = pair(RNDV, &gpu3, false);
    let va3 = gpu3.alloc(0, 1 << 20);
    let why = a3.register_device(0, va3, 4096).unwrap_err();
    assert!(
        why.contains("dma-buf: CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0"),
        "{why}"
    );
    assert!(why.contains("peer memory: neither nvidia_peermem"), "{why}");
    let t = a3.telemetry();
    assert!(t.contains("\"registration_failures\":1"), "{t}");
    assert!(
        t.contains("\"last_registration_failure\":\"dma-buf:"),
        "{t}"
    );

    // A failed export falls through to peer memory; a refused one by the driver too.
    gpu.fail_exports(Some(
        "cuMemGetHandleForAddressRange failed: CUDA_ERROR_NOT_SUPPORTED",
    ));
    let other = gpu.alloc(0, 1 << 20);
    let r4 = a.register_device(0, other, 4096).unwrap();
    assert_eq!(r4.method(), Method::PeerMemory);
    gpu.fail_exports(None);

    // Not device memory, another device's memory, past the allocation: refused, why.
    assert!(a
        .register_device(0, 0x1000, 64)
        .unwrap_err()
        .contains("not mock device memory"));
    assert!(a.register_device(1, va, 64).unwrap_err().contains("not 1"));
    assert!(a
        .register_device(0, va + (8 << 20) - 10, 64)
        .unwrap_err()
        .contains("run past"));
}

#[test]
fn settings_can_disable_each_path_and_no_driver_is_a_reason_too() {
    let (fabric, na, nb) = ib_fabric();
    let mut c = config(RNDV);
    c.direct.dmabuf = false;
    let eps = endpoints(&fabric, &[na, nb], &c);
    let why = eps[0].register_device(0, 0x7e00_0000_0000, 64).unwrap_err();
    assert!(why.contains("no GPU driver loaded"), "{why}");
    let gpu = MockGpu::new();
    eps[0].set_gpu(Arc::new(gpu.clone()), modules("settings", false));
    let va = gpu.alloc(0, 1 << 20);
    let why = eps[0].register_device(0, va, 64).unwrap_err();
    assert!(why.contains("dma-buf: disabled by the settings"), "{why}");
    assert!(eps[0]
        .gpu_capabilities(0)
        .contains("disabled by the settings"));
}

#[test]
fn a_dmabuf_descriptor_is_closed_once_after_its_region_is_deregistered() {
    let gpu = MockGpu::new();
    let (fabric, a, b) = pair(RNDV, &gpu, false);
    let va = gpu.alloc(0, 4 << 20);
    let r = a.register_device(0, va, 1 << 20).unwrap();
    assert_eq!(gpu.exported(), 1);
    assert_eq!(gpu.open_fds(), 1);
    // SYNC_MEMOPS once per new registration; a lookup inside it hits the cache.
    assert_eq!(gpu.sync_memops(), vec![va]);
    let again = a.register_device(0, va + 100, 1000).unwrap();
    assert!(again.region().same(r.region()));
    assert_eq!(gpu.exported(), 1);
    assert_eq!(gpu.sync_memops().len(), 1);
    drop(again);
    drop(r);
    // Still cached (idle): registered, descriptor open.
    assert_eq!(gpu.open_fds(), 1);
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    // Closing the endpoint deregistered the region and then closed the descriptor.
    assert_eq!(gpu.open_fds(), 0);
    assert!(gpu.violations().is_empty(), "{:?}", gpu.violations());
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn a_failed_dmabuf_registration_closes_the_descriptor_at_once() {
    let gpu = MockGpu::new();
    let (fabric, na, nb) = ib_fabric();
    let eps = endpoints(&fabric, &[na, nb], &config(RNDV));
    eps[0].set_gpu(Arc::new(gpu.clone()), modules("failreg", false));
    let va = gpu.alloc(0, 1 << 20);
    fabric.inject(Fault::FailCall {
        node: na,
        call: Call::RegMr,
        nth: 0,
        errno: 12,
    });
    let why = eps[0].register_device(0, va, 4096).unwrap_err();
    assert!(why.contains("ibv_reg_dmabuf_mr"), "{why}");
    assert_eq!(gpu.exported(), 1);
    assert_eq!(
        gpu.open_fds(),
        0,
        "the descriptor of a failed registration is closed"
    );
    assert!(gpu.violations().is_empty(), "{:?}", gpu.violations());
}

#[test]
fn a_reallocated_address_gets_a_new_registration_and_the_stale_one_goes() {
    let gpu = MockGpu::new();
    let (fabric, a, b) = pair(RNDV, &gpu, false);
    let va = gpu.alloc(0, 2 << 20);
    let first = a.register_device(0, va, 4096).unwrap();
    // While posted work (or a caller) holds the old registration, the address can't be
    // registered for the new allocation: it was freed too early.
    let same = gpu.realloc_same_address(va, 2 << 20);
    assert_eq!(same, va);
    let why = a.register_device(0, va, 4096).unwrap_err();
    assert!(why.contains("in use"), "{why}");
    drop(first);
    // Idle now: dropped as stale, a new export and registration for the new buffer ID.
    let second = a.register_device(0, va, 4096).unwrap();
    assert_eq!(gpu.exported(), 2);
    assert_eq!(
        gpu.open_fds(),
        1,
        "the stale registration's descriptor was closed"
    );
    let t = a.telemetry();
    assert!(t.contains("\"stale\":1"), "{t}");
    // The new registration carries data into the new allocation.
    let data = pattern(RNDV as usize, 3);
    let got = b
        .send_registered(0, &host(&b, &data, 1), 0, data.len() as u64, 1)
        .unwrap();
    let put = a
        .recv_registered(1, &second, 0, data.len() as u64, 1)
        .unwrap();
    b.wait(&[got], LONG).unwrap();
    a.wait(&[put], LONG).unwrap();
    assert_eq!(gpu.read(va, data.len()), data);
    drop(second);
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    assert_eq!(gpu.open_fds(), 0);
    assert!(gpu.violations().is_empty(), "{:?}", gpu.violations());
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn a_receive_window_never_registers_a_reused_allocation_through_an_old_handle() {
    for progress in [Progress::Latency, Progress::Balanced, Progress::Economy] {
        for peer_memory in [false, true] {
            let gpu = MockGpu::new();
            if peer_memory {
                gpu.set_dmabuf_unsupported(Some("test peer memory"));
            }
            let (fabric, a, b) = pair_with_progress(RNDV, &gpu, peer_memory, progress);
            let va = gpu.alloc(0, 2 << 20);
            let reg = b.register_device(0, va, 4096).unwrap();
            let exports = gpu.exported();
            gpu.realloc_same_address(va, 2 << 20);
            gpu.write(va, &vec![0xA5; 4096]);
            let error = b
                .recv_registered(0, &reg, va as usize - reg.addr(), RNDV, 7)
                .unwrap_err();
            assert!(error.to_string().contains("allocation changed"), "{error}");
            assert_eq!(
                gpu.exported(),
                exports,
                "a new allocation was exported through an old handle"
            );
            assert_eq!(gpu.read(va, 4096), vec![0xA5; 4096]);
            drop(reg);
            a.close(LONG).unwrap();
            b.close(LONG).unwrap();
            assert_eq!(gpu.open_fds(), 0);
            assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
        }
    }
}

/// `data` registered as host memory of `ep` (owner `owner`), leaked for the test's
/// lifetime so it outlives the registration.
fn host(ep: &Endpoint, data: &[u8], owner: u64) -> gpubridge_verbs::Registered {
    let buf: &'static mut [u8] = Box::leak(data.to_vec().into_boxed_slice());
    // SAFETY: leaked, so valid for the rest of the process.
    unsafe { ep.register_host(buf.as_mut_ptr() as usize, buf.len(), owner) }.unwrap()
}

#[test]
fn device_memory_is_written_by_rdma_write_and_read_in_place_by_the_device() {
    let gpu = MockGpu::new();
    let (fabric, a, b) = pair(RNDV, &gpu, false);
    let src = gpu.alloc(0, 4 << 20);
    let dst = gpu.alloc(0, 4 << 20);
    let sizes = [1usize, 100, 1000, RNDV as usize - 1, RNDV as usize, 3 << 20];
    let mut offset = 0u64;
    for (i, &n) in sizes.iter().enumerate() {
        let data = pattern(n, i as u8);
        gpu.write(src + offset, &data);
        let s = a.register_device(0, src + offset, n as u64).unwrap();
        let soff = (src + offset) as usize - s.addr();
        let id_s = a.send_registered(1, &s, soff, n as u64, i as u64).unwrap();
        let id_r = if n as u64 >= RNDV {
            let r = b.register_device(0, dst + offset, n as u64).unwrap();
            let roff = (dst + offset) as usize - r.addr();
            b.recv_registered(0, &r, roff, n as u64, i as u64).unwrap()
        } else {
            // Below the rendezvous size a message comes in chunks: device memory can't
            // take it, so the caller receives it into host memory (the transport then
            // copies it to the GPU on its stream).
            let r = b.register_device(0, dst + offset, n as u64).unwrap();
            let roff = (dst + offset) as usize - r.addr();
            let e = b
                .recv_registered(0, &r, roff, n as u64, i as u64)
                .unwrap_err();
            assert!(e.to_string().contains("rndv_bytes"), "{e}");
            let mut out = vec![0u8; n];
            // SAFETY: waited for below, before `out` goes.
            let id = unsafe { b.recv(0, out.as_mut_ptr() as usize, n as u64, i as u64) }.unwrap();
            b.wait(&[id], LONG).unwrap();
            assert_eq!(out, data, "size {n}");
            gpu.write(dst + offset, &out);
            id
        };
        a.wait(&[id_s], LONG).unwrap();
        if n as u64 >= RNDV {
            b.wait(&[id_r], LONG).unwrap();
        }
        assert_eq!(gpu.read(dst + offset, n), data, "size {n}");
        offset += n as u64 + 64;
    }
    let (ca, cb) = (a.counters(), b.counters());
    let big: u64 = sizes
        .iter()
        .filter(|&&n| n as u64 >= RNDV)
        .map(|&n| n as u64)
        .sum();
    let all: u64 = sizes.iter().map(|&n| n as u64).sum();
    assert_eq!(ca.device_bytes, all, "every send was read in place");
    assert_eq!(
        cb.device_bytes, big,
        "every rendezvous receive was written in place"
    );
    assert_eq!(cb.grants_sent, 2);
    assert_eq!(ca.grants_received, 2);
    assert_eq!((ca.fins_sent, cb.fins_received), (2, 2));
    assert_eq!(ca.writes_posted, 2, "one write each (below WRITE_CHUNK)");
    assert_eq!(ca.copied_bytes, 0);
    let t = a.telemetry();
    for key in ["rkey", "lkey", "\"key\""] {
        assert!(!t.contains(key), "{key} in {t}");
    }
    assert!(t.contains("RDMA WRITE (rendezvous)"), "{t}");
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    assert!(gpu.violations().is_empty(), "{:?}", gpu.violations());
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn every_pairing_of_plain_host_registered_and_device_memory_moves_the_same_bytes() {
    for p in [Progress::Latency, Progress::Balanced, Progress::Economy] {
        check_memory_pairings(p);
    }
}

fn check_memory_pairings(progress: Progress) {
    let gpu = MockGpu::new();
    let (fabric, a, b) = pair_with_progress(RNDV, &gpu, true, progress);
    let n = 3 * 1024 + RNDV as usize; // several slots' worth
    let kinds = ["plain", "host", "device"];
    let mut tag = 0;
    for send_kind in kinds {
        for recv_kind in kinds {
            tag += 1;
            let data = pattern(n, tag as u8);
            let mut plain_out = vec![0u8; n];
            let dev_out = gpu.alloc(0, n as u64);
            let host_out: &'static mut [u8] = Box::leak(vec![0u8; n].into_boxed_slice());
            let host_out_ptr = host_out.as_mut_ptr();
            let s = match send_kind {
                "plain" => {
                    // SAFETY: waited for below while `data` lives.
                    unsafe { a.send(1, data.as_ptr() as usize, n as u64, tag) }.unwrap()
                }
                "host" => a
                    .send_registered(1, &host(&a, &data, 1000 + tag), 0, n as u64, tag)
                    .unwrap(),
                _ => {
                    let va = gpu.alloc(0, n as u64);
                    gpu.write(va, &data);
                    let r = a.register_device(0, va, n as u64).unwrap();
                    a.send_registered(1, &r, (va as usize) - r.addr(), n as u64, tag)
                        .unwrap()
                }
            };
            let r = match recv_kind {
                // SAFETY: waited for below while `plain_out` lives.
                "plain" => {
                    unsafe { b.recv(0, plain_out.as_mut_ptr() as usize, n as u64, tag) }.unwrap()
                }
                "host" => {
                    // SAFETY: leaked above, valid for the process.
                    let reg =
                        unsafe { b.register_host(host_out_ptr as usize, n, 2000 + tag) }.unwrap();
                    b.recv_registered(0, &reg, 0, n as u64, tag).unwrap()
                }
                _ => {
                    let reg = b.register_device(0, dev_out, n as u64).unwrap();
                    b.recv_registered(0, &reg, (dev_out as usize) - reg.addr(), n as u64, tag)
                        .unwrap()
                }
            };
            a.wait(&[s], LONG).unwrap();
            b.wait(&[r], LONG).unwrap();
            let got = match recv_kind {
                "plain" => plain_out.clone(),
                // SAFETY: the receive is done; nothing writes it any more.
                "host" => unsafe { std::slice::from_raw_parts(host_out_ptr, n) }.to_vec(),
                _ => gpu.read(dev_out, n),
            };
            assert!(got == data, "{send_kind} -> {recv_kind}");
        }
    }
    let cb = b.counters();
    // Plain receives are cleared "in chunks"; registered ones get write grants.
    assert_eq!(cb.go_sent, 3);
    assert_eq!(cb.grants_sent, 6);
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    assert!(gpu.violations().is_empty(), "{:?}", gpu.violations());
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn a_message_waiting_for_its_grant_holds_up_only_its_own_tag() {
    let gpu = MockGpu::new();
    let (fabric, a, b) = pair(RNDV, &gpu, false);
    let big = pattern(2 * RNDV as usize, 1);
    let big2 = pattern(2 * RNDV as usize, 2);
    let small = pattern(64, 3);
    // SAFETY (all three): waited for below while the data lives.
    let s1 = unsafe { a.send(1, big.as_ptr() as usize, big.len() as u64, 7) }.unwrap();
    let s2 = unsafe { a.send(1, big2.as_ptr() as usize, big2.len() as u64, 7) }.unwrap();
    let s3 = unsafe { a.send(1, small.as_ptr() as usize, small.len() as u64, 8) }.unwrap();
    // Tag 8 arrives although tag 7's messages wait for their receives.
    let mut out_small = vec![0u8; 64];
    // SAFETY: waited for below.
    let r3 = unsafe { b.recv(0, out_small.as_mut_ptr() as usize, 64, 8) }.unwrap();
    b.wait(&[r3], LONG).unwrap();
    a.wait(&[s3], LONG).unwrap();
    assert_eq!(out_small, small);
    assert_eq!(a.counters().fins_sent + a.counters().grants_received, 0);
    // Then tag 7, in order: the first message to the first receive.
    let dev = gpu.alloc(0, 4 * RNDV);
    let reg = b.register_device(0, dev, 4 * RNDV).unwrap();
    let off = dev as usize - reg.addr();
    let r1 = b
        .recv_registered(0, &reg, off, big.len() as u64, 7)
        .unwrap();
    b.wait(&[r1], LONG).unwrap();
    let r2 = b
        .recv_registered(0, &reg, off + big.len(), big2.len() as u64, 7)
        .unwrap();
    b.wait(&[r2], LONG).unwrap();
    a.wait(&[s1, s2], LONG).unwrap();
    assert_eq!(gpu.read(dev, big.len()), big);
    assert_eq!(gpu.read(dev + big.len() as u64, big2.len()), big2);
    // Plain memory written by RDMA WRITE goes through slots, one write per slot.
    assert_eq!(a.counters().writes_posted, 2 * (2 * RNDV).div_ceil(1024));
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn a_grant_of_the_wrong_size_and_receives_device_memory_cant_take_are_refused() {
    let gpu = MockGpu::new();
    let (_fabric, a, b) = pair(RNDV, &gpu, false);
    let data = pattern(2 * RNDV as usize, 1);
    // SAFETY: the endpoint fails (and is closed) before `data` goes.
    let s = unsafe { a.send(1, data.as_ptr() as usize, data.len() as u64, 5) }.unwrap();
    let dev = gpu.alloc(0, 4 * RNDV);
    let reg = b.register_device(0, dev, 4 * RNDV).unwrap();
    let r = b
        .recv_registered(0, &reg, dev as usize - reg.addr(), 3 * RNDV, 5)
        .unwrap();
    let e = a.wait(&[s], LONG).unwrap_err();
    assert!(
        format!("{e:?}").contains("posted 12288 bytes for tag 0x5"),
        "{e:?}"
    );
    assert!(b.wait(&[r], Duration::from_secs(2)).is_err());
    a.close(LONG).ok();
    b.close(LONG).ok();

    // Without rendezvous, device memory can't be received at all.
    let gpu2 = MockGpu::new();
    let (_f2, a2, _b2) = pair(0, &gpu2, false);
    let dev2 = gpu2.alloc(0, 1 << 20);
    let reg2 = a2.register_device(0, dev2, 1 << 20).unwrap();
    let e = a2.recv_registered(1, &reg2, 0, 1 << 20, 1).unwrap_err();
    assert!(e.to_string().contains("rndv_bytes (0; 0 is off)"), "{e}");
    let t = a2.telemetry();
    assert!(t.contains("\"operations\":\"RC SEND/RECV\""), "{t}");
}

#[test]
fn aborting_with_a_write_granted_keeps_the_registration_until_close() {
    for p in [Progress::Latency, Progress::Balanced, Progress::Economy] {
        check_aborting_with_write_granted(p);
    }
}

fn check_aborting_with_write_granted(progress: Progress) {
    let gpu = MockGpu::new();
    let (fabric, a, b) = pair_with_progress(RNDV, &gpu, false, progress);
    let dev = gpu.alloc(0, 1 << 20);
    let reg = b.register_device(0, dev, 1 << 20).unwrap();
    let r = b
        .recv_registered(0, &reg, dev as usize - reg.addr(), 2 * RNDV, 9)
        .unwrap();
    drop(reg);
    // The grant went out (its completion polled), then the receiver gives up.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(b.counters().grants_sent, 1);
    b.abort("deadline");
    assert!(b.wait(&[r], LONG).is_err());
    // The failed receive's write window (and the registration it lies in) stays
    // registered (the peer may have been writing until the queue pair failed): both
    // descriptors, the cached registration's and the window's, stay open until close.
    assert_eq!(gpu.open_fds(), 2);
    assert_eq!(b.counters().exposures_retired, 1);
    b.close(LONG).unwrap();
    a.close(LONG).unwrap();
    assert_eq!(gpu.open_fds(), 0);
    assert!(gpu.violations().is_empty(), "{:?}", gpu.violations());
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

/// A failed wait fails the endpoint, which moves every queue pair to the error state:
/// a peer granted a write can't write the window any more once the wait has reported
/// the failure, though its own link is fine (here member 1's failed instead).
#[test]
fn a_failed_wait_stops_every_granted_write() {
    let fabric = MockFabric::new();
    let nodes: Vec<NodeId> = (1..=3)
        .map(|lid| {
            fabric.add_node(vec![MockDevice::new(
                "mlx0",
                vec![MockPort::infiniband(lid)],
            )])
        })
        .collect();
    let config = config(RNDV);
    let a = Endpoint::new(fabric.provider(nodes[0]), 3, 0, config.clone()).unwrap();
    let b = RawMember::new(&fabric, nodes[1], 3, 0, &config);
    let c = RawMember::new(&fabric, nodes[2], 3, 0, &config);
    b.connect(&a.local_info(), 1);
    c.connect(&a.local_info(), 2);
    a.connect(1, &b.info).unwrap();
    a.connect(2, &c.info).unwrap();
    let len = 2 * RNDV;
    let reg = host(&a, &vec![0xAAu8; len as usize], 3);
    let window = reg.addr() as u64;
    let r = a.recv_registered(2, &reg, 0, len, 5).unwrap();
    let grant = c.receive(LONG);
    assert_eq!((grant.kind, grant.offset), (Kind::Cts, window));
    // The next send on a's node, to member 1, fails: a's wait reports it.
    fabric.inject(Fault::FailNextSend {
        node: nodes[0],
        status: WcStatus::RetryExcErr,
    });
    let data = [1u8; 8];
    // SAFETY: data outlives the wait below, which fails the endpoint.
    let s = unsafe { a.send(1, data.as_ptr() as usize, 8, 1) }.unwrap();
    assert!(matches!(a.wait(&[s], LONG), Err(WaitError::Failed(_))));
    assert_ne!(c.write(&[0x55; 64], window, grant.key), WcStatus::Success);
    // SAFETY: the registration covers len bytes at window, all still allocated.
    let now = unsafe { std::slice::from_raw_parts(window as *const u8, len as usize) };
    assert!(
        now.iter().all(|&x| x == 0xAA),
        "a write landed after the failure"
    );
    assert!(a.wait(&[r], LONG).is_err());
    a.close(LONG).ok();
    drop((b, c));
}

#[test]
fn cached_registrations_are_never_remotely_writable() {
    let gpu = MockGpu::new();
    let (_fabric, a, _b) = pair(RNDV, &gpu, false);
    let dev = gpu.alloc(0, 1 << 20);
    let regs = [
        a.register_device(0, dev, 1 << 20).unwrap(),
        host(&a, &[0u8; 4096], 1),
    ];
    for reg in regs {
        assert_eq!(reg.region().access(), CACHED_ACCESS, "{:?}", reg.method());
        assert!(!reg.region().access().contains(Access::REMOTE_WRITE));
    }
    assert!(a.telemetry().contains("\"cached_access\":\"local\""));
}

/// A write grant from endpoint A (member 0) to a bare queue pair B (member 1) for a
/// receive of `len` bytes at `at` into `kind` memory ("host", "dmabuf", "peer"), with
/// `before` bytes of other data before it and after it in the same registration. Returns
/// what B checks against.
struct Granted {
    fabric: Arc<MockFabric>,
    gpu: MockGpu,
    a: Endpoint,
    b: RawMember,
    recv: i64,
    addr: u64,
    key: u32,
    len: u64,
    /// Reads `n` bytes at `addr` of the receiving memory.
    read: Reader,
}

type Reader = Box<dyn Fn(u64, usize) -> Vec<u8>>;

fn granted(kind: &str) -> Granted {
    let gpu = MockGpu::new();
    let (fabric, na, nb) = ib_fabric();
    let config = config(RNDV);
    let a = Endpoint::new(fabric.provider(na), 2, 0, config.clone()).unwrap();
    a.set_gpu(
        Arc::new(gpu.clone()),
        modules(&format!("granted-{kind}"), kind == "peer"),
    );
    if kind == "peer" {
        gpu.set_dmabuf_unsupported(Some("not in this test"));
    }
    let b = RawMember::new(&fabric, nb, 2, 0, &config);
    b.connect(&a.local_info(), 1);
    a.connect(1, &b.info).unwrap();
    let len = 2 * RNDV;
    let total = 4 * RNDV as usize;
    // The receive starts 1000 bytes in: neighbours on both sides share its registration.
    let (reg, base, read): (_, u64, Reader) = if kind == "host" {
        let buf: &'static mut [u8] = Box::leak(vec![0xAAu8; total].into_boxed_slice());
        let ptr = buf.as_mut_ptr() as u64;
        // SAFETY: leaked, valid for the process.
        let reg = unsafe { a.register_host(ptr as usize, total, 7) }.unwrap();
        let read = Box::new(|at: u64, n: usize| {
            // SAFETY: inside the leaked buffer; read after the writes are complete.
            unsafe { std::slice::from_raw_parts(at as *const u8, n) }.to_vec()
        });
        (reg, ptr, read)
    } else {
        let dev = gpu.alloc(0, total as u64);
        gpu.write(dev, &vec![0xAAu8; total]);
        let reg = a.register_device(0, dev, total as u64).unwrap();
        let want = if kind == "peer" {
            Method::PeerMemory
        } else {
            Method::DmaBuf
        };
        assert_eq!(reg.method(), want);
        let g = gpu.clone();
        (reg, dev, Box::new(move |at: u64, n: usize| g.read(at, n)))
    };
    let at = base + 1000;
    let recv = a
        .recv_registered(1, &reg, (at - reg.addr() as u64) as usize, len, 5)
        .unwrap();
    let h = b.receive(LONG);
    assert_eq!(
        (h.kind, h.tag, h.msg_len, h.offset),
        (Kind::Cts, 5, len, at)
    );
    Granted {
        fabric,
        gpu,
        a,
        b,
        recv,
        addr: at,
        key: h.key,
        len,
        read,
    }
}

#[test]
fn a_write_grant_reaches_exactly_the_receive_and_only_until_it_is_written() {
    for kind in ["host", "dmabuf", "peer"] {
        // The byte before the receive and the byte after it are out of the key's reach
        // (each fault fails the queue pair, so each gets a fresh grant).
        for (what, offset) in [("before", -1i64), ("after", 0)] {
            let g = granted(kind);
            let target = if offset < 0 {
                g.addr - 1
            } else {
                g.addr + g.len
            };
            assert_eq!(
                g.b.write(&[0x55], target, g.key),
                WcStatus::RemAccessErr,
                "{kind}: the byte {what} the receive"
            );
            assert_eq!((g.read)(target, 1), vec![0xAA], "{kind} {what}");
            g.a.abort("test over");
            g.a.close(LONG).ok();
        }
        // Inside, all of it, then "written": the receive completes, the neighbours are
        // untouched, and the key reaches nothing any more.
        let mut g = granted(kind);
        let mrs = g.fabric.live(0).mrs;
        let data = pattern(g.len as usize, 9);
        assert_eq!(g.b.write(&data, g.addr, g.key), WcStatus::Success);
        let fin = g.b.header(Kind::Fin, 5, g.len).encode();
        assert_eq!(g.b.send(&fin), WcStatus::Success);
        g.a.wait(&[g.recv], LONG).unwrap();
        assert!((g.read)(g.addr, g.len as usize) == data, "{kind}");
        assert_eq!((g.read)(g.addr - 1000, 1000), vec![0xAA; 1000], "{kind}");
        assert_eq!((g.read)(g.addr + g.len, 1000), vec![0xAA; 1000], "{kind}");
        eventually(LONG, "the window is deregistered", || {
            g.fabric.live(0).mrs == mrs - 1
        });
        assert_eq!(g.a.counters().exposures_closed, 1);
        assert_eq!(
            g.b.write(&[0x55], g.addr, g.key),
            WcStatus::RemAccessErr,
            "{kind}: after \"written\""
        );
        let t = g.a.telemetry();
        assert!(t.contains("\"write_windows\":1,"), "{t}");
        assert!(t.contains("\"write_windows_closed\":1,"), "{t}");
        g.a.close(LONG).ok();
        assert!(g.gpu.violations().is_empty(), "{:?}", g.gpu.violations());
        assert_eq!(g.gpu.open_fds(), 0, "{kind}");
    }
}
