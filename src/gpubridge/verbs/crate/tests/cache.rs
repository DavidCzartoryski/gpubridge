//! The registration cache on the mock provider.

mod common;

use common::*;
use gpubridge_verbs::{
    Access, CacheConfig, CacheKey, Error, MrCache, SendOp, SendRequest, Sge, WcStatus,
};

/// Memory for the cache to register: outlives every registration in a test (declared
/// first, so dropped last).
struct Arena(Vec<u8>);

impl Arena {
    fn new(len: usize) -> Arena {
        Arena(vec![0u8; len])
    }

    fn key(&self, offset: usize, len: usize, owner: u64) -> CacheKey {
        CacheKey {
            addr: self.0.as_ptr() as usize + offset,
            len,
            access: Access::LOCAL_WRITE,
            owner,
        }
    }
}

fn cache(side: &Side, max_entries: usize, max_bytes: usize) -> MrCache {
    MrCache::new(
        side.pd.clone(),
        CacheConfig {
            max_entries,
            max_bytes,
        },
    )
    .unwrap()
}

#[test]
fn hits_need_the_same_owner_a_covering_range_and_the_access() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, _) = ib_fabric();
    let side = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let cache = cache(&side, 8, 1 << 20);
    // SAFETY (every get below): `arena` outlives the cache and every region from it, and
    // nothing frees it in between.
    let whole = unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap();
    let inside = unsafe { cache.get(arena.key(100, 200, 1)) }.unwrap();
    assert!(inside.same(&whole));
    let more_access = CacheKey {
        access: Access::LOCAL_WRITE | Access::REMOTE_WRITE,
        ..arena.key(0, 4096, 1)
    };
    let remote = unsafe { cache.get(more_access) }.unwrap();
    assert!(!remote.same(&whole));
    // Asking for less access than an entry has is a hit.
    let less = unsafe {
        cache.get(CacheKey {
            access: Access::NONE,
            ..arena.key(0, 64, 1)
        })
    }
    .unwrap();
    assert!(less.same(&whole) || less.same(&remote));
    // Past the end of the entry: a new registration.
    let longer = unsafe { cache.get(arena.key(0, 8192, 1)) }.unwrap();
    assert!(!longer.same(&whole));
    let s = cache.stats();
    assert_eq!((s.hits, s.misses, s.registrations), (2, 3, 3));
    assert_eq!(cache.len(), 3);
    assert_eq!(fabric.live(na).mrs, 3);
}

#[test]
fn reused_addresses_get_a_new_registration_and_the_stale_one_goes() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, _) = ib_fabric();
    let side = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let cache = cache(&side, 8, 1 << 20);
    // SAFETY: as in the first test.
    let first = unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap();
    let lkey = first.lkey();
    drop(first);
    // Same address and length, a new allocation (owner 2): never the old registration.
    let second = unsafe { cache.get(arena.key(0, 4096, 2)) }.unwrap();
    assert_ne!(second.lkey(), lkey);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.stats().stale, 1);
    assert_eq!(
        fabric.live(na).mrs,
        1,
        "the stale registration was released"
    );
    // A stale entry still in use means the old allocation was freed too early: refused.
    let held = second;
    let err = unsafe { cache.get(arena.key(1024, 64, 3)) }.unwrap_err();
    assert!(matches!(err, Error::Busy(_)), "{err:?}");
    drop(held);
}

#[test]
fn eviction_takes_the_least_recently_used_idle_entry() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, _) = ib_fabric();
    let side = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let cache = cache(&side, 2, 1 << 20);
    // SAFETY: as in the first test.
    let a = unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap().lkey();
    let b = unsafe { cache.get(arena.key(8192, 4096, 1)) }
        .unwrap()
        .lkey();
    // Touch the first: the second is now the least recently used.
    unsafe { cache.get(arena.key(0, 16, 1)) }.unwrap();
    unsafe { cache.get(arena.key(16384, 4096, 1)) }.unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.stats().evictions, 1);
    assert_eq!(
        unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap().lkey(),
        a
    );
    assert_ne!(
        unsafe { cache.get(arena.key(8192, 4096, 1)) }
            .unwrap()
            .lkey(),
        b
    );
    assert_eq!(fabric.live(na).mrs, 2);
}

#[test]
fn the_byte_bound_holds_and_oversized_requests_are_refused() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, _) = ib_fabric();
    let side = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let cache = cache(&side, 16, 8192);
    // SAFETY: as in the first test.
    unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap();
    unsafe { cache.get(arena.key(4096, 4096, 1)) }.unwrap();
    unsafe { cache.get(arena.key(8192, 2048, 1)) }.unwrap();
    assert!(cache.bytes() <= 8192);
    assert_eq!(cache.stats().evictions, 1);
    let err = unsafe { cache.get(arena.key(0, 8193, 1)) }.unwrap_err();
    assert!(matches!(err, Error::Exhausted(_)));
    assert_eq!(fabric.live(na).mrs, cache.len());
}

#[test]
fn entries_in_use_are_never_evicted() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, _) = ib_fabric();
    let side = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let cache = cache(&side, 2, 1 << 20);
    // SAFETY: as in the first test.
    let held1 = unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap();
    let held2 = unsafe { cache.get(arena.key(4096, 4096, 1)) }.unwrap();
    let err = unsafe { cache.get(arena.key(8192, 4096, 1)) }.unwrap_err();
    assert!(matches!(err, Error::Exhausted(_)), "{err:?}");
    assert_eq!(cache.stats().refused, 1);
    assert_eq!(fabric.live(na).mrs, 2);
    drop(held1);
    unsafe { cache.get(arena.key(8192, 4096, 1)) }.unwrap();
    assert_eq!(cache.stats().evictions, 1);
    drop(held2);
}

#[test]
fn posted_work_pins_an_entry_until_its_completion_is_polled() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let cache = cache(&b, 1, 1 << 20);
    // SAFETY: as in the first test.
    let region = unsafe { cache.get(arena.key(0, 4096, 1)) }.unwrap();
    b.qp.post_recv(1, &[Sge::new(&region, 0, 4096)]).unwrap();
    drop(region);
    // No caller holds it, but the posted receive does.
    assert!(matches!(
        unsafe { cache.get(arena.key(8192, 4096, 1)) },
        Err(Error::Exhausted(_))
    ));
    let src = a.pd.register_buffer(16, Access::LOCAL_WRITE).unwrap();
    a.qp.post_send(&SendRequest {
        tag: 1,
        op: SendOp::Send,
        sges: &[Sge::new(&src, 0, 16)],
    })
    .unwrap();
    assert_eq!(wait(&b.cq, 1, 4)[0].status, WcStatus::Success);
    assert_eq!(&arena.0[..16], &[0u8; 16]);
    unsafe { cache.get(arena.key(8192, 4096, 1)) }.unwrap();
    assert_eq!(fabric.live(nb).mrs, 1);
    assert!(fabric.violations().is_empty());
}

#[test]
fn invalidation_removes_entries_at_once_and_releases_them_when_unused() {
    let arena = Arena::new(1 << 16);
    let (fabric, na, _) = ib_fabric();
    let side = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let cache = cache(&side, 8, 1 << 20);
    // SAFETY: as in the first test.
    let held = unsafe { cache.get(arena.key(0, 4096, 7)) }.unwrap();
    unsafe { cache.get(arena.key(4096, 4096, 7)) }.unwrap();
    unsafe { cache.get(arena.key(8192, 4096, 8)) }.unwrap();
    let out = cache.invalidate_owner(7);
    assert_eq!((out.removed, out.in_use), (2, 1));
    assert_eq!(cache.len(), 1);
    // The held one is still registered until its handle goes.
    assert_eq!(fabric.live(na).mrs, 2);
    drop(held);
    assert_eq!(fabric.live(na).mrs, 1);
    let out = cache.invalidate_range(arena.0.as_ptr() as usize + 8192 + 100, 1);
    assert_eq!((out.removed, out.in_use), (1, 0));
    assert!(cache.is_empty());
    assert_eq!(fabric.live(na).mrs, 0);
    assert_eq!(cache.stats().invalidations, 3);
}
