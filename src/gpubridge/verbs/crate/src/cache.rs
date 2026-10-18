//! A bounded cache of memory registrations, keyed by allocation identity.
//!
//! Registering memory is slow (the kernel pins its pages), so a transport keeps
//! registrations of buffers it sees again. An address alone doesn't identify a buffer: an
//! allocator hands the same address out again after a free. So the key is the address,
//! length and access rights plus an owner token the caller supplies, which changes when
//! the memory at that address is a different allocation (an allocation generation, a
//! storage id). A registration of the same owner that contains the requested range serves
//! it; registering for a new owner drops cached entries of other owners that overlap it,
//! since that memory was freed and handed out again (and refuses, [`Error::Busy`], if one
//! of those is still in use: it was freed too early).
//!
//! The cache holds at most `max_entries` registrations covering at most `max_bytes`,
//! evicting the least recently used idle one when full. An entry is idle when the cache
//! holds its only handle: no caller holds one and no posted work does (posted work holds
//! a clone of each region it uses until its completion is polled). Entries in use are
//! never evicted or deregistered, so a cache full of them refuses a new registration
//! ([`Error::Exhausted`]) instead of growing. Invalidated entries leave the cache at once
//! but stay registered until the last handle goes. Holding the address is not holding
//! the registration; holding the [`MemoryRegion`] is.
//!
//! Lookups scan the entries (the cache is small and bounded); that is cheap next to a
//! registration.

use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::resources::{MemoryRegion, ProtectionDomain};
use crate::types::Access;

/// Identifies an allocation's bytes: `len` bytes at `addr`, of allocation `owner`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub addr: usize,
    pub len: usize,
    pub access: Access,
    /// Changes whenever the memory at this address is a different allocation.
    pub owner: u64,
}

impl CacheKey {
    fn end(&self) -> usize {
        self.addr + self.len
    }

    fn overlaps(&self, addr: usize, len: usize) -> bool {
        self.addr < addr + len && addr < self.end()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub max_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub registrations: u64,
    pub evictions: u64,
    /// Entries dropped because another owner's allocation overlapped them.
    pub stale: u64,
    /// Entries dropped by `invalidate_*`.
    pub invalidations: u64,
    /// Lookups refused because every entry was in use.
    pub refused: u64,
}

/// What an invalidation did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Invalidated {
    /// Entries removed from the cache.
    pub removed: usize,
    /// Of those, ones still in use (by the caller or by posted work): they stay registered
    /// until that use ends, and the memory must stay valid until then.
    pub in_use: usize,
}

struct CacheEntry {
    key: CacheKey,
    region: MemoryRegion,
    last_use: u64,
}

impl CacheEntry {
    fn idle(&self) -> bool {
        // The cache's own handle is the only one: nobody can clone it but the cache, which
        // holds its lock while deciding.
        self.region.holders() == 1
    }
}

struct CacheState {
    entries: Vec<CacheEntry>,
    bytes: usize,
    clock: u64,
    stats: CacheStats,
}

pub struct MrCache {
    pd: ProtectionDomain,
    config: CacheConfig,
    state: Mutex<CacheState>,
}

impl MrCache {
    pub fn new(pd: ProtectionDomain, config: CacheConfig) -> Result<MrCache> {
        if config.max_entries == 0 || config.max_bytes == 0 {
            return Err(Error::invalid(
                "a cache needs room for at least one entry and byte",
            ));
        }
        Ok(MrCache {
            pd,
            config,
            state: Mutex::new(CacheState {
                entries: Vec::new(),
                bytes: 0,
                clock: 0,
                stats: CacheStats::default(),
            }),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The registration for `key`: a cached one of the same owner whose range and access
    /// cover it, or a new one.
    ///
    /// # Safety
    ///
    /// `key.len` bytes at `key.addr` must be a live allocation identified by `key.owner`,
    /// and stay valid until the caller has invalidated it ([`invalidate_owner`] or
    /// [`invalidate_range`]) *and* every handle to its registration (the returned one and
    /// any work posted with it) is gone. Freeing it sooner lets the device write into
    /// freed memory.
    ///
    /// [`invalidate_owner`]: MrCache::invalidate_owner
    /// [`invalidate_range`]: MrCache::invalidate_range
    pub unsafe fn get(&self, key: CacheKey) -> Result<MemoryRegion> {
        if key.len == 0 {
            return Err(Error::invalid("a memory region can't be empty"));
        }
        if key.addr.checked_add(key.len).is_none() {
            return Err(Error::invalid("address range overflows"));
        }
        let mut st = self.lock();
        st.clock += 1;
        let now = st.clock;
        if let Some(entry) = st.entries.iter_mut().find(|e| {
            e.key.owner == key.owner
                && e.key.access.contains(key.access)
                && e.key.addr <= key.addr
                && key.end() <= e.key.end()
        }) {
            entry.last_use = now;
            let region = entry.region.clone();
            st.stats.hits += 1;
            return Ok(region);
        }
        st.stats.misses += 1;
        // Another owner's entries over this memory: that allocation was freed. If one is
        // still in use, it was freed too early (see the contract): refuse rather than
        // register the same pages for two owners.
        let stale = |k: &CacheKey| k.owner != key.owner && k.overlaps(key.addr, key.len);
        if let Some(busy) = st.entries.iter().find(|e| stale(&e.key) && !e.idle()) {
            let owner = busy.key.owner;
            st.stats.refused += 1;
            return Err(Error::Busy(format!(
                "this memory is still registered and in use for owner {owner:#x}"
            )));
        }
        let removed = self.remove(&mut st, stale);
        st.stats.stale += removed.removed as u64;
        if key.len > self.config.max_bytes {
            st.stats.refused += 1;
            return Err(Error::Exhausted(format!(
                "{} bytes is more than the cache's {}",
                key.len, self.config.max_bytes
            )));
        }
        while st.entries.len() + 1 > self.config.max_entries
            || st.bytes + key.len > self.config.max_bytes
        {
            let victim = st
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.idle())
                .min_by_key(|(_, e)| e.last_use)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    let entry = st.entries.swap_remove(i);
                    st.bytes -= entry.key.len;
                    st.stats.evictions += 1;
                    // The last handle: deregistered here.
                    drop(entry);
                }
                None => {
                    st.stats.refused += 1;
                    return Err(Error::Exhausted(format!(
                        "all {} cached registrations ({} bytes) are in use",
                        st.entries.len(),
                        st.bytes
                    )));
                }
            }
        }
        // SAFETY: the caller keeps the allocation valid until it is invalidated and every
        // handle to the registration is gone (this function's contract); the registration
        // is released no sooner than that.
        let region = unsafe { self.pd.register(key.addr as *mut u8, key.len, key.access) }?;
        st.stats.registrations += 1;
        st.bytes += key.len;
        st.entries.push(CacheEntry {
            key,
            region: region.clone(),
            last_use: now,
        });
        Ok(region)
    }

    fn remove(&self, st: &mut CacheState, matches: impl Fn(&CacheKey) -> bool) -> Invalidated {
        let mut out = Invalidated::default();
        let mut i = 0;
        while i < st.entries.len() {
            if matches(&st.entries[i].key) {
                let entry = st.entries.swap_remove(i);
                st.bytes -= entry.key.len;
                out.removed += 1;
                if !entry.idle() {
                    out.in_use += 1;
                }
                // Deregistered now if idle, else when the last handle goes.
                drop(entry);
            } else {
                i += 1;
            }
        }
        out
    }

    /// Drops every entry of allocation `owner` (it is being freed).
    pub fn invalidate_owner(&self, owner: u64) -> Invalidated {
        let mut st = self.lock();
        let out = self.remove(&mut st, |k| k.owner == owner);
        st.stats.invalidations += out.removed as u64;
        out
    }

    /// Drops every entry overlapping `len` bytes at `addr`, whatever its owner.
    pub fn invalidate_range(&self, addr: usize, len: usize) -> Invalidated {
        let mut st = self.lock();
        let out = self.remove(&mut st, |k| k.overlaps(addr, len));
        st.stats.invalidations += out.removed as u64;
        out
    }

    /// Drops every entry.
    pub fn clear(&self) -> Invalidated {
        let mut st = self.lock();
        let out = self.remove(&mut st, |_| true);
        st.stats.invalidations += out.removed as u64;
        out
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes the cached entries cover.
    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    pub fn stats(&self) -> CacheStats {
        self.lock().stats
    }

    pub fn config(&self) -> CacheConfig {
        self.config
    }
}
