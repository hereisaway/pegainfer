//! Joint full-attention KV and recurrent/conv prefix cache for Qwen3.5.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use cudarc::driver::PinnedHostSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvCacheManager;
use pegainfer_kv_cache::KvView;
use pegainfer_kv_cache::RequestKv;

use crate::config::Config35;
use crate::config::LocalGeometry;
use crate::recurrent_state::RecurrentState;

/// Token interval at which a complete recurrent/conv snapshot may be cached.
pub(crate) const SNAPSHOT_STRIDE_TOKENS: usize = 256;

/// Last stride boundary at or below `history_tokens` (the prompt length without one).
pub(crate) fn history_stop(
    prompt_len: usize,
    history_tokens: Option<usize>,
    stride: usize,
) -> usize {
    history_tokens.map_or(prompt_len, |h| h.min(prompt_len)) / stride * stride
}

/// Content-addressed identity of one complete hybrid-model prefix boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PrefixBoundaryKey {
    /// Canonical full-attention KV lineage hash for this prefix.
    pub(crate) sequence_hash: [u8; 16],
    /// Exclusive token position represented by both KV and recurrent state.
    pub(crate) boundary_tokens: usize,
}

/// Cumulative counters for joint KV/recurrent prefix-cache activity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixCacheStats {
    /// Requests that restored both KV and a recurrent snapshot.
    pub(crate) joint_hits: u64,
    /// Prompt tokens reused across all joint hits.
    pub(crate) joint_hit_tokens: u64,
    /// Requests with eligible resident KV but no matching snapshot.
    pub(crate) kv_only_fallbacks: u64,
    /// Individual eligible boundaries without a matching snapshot.
    pub(crate) snapshot_misses: u64,
    /// New recurrent snapshots published.
    pub(crate) inserts: u64,
    /// Published snapshots replaced by LRU insertion.
    pub(crate) evictions: u64,
    /// Prompt tokens looked up in the joint prefix cache.
    pub(crate) queries: u64,
    /// Of `joint_hit_tokens`, tokens restored from the pinned host tier.
    pub(crate) host_hits: u64,
    /// GPU -> host spills.
    pub(crate) spills: u64,
    /// Host entries evicted to make room for a spill.
    pub(crate) host_evictions: u64,
    /// Insertions skipped because every slot was pinned or no host slot was free.
    pub(crate) skips: u64,
}

/// Counters that live in the snapshot store rather than the prefix-cache facade.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SnapshotStats {
    pub(crate) spills: u64,
    pub(crate) host_evictions: u64,
    pub(crate) skips: u64,
    pub(crate) gpu_occupancy: usize,
    pub(crate) host_occupancy: usize,
}

/// Which backing tier holds one published snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SnapshotTier {
    Gpu,
    Host,
}

/// One recurrent snapshot indexed by a reusable prefix boundary.
struct SnapshotEntry {
    /// Physical slot in the GPU store (`Gpu`) or the pinned host slab (`Host`).
    slot: usize,
    tier: SnapshotTier,
    /// Logical timestamp used to select an unpinned LRU victim.
    last_used: AtomicU64,
}

/// RAII pin on a recurrent snapshot during restore.
pub(crate) struct SnapshotGuard {
    /// Token boundary represented by the pinned entry.
    boundary: usize,
    /// The directory's extra strong references are active restore pins.
    entry: Arc<SnapshotEntry>,
}

impl SnapshotGuard {
    pub(crate) fn boundary(&self) -> usize {
        self.boundary
    }

    /// Slot number in the GPU store or the host slab, depending on [`Self::tier`].
    pub(crate) fn recurrent_slot(&self) -> usize {
        self.entry.slot
    }

    pub(crate) fn tier(&self) -> SnapshotTier {
        self.entry.tier
    }
}

/// Mutable prefix-cache directory shared by every execution rank.
///
/// This owns no device memory; the physical GPU slots and the pinned host slab
/// live in [`RecurrentSnapshotStore`].
struct SnapshotCache {
    /// Published boundary key to its recurrent snapshot entry.
    entries: HashMap<PrefixBoundaryKey, Arc<SnapshotEntry>>,
    /// GPU slots never published or returned by an abort.
    free_gpu_slots: Vec<usize>,
    /// Host slots never published or returned by a host eviction.
    free_host_slots: Vec<usize>,
    /// Number of preallocated GPU snapshot slots (the legacy capacity metric).
    gpu_slot_count: usize,
    /// Monotonic logical time for LRU ordering.
    clock: u64,
    spills: u64,
    host_evictions: u64,
    skips: u64,
}

impl SnapshotCache {
    fn new(gpu_slot_count: usize, host_slot_count: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(gpu_slot_count + host_slot_count),
            free_gpu_slots: (0..gpu_slot_count).rev().collect(),
            free_host_slots: (0..host_slot_count).rev().collect(),
            gpu_slot_count,
            clock: 0,
            spills: 0,
            host_evictions: 0,
            skips: 0,
        }
    }

    /// Number of currently published joint entries, GPU and host.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of preallocated GPU snapshot slots.
    fn capacity(&self) -> usize {
        self.gpu_slot_count
    }

    /// Advance the non-zero logical clock used by the LRU policy.
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    /// Look up `key`, refresh its LRU timestamp, and pin its entry.
    fn lookup(&mut self, key: PrefixBoundaryKey) -> Option<SnapshotGuard> {
        let last_used = self.tick();
        let entry = self.entries.get(&key)?;
        entry.last_used.store(last_used, Ordering::Relaxed);
        Some(SnapshotGuard {
            boundary: key.boundary_tokens,
            entry: Arc::clone(entry),
        })
    }

    fn contains(&self, key: PrefixBoundaryKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Refresh a duplicate key without reserving another snapshot.
    fn touch(&mut self, key: PrefixBoundaryKey) {
        let last_used = self.tick();
        if let Some(entry) = self.entries.get(&key) {
            entry.last_used.store(last_used, Ordering::Relaxed);
        }
    }

    fn take_free_gpu_slot(&mut self) -> Option<usize> {
        self.free_gpu_slots.pop()
    }

    fn release_gpu_slot(&mut self, slot: usize) {
        self.free_gpu_slots.push(slot);
    }

    fn take_free_host_slot(&mut self) -> Option<usize> {
        self.free_host_slots.pop()
    }

    fn release_host_slot(&mut self, slot: usize) {
        self.free_host_slots.push(slot);
    }

    fn record_spill(&mut self) {
        self.spills += 1;
    }

    fn record_host_eviction(&mut self) {
        self.host_evictions += 1;
    }

    fn record_skip(&mut self) {
        self.skips += 1;
    }

    fn occupancy(&self, tier: SnapshotTier) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.tier == tier)
            .count()
    }

    /// Remove the oldest unpinned entry of `tier`, returning its key and slot.
    fn take_unpinned_victim(&mut self, tier: SnapshotTier) -> Option<(PrefixBoundaryKey, usize)> {
        let (&victim_key, slot) = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.tier == tier && Arc::strong_count(entry) == 1)
            .min_by_key(|(_, entry)| entry.last_used.load(Ordering::Relaxed))
            .map(|(key, entry)| (key, entry.slot))?;
        let victim = self
            .entries
            .remove(&victim_key)
            .expect("LRU victim must still be present");
        debug_assert_eq!(victim.slot, slot);
        debug_assert_eq!(victim.tier, tier);
        Some((victim_key, slot))
    }

    /// Publish a new GPU-resident entry after its copy succeeded.
    fn insert(&mut self, key: PrefixBoundaryKey, slot: usize, tier: SnapshotTier) {
        let last_used = self.tick();
        let previous = self.entries.insert(
            key,
            Arc::new(SnapshotEntry {
                slot,
                tier,
                last_used: AtomicU64::new(last_used),
            }),
        );
        debug_assert!(previous.is_none());
    }
}

/// One pending central-directory insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SnapshotReservation {
    k: PrefixBoundaryKey,
    slot: usize,
    /// Set when an unpinned GPU entry was dropped instead of spilled.
    replaced: Option<SnapshotTier>,
}

impl SnapshotReservation {
    pub(crate) fn recurrent_slot(self) -> usize {
        self.slot
    }

    fn was_replacement(self) -> bool {
        self.replaced.is_some()
    }
}

/// Pinned-host backing for recurrent/conv snapshots.
struct HostSnapshotSlab {
    state: PinnedHostSlice<f32>,
    conv: PinnedHostSlice<half::bf16>,
    slots: usize,
    layers: usize,
    state_per_layer: usize,
    conv_per_layer: usize,
    state_per_slot: usize,
    conv_per_slot: usize,
}

impl HostSnapshotSlab {
    fn new(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        slots: usize,
    ) -> Result<Self> {
        anyhow::ensure!(slots > 0, "host snapshot slab needs at least one slot");
        let layers = config.num_hidden_layers - config.num_full_attention_layers();
        let state_per_layer = geometry.local_linear_num_value_heads()
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_per_layer = geometry.local_linear_qkv_dim() * (config.linear_conv_kernel_dim - 1);
        let state_per_slot = layers * state_per_layer;
        let conv_per_slot = layers * conv_per_layer;
        let state = unsafe { ctx.ctx.alloc_pinned::<f32>(slots * state_per_slot) }
            .map_err(|e| anyhow::anyhow!("allocate pinned recurrent slab: {e}"))?;
        let conv = unsafe { ctx.ctx.alloc_pinned::<half::bf16>(slots * conv_per_slot) }
            .map_err(|e| anyhow::anyhow!("allocate pinned conv slab: {e}"))?;
        Ok(Self {
            state,
            conv,
            slots,
            layers,
            state_per_layer,
            conv_per_layer,
            state_per_slot,
            conv_per_slot,
        })
    }

    fn spill_from(&mut self, ctx: &DeviceContext, src: &RecurrentState, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.slots, "host snapshot slot {slot} out of range");
        anyhow::ensure!(
            src.layers.len() == self.layers,
            "host snapshot layer mismatch: src={}, expected={}",
            src.layers.len(),
            self.layers
        );
        let state = self.state.as_mut_slice()?;
        let conv = self.conv.as_mut_slice()?;
        let state_base = slot * self.state_per_slot;
        let conv_base = slot * self.conv_per_slot;
        for layer in 0..self.layers {
            let state_at = state_base + layer * self.state_per_layer;
            ctx.stream.memcpy_dtoh(
                &src.layers[layer].state,
                &mut state[state_at..state_at + self.state_per_layer],
            )?;
            let conv_at = conv_base + layer * self.conv_per_layer;
            ctx.stream.memcpy_dtoh(
                &src.layers[layer].conv_state.data,
                &mut conv[conv_at..conv_at + self.conv_per_layer],
            )?;
        }
        ctx.stream.synchronize()?;
        Ok(())
    }

    fn restore_into(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        anyhow::ensure!(slot < self.slots, "host snapshot slot {slot} out of range");
        anyhow::ensure!(
            dst.layers.len() == self.layers,
            "host snapshot layer mismatch: dst={}, expected={}",
            dst.layers.len(),
            self.layers
        );
        let state = self.state.as_slice()?;
        let conv = self.conv.as_slice()?;
        let state_base = slot * self.state_per_slot;
        let conv_base = slot * self.conv_per_slot;
        for layer in 0..self.layers {
            let state_at = state_base + layer * self.state_per_layer;
            ctx.stream.memcpy_htod(
                &state[state_at..state_at + self.state_per_layer],
                &mut dst.layers[layer].state,
            )?;
            let conv_at = conv_base + layer * self.conv_per_layer;
            ctx.stream.memcpy_htod(
                &conv[conv_at..conv_at + self.conv_per_layer],
                &mut dst.layers[layer].conv_state.data,
            )?;
        }
        // v1 restores synchronously on the admission thread. The explicit sync
        // also keeps the pinned source alive until the H2D copy has completed.
        ctx.stream.synchronize()?;
        Ok(())
    }
}

/// Rank-local physical recurrent/conv snapshot allocations.
pub(crate) struct RecurrentStateStore {
    slots: Vec<RecurrentState>,
}

impl RecurrentStateStore {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        slot_count: usize,
    ) -> Result<Self> {
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(RecurrentState::new(ctx, config, geometry)?);
        }
        Ok(Self { slots })
    }

    pub(crate) fn save(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        src: &RecurrentState,
    ) -> Result<()> {
        let dst = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?;
        dst.copy_from(ctx, src)
    }

    pub(crate) fn restore(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        let src = self
            .slots
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?;
        dst.copy_from(ctx, src)
    }
}

/// Owns the recurrent/conv snapshot directory plus the physical GPU slots and,
/// when configured, the pinned host backing slab. The TP controller builds a
/// directory-only store; each TP worker keeps its own `RecurrentStateStore`.
pub(crate) struct RecurrentSnapshotStore {
    cache: SnapshotCache,
    slots: Option<RecurrentStateStore>,
    host: Option<HostSnapshotSlab>,
    ctx: Option<DeviceContext>,
}

impl RecurrentSnapshotStore {
    /// Directory only: no physical slots (TP controller, zero-budget builds).
    pub(crate) fn new_directory(gpu_slot_count: usize) -> Self {
        Self {
            cache: SnapshotCache::new(gpu_slot_count, 0),
            slots: None,
            host: None,
            ctx: None,
        }
    }

    /// Directory plus GPU-resident physical slots and an optional pinned host
    /// slab. `host_slot_count == 0` is exactly the step-1 behavior.
    pub(crate) fn new_gpu(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        gpu_slot_count: usize,
        host_slot_count: usize,
    ) -> Result<Self> {
        let slots = RecurrentStateStore::new(ctx, config, geometry, gpu_slot_count)?;
        let host = if host_slot_count == 0 {
            None
        } else {
            Some(HostSnapshotSlab::new(
                ctx,
                config,
                geometry,
                host_slot_count,
            )?)
        };
        Ok(Self {
            cache: SnapshotCache::new(gpu_slot_count, host_slot_count),
            slots: Some(slots),
            host,
            ctx: Some(ctx.clone()),
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.cache.capacity()
    }

    pub(crate) fn len(&self) -> usize {
        self.cache.len()
    }

    pub(crate) fn stats(&self) -> SnapshotStats {
        SnapshotStats {
            spills: self.cache.spills,
            host_evictions: self.cache.host_evictions,
            skips: self.cache.skips,
            gpu_occupancy: self.cache.occupancy(SnapshotTier::Gpu),
            host_occupancy: self.cache.occupancy(SnapshotTier::Host),
        }
    }

    pub(crate) fn contains(&self, key: PrefixBoundaryKey) -> bool {
        self.cache.contains(key)
    }

    pub(crate) fn lookup(&mut self, key: PrefixBoundaryKey) -> Option<SnapshotGuard> {
        self.cache.lookup(key)
    }

    /// Reserve a free or reused GPU slot. With a host slab, the GPU victim is
    /// spilled to pinned host before its slot is reused.
    pub(crate) fn reserve(
        &mut self,
        key: PrefixBoundaryKey,
    ) -> Result<Option<SnapshotReservation>> {
        if self.cache.contains(key) {
            self.cache.touch(key);
            return Ok(None);
        }
        if let Some(slot) = self.cache.take_free_gpu_slot() {
            return Ok(Some(SnapshotReservation {
                k: key,
                slot,
                replaced: None,
            }));
        }
        let Some((victim_key, victim_slot)) = self.cache.take_unpinned_victim(SnapshotTier::Gpu)
        else {
            self.cache.record_skip();
            return Ok(None);
        };

        let Some(host) = self.host.as_mut() else {
            return Ok(Some(SnapshotReservation {
                k: key,
                slot: victim_slot,
                replaced: Some(SnapshotTier::Gpu),
            }));
        };
        let host_slot = if let Some(slot) = self.cache.take_free_host_slot() {
            slot
        } else if let Some((_, slot)) = self.cache.take_unpinned_victim(SnapshotTier::Host) {
            self.cache.record_host_eviction();
            slot
        } else {
            self.cache.record_skip();
            self.cache
                .insert(victim_key, victim_slot, SnapshotTier::Gpu);
            return Ok(None);
        };
        let ctx = self
            .ctx
            .as_ref()
            .expect("host snapshot store requires a device context");
        let slots = self
            .slots
            .as_ref()
            .expect("host snapshot store requires GPU slots");
        let src = slots
            .slots
            .get(victim_slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {victim_slot} out of range"))?;
        if let Err(error) = host.spill_from(ctx, src, host_slot) {
            self.cache.release_host_slot(host_slot);
            self.cache
                .insert(victim_key, victim_slot, SnapshotTier::Gpu);
            return Err(error);
        }
        self.cache.record_spill();
        self.cache.insert(victim_key, host_slot, SnapshotTier::Host);
        Ok(Some(SnapshotReservation {
            k: key,
            slot: victim_slot,
            replaced: None,
        }))
    }

    pub(crate) fn publish(&mut self, reservation: SnapshotReservation) {
        self.cache
            .insert(reservation.k, reservation.slot, SnapshotTier::Gpu);
    }

    pub(crate) fn abort(&mut self, reservation: SnapshotReservation) {
        debug_assert!(!self.cache.contains(reservation.k));
        self.cache.release_gpu_slot(reservation.slot);
    }

    pub(crate) fn save(
        &mut self,
        ctx: &DeviceContext,
        reservation: SnapshotReservation,
        src: &RecurrentState,
    ) -> Result<()> {
        let slots = self
            .slots
            .as_mut()
            .expect("snapshot store has no GPU slots; directory-only stores cannot save");
        slots.save(ctx, reservation.recurrent_slot(), src)
    }

    pub(crate) fn restore_into(
        &self,
        ctx: &DeviceContext,
        guard: &SnapshotGuard,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        match guard.tier() {
            SnapshotTier::Gpu => {
                let slots = self.slots.as_ref().expect(
                    "snapshot store has no GPU slots; directory-only stores cannot restore",
                );
                slots.restore(ctx, guard.recurrent_slot(), dst)
            }
            SnapshotTier::Host => {
                let host = self
                    .host
                    .as_ref()
                    .expect("host snapshot entry without a host slab");
                host.restore_into(ctx, guard.recurrent_slot(), dst)?;
                dst.seq_len = guard.boundary();
                Ok(())
            }
        }
    }
}

/// The only Qwen3.5 scheduler interface allowed to reconcile paged KV with
/// recurrent/conv state.
pub(crate) struct Qwen35PrefixCache {
    /// Logical block pool paired with the full-attention GPU KV buffer.
    kv: KvCacheManager,
    /// Prefix key, slot, pin, and LRU state for reusable entries.
    snapshots: RecurrentSnapshotStore,
    /// Scheduler-thread-owned cumulative metrics.
    stats: PrefixCacheStats,
}

impl Qwen35PrefixCache {
    /// Build the joint coordinator for `snapshot_slots` per-rank allocations.
    pub(crate) fn new(kv: KvCacheManager, snapshot_slots: usize) -> Result<Self> {
        Self::new_with_store(kv, RecurrentSnapshotStore::new_directory(snapshot_slots))
    }

    /// Build the joint coordinator around a caller-provided snapshot store.
    pub(crate) fn new_with_store(
        kv: KvCacheManager,
        snapshots: RecurrentSnapshotStore,
    ) -> Result<Self> {
        anyhow::ensure!(
            SNAPSHOT_STRIDE_TOKENS.is_multiple_of(kv.pool().block_size()),
            "Qwen3.5 snapshot stride {SNAPSHOT_STRIDE_TOKENS} must be a multiple of KV block size {}",
            kv.pool().block_size()
        );
        Ok(Self {
            kv,
            snapshots,
            stats: PrefixCacheStats::default(),
        })
    }

    /// Logical KV block pool used for request allocation and admission.
    pub(crate) fn pool(&self) -> &pegainfer_kv_cache::BlockPool {
        self.kv.pool()
    }

    /// Rank-0 physical full-attention KV storage indexed by [`Self::pool`].
    pub(crate) fn buffer(&self) -> &KvBuffer {
        self.kv.buffer()
    }

    /// Whether joint prefix reuse and snapshot publication are enabled.
    pub(crate) fn enabled(&self) -> bool {
        self.snapshots.capacity() > 0
    }

    /// Total number of preallocated recurrent snapshot slots.
    pub(crate) fn snapshot_slots(&self) -> usize {
        self.snapshots.capacity()
    }

    /// Number of snapshot slots that currently have a published key.
    pub(crate) fn snapshot_occupancy(&self) -> usize {
        self.snapshots.len()
    }

    /// Return a point-in-time copy of cumulative cache metrics.
    pub(crate) fn stats(&self) -> PrefixCacheStats {
        let mut stats = self.stats;
        let store = self.snapshots.stats();
        stats.spills = store.spills;
        stats.host_evictions = store.host_evictions;
        stats.skips = store.skips;
        stats
    }

    /// Return the longest prefix with both active KV and a recurrent snapshot.
    /// Its KV pages are outside the available pool and may be shared.
    pub(crate) fn active_joint_prefix_pages(
        &self,
        prompt_tokens: &[u32],
        lora_name: Option<&str>,
        history_tokens: Option<usize>,
    ) -> usize {
        if !self.enabled() {
            return 0;
        }
        let hashes = self
            .kv
            .pool()
            .active_prefix_hashes(prompt_tokens.to_vec(), lora_name);
        let block_size = self.kv.pool().block_size();
        let resident_tokens = (hashes.len() * block_size).min(history_stop(
            prompt_tokens.len(),
            history_tokens,
            SNAPSHOT_STRIDE_TOKENS,
        ));
        eligible_boundaries(resident_tokens, SNAPSHOT_STRIDE_TOKENS)
            .find(|&boundary| {
                let key = PrefixBoundaryKey {
                    sequence_hash: hashes[boundary / block_size - 1],
                    boundary_tokens: boundary,
                };
                self.snapshots.contains(key)
            })
            .map_or(0, |boundary| boundary / block_size)
    }

    /// Create request-local KV state and select the longest joint prefix.
    ///
    /// If a joint prefix is found, the KV blocks are attached to the request
    /// and a guard is returned to prevent eviction until the recurrent restore
    /// is enqueued (see `Qwen35PrefixCache::finish_restore`).
    /// If no joint prefix is found, the request is still created and returned.
    pub(crate) fn begin_request(
        &mut self,
        prompt_tokens: &[u32],
        max_output_tokens: usize,
        lora_name: Option<&str>,
        history_tokens: Option<usize>,
        allow_match: bool,
    ) -> Result<(RequestKv, Option<SnapshotGuard>)> {
        let mut request =
            self.kv
                .pool()
                .new_request(prompt_tokens.to_vec(), max_output_tokens, lora_name);
        if !self.enabled() || !allow_match {
            return Ok((request, None));
        }
        self.stats.queries += prompt_tokens.len() as u64;

        let probe = self
            .kv
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_name);
        // Resuming past the history stop would skip it, so the next turn could not reuse it.
        let resident_tokens = (probe.reusable_blocks() * self.kv.pool().block_size()).min(
            history_stop(prompt_tokens.len(), history_tokens, SNAPSHOT_STRIDE_TOKENS),
        );
        let mut saw_eligible_kv = false;
        for boundary in eligible_boundaries(resident_tokens, SNAPSHOT_STRIDE_TOKENS) {
            saw_eligible_kv = true;
            let Some(sequence_hash) = probe.boundary_hash(boundary) else {
                continue;
            };
            let key = PrefixBoundaryKey {
                sequence_hash,
                boundary_tokens: boundary,
            };
            let Some(guard) = self.snapshots.lookup(key) else {
                self.stats.snapshot_misses += 1;
                continue;
            };

            let max_blocks = boundary / self.kv.pool().block_size();
            let attached = match request.match_and_add_prefix_up_to(self.kv.pool(), max_blocks) {
                Ok(attached) => attached,
                Err(error) => {
                    let _ = request.release();
                    return Err(error);
                }
            };
            if attached != boundary {
                let _ = request.release();
                anyhow::bail!(
                    "Qwen3.5 joint prefix attach selected {boundary} tokens but attached {attached}"
                );
            }
            return Ok((request, Some(guard)));
        }

        if saw_eligible_kv {
            self.stats.kv_only_fallbacks += 1;
        }
        Ok((request, None))
    }

    /// Finish a restore after the caller has enqueued the recurrent-state copy.
    ///
    /// `begin_request` performs lookup and KV attachment. The physical
    /// recurrent-state copy remains executor-specific: single-GPU execution
    /// copies from the local store, while TP coordinates all workers. This
    /// validates the positions and releases the guard once every copy is
    /// enqueued in stream order.
    pub(crate) fn finish_restore(
        &mut self,
        request: &RequestKv,
        guard: SnapshotGuard,
        recurrent_positions: &[usize],
    ) -> Result<usize> {
        let boundary = guard.boundary();
        anyhow::ensure!(
            request.kv_position() == boundary
                && !recurrent_positions.is_empty()
                && recurrent_positions
                    .iter()
                    .all(|&position| position == boundary),
            "Qwen3.5 joint prefix restore position mismatch: kv={}, recurrent={recurrent_positions:?}, boundary={}",
            request.kv_position(),
            boundary,
        );
        let tier = guard.tier();
        self.stats.joint_hits += 1;
        self.stats.joint_hit_tokens += boundary as u64;
        if tier == SnapshotTier::Host {
            self.stats.host_hits += boundary as u64;
        }
        // End the cache pin after restore enqueue and position validation.
        drop(guard);
        Ok(boundary)
    }

    /// Reserve the KV pages required by the next prefill forward.
    pub(crate) fn schedule_prefill(&self, request: &mut RequestKv, tokens: usize) -> Result<()> {
        request
            .schedule_prefill(tokens, self.kv.pool())
            .map_err(|e| anyhow::anyhow!("Qwen3.5 prefill KV schedule failed: {e}"))
    }

    /// Build the exact, immutable KV page-table view for prefill kernels.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn prefill_view(&self, request: &RequestKv, tokens: usize) -> KvView {
        request.prefill_view(tokens)
    }

    /// Apply one successful whole-model prefill window.
    ///
    /// KV is applied for every window. The returned boundary can be passed to
    /// [`Self::reserve_snapshot`], which accepts only non-zero multiples of
    /// [`SNAPSHOT_STRIDE_TOKENS`].
    pub(crate) fn apply_prefill(
        &self,
        request: &mut RequestKv,
        first_token: Option<u32>,
    ) -> Result<usize> {
        if let Some(first_token) = first_token {
            request.apply_prefill(first_token, self.kv.pool())
        } else {
            request.apply_prefill_chunk(self.kv.pool())
        }?;
        // Retain registered KV only when prefix caching is enabled.
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        let boundary = request.kv_position();
        Ok(boundary)
    }

    /// Reserve a recurrent-state slot for an eligible applied boundary.
    ///
    /// Returns a reservation when the boundary is eligible and a free or
    /// unpinned slot is available. The caller copies recurrent state into that
    /// slot on every rank, then publishes or aborts the reservation. Duplicate
    /// boundaries, disabled caching, ineligible boundaries, and fully pinned
    /// capacity return `None`.
    pub(crate) fn reserve_snapshot(
        &mut self,
        request: &RequestKv,
        boundary: usize,
    ) -> Result<Option<SnapshotReservation>> {
        if !self.enabled() {
            return Ok(None);
        }
        if boundary == 0 || !boundary.is_multiple_of(SNAPSHOT_STRIDE_TOKENS) {
            return Ok(None);
        }
        let sequence_hash = request
            .registered_boundary_hash(boundary)
            .ok_or_else(|| anyhow::anyhow!("no registered KV hash at boundary {boundary}"))?;
        let key = PrefixBoundaryKey {
            sequence_hash,
            boundary_tokens: boundary,
        };
        self.snapshots.reserve(key)
    }

    /// Publish after rank-local snapshot copies succeed. `reserve_snapshot`
    /// already validated registered KV at the boundary.
    pub(crate) fn publish_snapshot(&mut self, reservation: SnapshotReservation) {
        self.stats.inserts += 1;
        if reservation.was_replacement() {
            self.stats.evictions += 1;
        }
        self.snapshots.publish(reservation);
    }

    /// Copy the request's recurrent/conv state into the store and publish the
    /// reservation. Single-GPU callers use this instead of a separate
    /// `RecurrentStateStore`.
    ///
    /// TP keeps the worker-local save plus the controller-side
    /// `publish_snapshot` because the copy crosses the worker/controller
    /// boundary. Step 2 should route both shapes through this store rather
    /// than growing a third copy of the transaction.
    pub(crate) fn save_and_publish_snapshot(
        &mut self,
        ctx: &DeviceContext,
        reservation: SnapshotReservation,
        src: &RecurrentState,
    ) -> Result<()> {
        if let Err(error) = self.snapshots.save(ctx, reservation, src) {
            self.snapshots.abort(reservation);
            return Err(error);
        }
        self.publish_snapshot(reservation);
        Ok(())
    }

    /// Copy a pinned snapshot into a request's recurrent/conv state.
    pub(crate) fn restore_snapshot(
        &self,
        ctx: &DeviceContext,
        guard: &SnapshotGuard,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        self.snapshots.restore_into(ctx, guard, dst)
    }

    /// Abort a prefix reservation after any rank-local copy failure.
    pub(crate) fn abort_snapshot(&mut self, reservation: SnapshotReservation) {
        self.snapshots.abort(reservation);
    }

    /// Reserve KV capacity for the next one-token decode forward.
    pub(crate) fn schedule_decode(&self, request: &mut RequestKv) -> Result<()> {
        request
            .schedule_decode(self.kv.pool())
            .map_err(|e| anyhow::anyhow!("Qwen3.5 decode KV schedule failed: {e}"))
    }

    /// Build the exact, immutable KV page-table view for decode kernels.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn decode_view(&self, request: &RequestKv) -> KvView {
        request.decode_view()
    }

    /// Apply the KV written by decode and record the newly sampled token.
    pub(crate) fn apply_decode(&self, request: &mut RequestKv, token: u32) -> Result<()> {
        request.apply_decode(token, self.kv.pool())?;
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        Ok(())
    }

    /// Roll back pages reserved by a scheduled step that did not apply.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn revert_schedule(&self, request: &mut RequestKv) -> Result<()> {
        request.revert_schedule()
    }

    /// Best-effort rollback for every request scheduled by one failed step.
    pub(crate) fn revert_scheduled_requests<'a>(
        &self,
        requests: impl IntoIterator<Item = &'a mut RequestKv>,
    ) {
        for request in requests {
            if let Err(error) = self.revert_schedule(request) {
                log::warn!("failed to revert KV schedule: {error}");
            }
        }
    }

    /// Release all request KV.
    pub(crate) fn release_request(&self, request: &mut RequestKv) -> Result<()> {
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        request.release()
    }
}

/// Yield reusable snapshot boundaries from longest to shortest.
fn eligible_boundaries(resident_tokens: usize, stride: usize) -> impl Iterator<Item = usize> {
    let highest = resident_tokens / stride * stride;
    (1..=highest / stride).rev().map(move |n| n * stride)
}

#[cfg(test)]
mod tests {
    use super::Config35;
    use super::DeviceContext;
    use super::LocalGeometry;
    use super::PrefixBoundaryKey;
    use super::RecurrentSnapshotStore;
    use super::RecurrentState;
    use super::SnapshotTier;
    use super::eligible_boundaries;

    fn key(tag: u8) -> PrefixBoundaryKey {
        PrefixBoundaryKey {
            sequence_hash: [tag; 16],
            boundary_tokens: 256,
        }
    }

    #[test]
    fn joint_boundaries_descend_on_snapshot_stride() {
        assert_eq!(
            eligible_boundaries(255, 256).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(eligible_boundaries(256, 256).collect::<Vec<_>>(), [256]);
        assert_eq!(
            eligible_boundaries(900, 256).collect::<Vec<_>>(),
            [768, 512, 256]
        );
    }

    #[test]
    fn state_publishes_only_after_explicit_commit() {
        let mut state = RecurrentSnapshotStore::new_directory(1);
        let reservation = state
            .reserve(key(1))
            .expect("reserve failed")
            .expect("empty state must reserve a write");
        assert!(state.lookup(key(1)).is_none());
        state.publish(reservation);
        assert!(state.lookup(key(1)).is_some());
    }

    #[test]
    fn aborted_write_does_not_expose_partial_snapshot() {
        let mut state = RecurrentSnapshotStore::new_directory(1);
        let reservation = state
            .reserve(key(1))
            .expect("reserve failed")
            .expect("empty state must reserve a write");
        state.abort(reservation);
        assert!(state.lookup(key(1)).is_none());
        assert!(state.reserve(key(2)).expect("reserve").is_some());
    }

    #[test]
    fn duplicate_reservation_refreshes_lru() {
        let mut state = RecurrentSnapshotStore::new_directory(2);
        for tag in [1, 2] {
            let reservation = state
                .reserve(key(tag))
                .expect("reserve failed")
                .expect("state should have a free slot");
            state.publish(reservation);
        }
        assert!(state.reserve(key(1)).expect("reserve").is_none());
        let reservation = state
            .reserve(key(3))
            .expect("reserve failed")
            .expect("an unpinned LRU victim should be available");
        state.publish(reservation);
        assert!(state.lookup(key(1)).is_some());
        assert!(state.lookup(key(2)).is_none());
        assert!(state.lookup(key(3)).is_some());
    }

    #[test]
    fn lru_evicts_untouched_snapshot_but_preserves_touched_snapshot() {
        let mut state = RecurrentSnapshotStore::new_directory(2);
        for tag in [1, 2] {
            let reservation = state
                .reserve(key(tag))
                .expect("reserve failed")
                .expect("state should have a free slot");
            state.publish(reservation);
        }
        drop(state.lookup(key(1)).expect("key 1 should be present"));
        let reservation = state
            .reserve(key(3))
            .expect("reserve failed")
            .expect("an unpinned LRU victim should be available");
        state.publish(reservation);
        assert!(state.lookup(key(1)).is_some());
        assert!(state.lookup(key(2)).is_none());
        assert!(state.lookup(key(3)).is_some());
    }

    #[test]
    fn all_pinned_snapshots_make_insertion_a_soft_miss() {
        let mut state = RecurrentSnapshotStore::new_directory(1);
        let reservation = state
            .reserve(key(1))
            .expect("reserve failed")
            .expect("empty state must reserve a write");
        state.publish(reservation);
        let guard = state.lookup(key(1)).expect("key 1 should be present");
        assert!(state.reserve(key(2)).expect("reserve").is_none());
        drop(guard);
        assert!(state.reserve(key(2)).expect("reserve").is_some());
    }

    #[test]
    #[ignore = "requires a CUDA device and a Qwen3.5 model directory (PEGAINFER_TEST_MODEL_PATH)"]
    fn gpu_snapshot_store_round_trips_recurrent_state() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "gpu_snapshot_store_round_trips_recurrent_state",
        ) else {
            return;
        };
        let ctx = DeviceContext::new_with_device(0).expect("create CUDA context");
        let config = Config35::from_file(&model_path).expect("load Qwen3.5 config");
        let geometry =
            LocalGeometry::try_new(&config, Default::default()).expect("derive local geometry");

        let mut store =
            RecurrentSnapshotStore::new_gpu(&ctx, &config, geometry, 1, 0).expect("create store");
        let mut src = RecurrentState::new(&ctx, &config, geometry).expect("allocate source");
        src.seq_len = 256;

        let state_len = geometry.local_linear_num_value_heads()
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_len = geometry.local_linear_qkv_dim() * (config.linear_conv_kernel_dim - 1);
        let host_state: Vec<f32> = (0..state_len).map(|i| i as f32 * 0.25 + 1.0).collect();
        let host_conv: Vec<half::bf16> = (0..conv_len)
            .map(|i| half::bf16::from_f32(i as f32 * 0.5 - 3.0))
            .collect();
        ctx.stream
            .memcpy_htod(&host_state, &mut src.layers[0].state)
            .expect("fill recurrent state");
        ctx.stream
            .memcpy_htod(&host_conv, &mut src.layers[0].conv_state.data)
            .expect("fill conv state");

        let key = PrefixBoundaryKey {
            sequence_hash: [7; 16],
            boundary_tokens: 256,
        };
        let reservation = store
            .reserve(key)
            .expect("reserve failed")
            .expect("empty store must reserve");
        store.save(&ctx, reservation, &src).expect("save snapshot");
        store.publish(reservation);
        assert!(store.contains(key));

        let guard = store.lookup(key).expect("published key must be found");
        let mut restored = RecurrentState::new(&ctx, &config, geometry).expect("allocate restore");
        store
            .restore_into(&ctx, &guard, &mut restored)
            .expect("restore snapshot");
        assert_eq!(restored.seq_len, src.seq_len);
        let restored_state = ctx
            .stream
            .clone_dtoh(&restored.layers[0].state)
            .expect("read restored recurrent state");
        let restored_conv = ctx
            .stream
            .clone_dtoh(&restored.layers[0].conv_state.data)
            .expect("read restored conv state");
        assert_eq!(restored_state, host_state);
        assert_eq!(restored_conv, host_conv);
    }

    fn fill_first_layer(
        ctx: &DeviceContext,
        state: &mut RecurrentState,
        state_len: usize,
        conv_len: usize,
        base: f32,
    ) -> (Vec<f32>, Vec<half::bf16>) {
        let host_state: Vec<f32> = (0..state_len).map(|i| i as f32 * 0.25 + base).collect();
        let host_conv: Vec<half::bf16> = (0..conv_len)
            .map(|i| half::bf16::from_f32(i as f32 * 0.5 - base))
            .collect();
        ctx.stream
            .memcpy_htod(&host_state, &mut state.layers[0].state)
            .expect("fill recurrent state");
        ctx.stream
            .memcpy_htod(&host_conv, &mut state.layers[0].conv_state.data)
            .expect("fill conv state");
        (host_state, host_conv)
    }

    #[test]
    #[ignore = "requires a CUDA device and a Qwen3.5 model directory (PEGAINFER_TEST_MODEL_PATH)"]
    fn gpu_snapshot_store_spills_to_pinned_host() {
        let Some(model_path) =
            crate::test_fixture::model_path_or_skip("gpu_snapshot_store_spills_to_pinned_host")
        else {
            return;
        };
        let ctx = DeviceContext::new_with_device(0).expect("create CUDA context");
        let config = Config35::from_file(&model_path).expect("load Qwen3.5 config");
        let geometry =
            LocalGeometry::try_new(&config, Default::default()).expect("derive local geometry");
        let state_len = geometry.local_linear_num_value_heads()
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_len = geometry.local_linear_qkv_dim() * (config.linear_conv_kernel_dim - 1);

        let mut store = RecurrentSnapshotStore::new_gpu(&ctx, &config, geometry, 1, 1)
            .expect("create store with one GPU and one host slot");
        let mut src_a = RecurrentState::new(&ctx, &config, geometry).expect("allocate A");
        src_a.seq_len = 256;
        let expected_a = fill_first_layer(&ctx, &mut src_a, state_len, conv_len, 1.0);
        let mut src_b = RecurrentState::new(&ctx, &config, geometry).expect("allocate B");
        src_b.seq_len = 256;
        let expected_b = fill_first_layer(&ctx, &mut src_b, state_len, conv_len, 7.0);

        let key_a = PrefixBoundaryKey {
            sequence_hash: [1; 16],
            boundary_tokens: 256,
        };
        let key_b = PrefixBoundaryKey {
            sequence_hash: [2; 16],
            boundary_tokens: 256,
        };

        let reservation_a = store
            .reserve(key_a)
            .expect("reserve A")
            .expect("A gets the free GPU slot");
        store.save(&ctx, reservation_a, &src_a).expect("save A");
        store.publish(reservation_a);

        // No guard is held, so reserving B spills A to the pinned host slab and
        // reuses A's GPU slot.
        let reservation_b = store
            .reserve(key_b)
            .expect("reserve B")
            .expect("B reuses A's slot after the spill");
        store.save(&ctx, reservation_b, &src_b).expect("save B");
        store.publish(reservation_b);

        let guard_a = store.lookup(key_a).expect("A must survive on host");
        assert_eq!(guard_a.tier(), SnapshotTier::Host);
        let guard_b = store.lookup(key_b).expect("B must be on GPU");
        assert_eq!(guard_b.tier(), SnapshotTier::Gpu);
        assert_eq!(store.len(), 2);

        let mut restored_a =
            RecurrentState::new(&ctx, &config, geometry).expect("allocate restore A");
        store
            .restore_into(&ctx, &guard_a, &mut restored_a)
            .expect("restore A from host");
        let mut restored_b =
            RecurrentState::new(&ctx, &config, geometry).expect("allocate restore B");
        store
            .restore_into(&ctx, &guard_b, &mut restored_b)
            .expect("restore B from GPU");

        assert_eq!(restored_a.seq_len, 256);
        assert_eq!(restored_b.seq_len, 256);
        assert_eq!(
            ctx.stream
                .clone_dtoh(&restored_a.layers[0].state)
                .expect("read A state"),
            expected_a.0
        );
        assert_eq!(
            ctx.stream
                .clone_dtoh(&restored_a.layers[0].conv_state.data)
                .expect("read A conv"),
            expected_a.1
        );
        assert_eq!(
            ctx.stream
                .clone_dtoh(&restored_b.layers[0].state)
                .expect("read B state"),
            expected_b.0
        );
        assert_eq!(
            ctx.stream
                .clone_dtoh(&restored_b.layers[0].conv_state.data)
                .expect("read B conv"),
            expected_b.1
        );
    }
}
