"""MLX model runner for Apple Silicon.

When radix cache is enabled (the default), KV data lives in a flat pool
indexed by a radix trie for prefix sharing.  When
``disable_radix_cache=True``, the pool and trie are not allocated, and
each request uses its own contiguous cache.
"""

import logging
import time

import mlx.core as mx
import psutil
from mlx_lm import load as mlx_lm_load

from sglang.srt.hardware_backend.mlx.kv_cache import (
    BatchedDecodeContext,
    ContiguousKVCache,
    MLXAttentionWrapper,
    OffsetCache,
    PoolBackedCache,
    clear_context,
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.hardware_backend.mlx.kv_cache.radix_trie import MlxRadixTrie

logger = logging.getLogger(__name__)


class MlxModelRunner:
    """MLX model runner with optional radix-cache prefix sharing."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static

        self._load_model()

        # Pin MLX allocations to prevent OS paging
        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Wired memory limit set to {max_wired / (1024**3):.1f} GB")

        patch_model_attention(self.model)

        self._num_layers = get_num_layers(self.model)
        self._max_seq_len = 4096  # doubles on overflow

        self._req_caches: dict[str, list[ContiguousKVCache | PoolBackedCache]] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._cache_pool: list[list[ContiguousKVCache]] = []  # reusable caches

        # Radix cache state
        self._kv_pool: MlxKVPool | None = None
        self._radix_trie: MlxRadixTrie | None = None
        self._req_slot_ids: dict[str, list[int]] = {}
        self._pool_dirty: set[str] = set()  # reqs needing pool sync
        self._req_last_node: dict[str, object | None] = {}
        self._req_prefix_len: dict[str, int] = {}

        if not self.disable_radix_cache:
            self._init_radix_cache(pool_size)

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _acquire_cache(self) -> list[ContiguousKVCache]:
        """Get a reusable cache list from the pool, or create a new one."""
        if self._cache_pool:
            cache = self._cache_pool.pop()
            for c in cache:
                c.offset = 0
            return cache
        return [
            ContiguousKVCache(max_seq_len=self._max_seq_len)
            for _ in range(self._num_layers)
        ]

    def _release_cache(self, cache: list[ContiguousKVCache]) -> None:
        """Return a cache list to the pool for reuse."""
        self._cache_pool.append(cache)

    @staticmethod
    def _eval_with_cache(
        token_result: mx.array, cache: list[ContiguousKVCache | PoolBackedCache]
    ) -> None:
        """Evaluate token result and all cache buffers in one mx.eval call."""
        mx.eval(token_result, *[s for c in cache for s in c.state])

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )
        # Force-evaluate weights so mx.get_active_memory() reflects
        # actual usage before KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def _init_radix_cache(self, pool_size: int | None) -> None:
        """Initialize pool and trie.  Auto-sizes from available memory if needed."""
        num_layers = self._num_layers

        layer_list, attn_attr = find_attention_layers(self.model)
        if not layer_list:
            raise RuntimeError("Cannot init radix cache: no attention layers found")

        sample_block = layer_list[0]
        sample_attn = getattr(sample_block, attn_attr)
        if isinstance(sample_attn, MLXAttentionWrapper):
            sample_attn = sample_attn._inner

        n_kv_heads = sample_attn.n_kv_heads

        if hasattr(sample_attn, "head_dim"):
            head_dim = sample_attn.head_dim
        elif hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            # k_proj.weight shape: (n_kv_heads * head_dim, hidden_size)
            head_dim = sample_attn.k_proj.weight.shape[0] // n_kv_heads
        else:
            raise RuntimeError("Cannot determine head_dim from attention module")

        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype

        if pool_size is None:
            pool_size = self._profile_pool_size(num_layers, n_kv_heads, head_dim, dtype)

        self._kv_pool = MlxKVPool(
            pool_size=pool_size,
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self._radix_trie = MlxRadixTrie(pool_capacity=pool_size)
        logger.info(
            f"KV pool initialized: pool_size={pool_size}, "
            f"{num_layers} layers, {n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def _profile_pool_size(
        self,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
    ) -> int:
        """Derive KV pool slot count from available memory."""
        vm = psutil.virtual_memory()
        metal_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        mlx_used = mx.get_active_memory()

        usable = min(int(vm.total * self._mem_fraction_static), metal_limit)
        kv_budget = min(
            max(usable - mlx_used, 0),
            int(vm.available * self._mem_fraction_static),
        )

        bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * dtype.size
        pool_size = max(kv_budget // bytes_per_slot, 256)
        logger.info(
            f"Auto-sized KV pool: total_ram={vm.total / (1024**3):.1f} GB, "
            f"sys_available={vm.available / (1024**3):.2f} GB, "
            f"metal_limit={metal_limit / (1024**3):.1f} GB, "
            f"mlx_used={mlx_used / (1024**3):.2f} GB, "
            f"kv_budget={kv_budget / (1024**3):.2f} GB, "
            f"bytes_per_slot={bytes_per_slot}, pool_size={pool_size}"
        )
        return pool_size

    def prefill(
        self,
        req_id: str,
        token_ids: list[int],
    ) -> int:
        """Prefill a request.  Returns ``(next_token_id, prefix_len)``."""
        num_layers = self._num_layers
        num_tokens = len(token_ids)

        if self.disable_radix_cache:
            cache = self._acquire_cache()
            input_ids = mx.array([token_ids], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            next_token_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            self._eval_with_cache(next_token_mlx, cache)
            next_token = int(next_token_mlx.item())

            self._req_token_ids[req_id] = list(token_ids) + [next_token]
            self._req_caches[req_id] = cache
            self._req_slot_ids[req_id] = []
            return next_token, 0

        assert self._kv_pool is not None and self._radix_trie is not None

        match = self._radix_trie.match_prefix(token_ids)
        prefix_len = match.prefix_len
        matched_node = match.last_node

        if prefix_len > 0:
            self._radix_trie.inc_ref(matched_node)

        new_token_count = num_tokens - prefix_len
        if new_token_count > 0:
            new_slots = self._kv_pool.allocator.alloc(new_token_count)
            if new_slots is None:
                freed = self._radix_trie.evict(new_token_count)
                if freed:
                    self._kv_pool.allocator.free(freed)
                new_slots = self._kv_pool.allocator.alloc(new_token_count)
                if new_slots is None:
                    if prefix_len > 0:
                        self._radix_trie.dec_ref(matched_node)
                    raise RuntimeError(
                        f"KV pool exhausted: need {new_token_count} slots, "
                        f"only {self._kv_pool.allocator.available} available"
                    )
        else:
            new_slots = []

        if prefix_len > 0:
            all_slots = match.slot_ids + new_slots
        else:
            all_slots = new_slots

        if prefix_len > 0:
            self._flush_pending_syncs()
            slot_ids_mx = mx.array(match.slot_ids, dtype=mx.int32)
            cache = [
                PoolBackedCache(self._kv_pool, i, slot_ids_mx, prefix_len)
                for i in range(num_layers)
            ]

            if new_token_count > 0:
                logger.info(
                    f"Prefix reuse: {prefix_len}/{num_tokens} tokens cached, "
                    f"computing {new_token_count} new tokens"
                )
        else:
            cache = self._acquire_cache()

        if new_token_count > 0:
            extend_tokens = token_ids[prefix_len:]
        else:
            extend_tokens = token_ids[-1:]
            for c in cache:
                c.offset = max(c.offset - 1, 0)

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        # Convert PoolBackedCache → ContiguousKVCache for decode
        if prefix_len > 0:
            contiguous_cache = self._acquire_cache()
            for layer_idx in range(num_layers):
                pbc = cache[layer_idx]
                contiguous_cache[layer_idx].update_and_fetch(
                    pbc._full_keys, pbc._full_values
                )
            cache = contiguous_cache

        self._eval_with_cache(next_token_mlx, cache)

        next_token = int(next_token_mlx.item())

        self._radix_trie.insert(token_ids, all_slots)
        self._req_slot_ids[req_id] = all_slots
        self._req_token_ids[req_id] = list(token_ids) + [next_token]
        self._req_caches[req_id] = cache
        self._pool_dirty.add(req_id)
        self._req_last_node[req_id] = matched_node if prefix_len > 0 else None
        self._req_prefix_len[req_id] = prefix_len

        return next_token, prefix_len

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
    ) -> int:
        """Continue prefill for a chunked request.

        Returns:
            Next token ID (greedy sampled).
        """
        assert req_id in self._req_caches, f"extend called for unknown request {req_id}"

        cache = self._req_caches[req_id]
        num_new = len(new_token_ids)

        if not self.disable_radix_cache:
            new_slots = self._kv_pool.allocator.alloc(num_new)
            if new_slots is None:
                freed = self._radix_trie.evict(num_new)
                if freed:
                    self._kv_pool.allocator.free(freed)
                new_slots = self._kv_pool.allocator.alloc(num_new)
                if new_slots is None:
                    raise RuntimeError(
                        f"KV pool exhausted: need {num_new} slots, "
                        f"only {self._kv_pool.allocator.available} available"
                    )

        input_ids = mx.array([new_token_ids], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)
        self._eval_with_cache(next_token_mlx, cache)
        next_token = int(next_token_mlx.item())

        prev_tokens = self._req_token_ids[req_id]
        if prev_tokens:
            prev_tokens.pop()  # remove stale intermediate token
        prev_tokens.extend(new_token_ids)
        prev_tokens.append(next_token)

        if not self.disable_radix_cache:
            self._req_slot_ids[req_id].extend(new_slots)
            self._pool_dirty.add(req_id)
            full_prompt = prev_tokens[:-1]
            self._radix_trie.insert(full_prompt, self._req_slot_ids[req_id])

        logger.info(
            f"Extend req {req_id}: +{num_new} tokens, "
            f"total cache offset {cache[0].offset}"
        )

        return next_token

    def _flush_pending_syncs(self) -> None:
        """Write pending KV to pool before a pool read."""
        if not self._pool_dirty:
            return
        for rid in list(self._pool_dirty):
            self._sync_request_to_pool(rid)

    def _sync_request_to_pool(self, req_id: str) -> None:
        """Copy new KV from contiguous cache into the pool (no-op for PoolBackedCache)."""
        if req_id not in self._pool_dirty:
            return
        cache = self._req_caches.get(req_id)
        if cache is None or (cache and isinstance(cache[0], PoolBackedCache)):
            self._pool_dirty.discard(req_id)
            return
        slots = self._req_slot_ids.get(req_id)
        if slots is None:
            self._pool_dirty.discard(req_id)
            return

        num_layers = len(cache)
        num_prefill_slots = len(slots)
        prefill_len = min(num_prefill_slots, cache[0].offset)

        cached_prefix_len = self._req_prefix_len.get(req_id, 0)
        sync_start = cached_prefix_len
        sync_len = prefill_len - sync_start
        if sync_len > 0:
            sync_slots_mx = mx.array(slots[sync_start:prefill_len], dtype=mx.int32)
            # Transpose cache (n_kv_heads, S, head_dim) → pool (S, n_kv_heads, head_dim)
            k_all = mx.stack(
                [
                    cache[i].keys[0, :, sync_start:prefill_len, :].transpose(1, 0, 2)
                    for i in range(num_layers)
                ]
            )
            v_all = mx.stack(
                [
                    cache[i].values[0, :, sync_start:prefill_len, :].transpose(1, 0, 2)
                    for i in range(num_layers)
                ]
            )
            self._kv_pool.set_kv_all_layers(sync_slots_mx, k_all, v_all)

        self._pool_dirty.discard(req_id)

    def prefill_batch(
        self,
        req_ids: list[str],
        token_ids_list: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Prefill multiple requests serially (BS=1 per forward)."""
        return [self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)]

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Decode one token per request (pool sync deferred)."""
        batch_size = len(req_ids)
        num_layers = self._num_layers

        caches = [self._req_caches[rid] for rid in req_ids]
        seq_lens = [caches[i][0].offset for i in range(batch_size)]

        if batch_size == 1:
            cache = caches[0]
            last_token = self._req_token_ids[req_ids[0]][-1]
            input_ids = mx.array([[last_token]], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            self._eval_with_cache(next_tokens_mlx, cache)
        else:
            layer_caches = [
                [caches[i][layer_idx] for i in range(batch_size)]
                for layer_idx in range(num_layers)
            ]
            ctx = BatchedDecodeContext(
                batch_size=batch_size,
                seq_lens=seq_lens,
                layer_caches=layer_caches,
            )
            set_context(ctx)
            try:
                max_offset = max(seq_lens)
                shim_cache = [OffsetCache(offset=max_offset) for _ in range(num_layers)]
                last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
                batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
                model_output = self.model(batched_input, cache=shim_cache)
                logits = self._extract_logits(model_output)
                next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)

                eval_targets = [next_tokens_mlx]
                for c_list in caches:
                    for c in c_list:
                        eval_targets.append(c.keys)
                        eval_targets.append(c.values)
                mx.eval(*eval_targets)
            finally:
                clear_context()

        next_tokens = next_tokens_mlx.tolist()

        for i, rid in enumerate(req_ids):
            self._req_token_ids[rid].append(next_tokens[i])

        return next_tokens

    def has_request(self, req_id: str) -> bool:
        """Check if a request has active state."""
        return req_id in self._req_slot_ids or req_id in self._req_caches

    def remove_request(self, req_id: str):
        """Clean up state for a completed request."""
        if not self.disable_radix_cache:
            self._sync_request_to_pool(req_id)
            last_node = self._req_last_node.pop(req_id, None)
            if last_node is not None:
                self._radix_trie.dec_ref(last_node)

        self._req_slot_ids.pop(req_id, None)
        self._req_token_ids.pop(req_id, None)
        cache = self._req_caches.pop(req_id, None)
        if cache is not None:
            self._release_cache(cache)
        self._pool_dirty.discard(req_id)
        self._req_prefix_len.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._req_slot_ids.clear()
        self._req_token_ids.clear()
        for cache in self._req_caches.values():
            self._cache_pool.append(cache)
        self._req_caches.clear()
        self._pool_dirty.clear()
        self._req_last_node.clear()
        self._req_prefix_len.clear()
        if self._radix_trie is not None:
            freed = self._radix_trie.reset()
            if freed and self._kv_pool is not None:
                self._kv_pool.allocator.free(freed)
        if self._kv_pool is not None:
            self._kv_pool.clear()
