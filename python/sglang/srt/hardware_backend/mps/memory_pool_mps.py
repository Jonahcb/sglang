from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import torch
from sglang.srt.mem_cache.memory_pool import KVCache

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
)
from sglang.srt.utils import get_bool_env_var

# if TYPE_CHECKING:
#     from sglang.srt.layers.radix_attention import RadixAttention



from sglang.srt.layers.radix_attention import RadixAttention

TORCH_TO_MLX_DTYPE = {
    torch.float32: mx.float32,
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.float64: mx.float64,  # Warning: CPU only in MLX
    torch.int32: mx.int32,
    torch.int64: mx.int64,
    torch.int16: mx.int16,
    torch.int8: mx.int8,
    torch.uint8: mx.uint8,
    torch.bool: mx.bool_
}

# We need to have our own version because store_dtype.itemsize doesn't exist in MLX
def _set_kv_buffer_impl(
    k: mx.array,
    v: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    indices: mx.array,
    # row_dim: int,  # head_num * head_dim
    # store_dtype: torch.dtype,
    # device_module: Any,
    # alt_stream: Optional[torch.cuda.Stream] = None,
    # same_kv_dim: bool = True,
) -> None:

    # from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    # TODO (Jonacb): see if we can use streams here in MLX for performance improvements
    # if get_is_capture_mode() and alt_stream is not None:
    #     current_stream = device_module.current_stream()
    #     alt_stream.wait_stream(current_stream)
    #     k_cache[indices] = k
    #     with device_module.stream(alt_stream):
    #         v_cache[indices] = v
    #     current_stream.wait_stream(alt_stream)
    # else:  # fallback to naive implementation
    k_cache[indices] = k
    v_cache[indices] = v

# We cannot reuse the helper function from memory_pool.py because it is PyTorch specific
def get_tensor_size_bytes(t: Union[mx.array, List[mx.array]]) -> int:
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    
    return t.nbytes

# The reason we inherit directly from KVCache insteado of MHATokenToKVPool, like NPU does, is because many of the implementations in MHATokenToKVPool are not in PyTorch, and we want to use MLX.
class MPSMHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: mx.Dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.store_dtype = TORCH_TO_MLX_DTYPE[dtype]
        self.dtype = TORCH_TO_MLX_DTYPE[dtype] # TODO (Jonahcb): Investigate whether why we set both of these using the same "dtype" parameter
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()

        #self.device_module = torch.get_device_module(self.device)
        # self.alt_stream = (
        #     self.device_module.Stream() if _is_cuda and enable_alt_stream else None
        # )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # TODO (Jonacb): check if we need these then
        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    # TODO (Jonahcb): this is not needed for KVCache so investigate whether it would even be useful for MLX
    #def _init_kv_copy_and_warmup(self):
        # TODO (Jonahcb): in MHATokenToKVPool, this is done using a Triton kernel with grid: (layers, chunks of bytes). We need a custom Metal kernel for this. Right now, we just do it naively in MLX.

        # Heuristics for KV copy tiling
        # _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        # _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        # _KV_COPY_TILE_SIZE_LARGE = 512
        # _KV_COPY_TILE_SIZE_MEDIUM = 256
        # _KV_COPY_TILE_SIZE_SMALL = 128
        # _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        # _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        # stride_bytes = int(self.data_strides[0].item())
        # if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
        #     bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        # elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
        #     bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        # else:
        #     bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        # chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        # self._kv_copy_config = {
        #     "bytes_per_tile": bytes_per_tile,
        #     "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
        #     "num_warps": (
        #         _KV_COPY_NUM_WARPS_SMALL_TILE
        #         if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
        #         else _KV_COPY_NUM_WARPS_LARGE_TILE
        #     ),
        #     "num_locs_upper": chunk_upper,
        # }

        # dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        # grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        
        # copy_all_layer_kv_cache_tiled[grid](
        #     self.data_ptrs,
        #     self.data_strides,
        #     dummy_loc,
        #     dummy_loc,
        #     1,
        #     chunk_upper,
        #     BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
        #     num_warps=self._kv_copy_config["num_warps"],
        #     num_stages=2,
        # )



    def _create_buffers(self):
        # with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        #     with (
        #         torch.cuda.use_mem_pool(self.custom_mem_pool)
        #         if self.enable_custom_mem_pool
        #         else nullcontext()
        #     ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            mx.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                stream=mx.gpu, # TODO (Jonahcb): MLX has a concept of device to determine what device to run operations on but I don't know what happens when we specify the device of a new array. Investigate this.
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            mx.zeros(
                (self.size + self.page_size, self.head_num, self.v_head_dim),
                dtype=self.store_dtype,
                stream=mx.gpu,
            )
            for _ in range(self.layer_num)
        ]

        # TODO (Jonahcb): Investigate what use we would have for this in MLX implementation
        # self.k_data_ptrs = mx.array(
        #     [x.data_pointer() for x in self.k_buffer],
        #     dtype=mx.uint64,
        #     stream=mx.gpu, # TODO (Jonahcb): MLX has a concept of device to determine what device to run operations on but I don't know what happens when we specify the device of a new array. Investigate this.
        # )
        # self.v_data_ptrs = mx.array(
        #     [x.data_pointer() for x in self.v_buffer],
        #     dtype=mx.uint64,
        #     stream=mx.gpu, # TODO (Jonahcb): MLX has a concept of device to determine what device to run operations on but I don't know what happens when we specify the device of a new array. Investigate this.
        # )
        # self.data_ptrs = mx.concatenate([self.k_data_ptrs, self.v_data_ptrs], axis=0)
        # self.data_strides = mx.array(
        #     [
        #         np.prod(x.shape[1:]) * x.dtype.itemsize
        #         for x in self.k_buffer + self.v_buffer
        #     ],
        #     stream=mx.gpu, # TODO (Jonahcb): MLX has a concept of device to determine what device to run operations on but I don't know what happens when we specify the device of a new array. Investigate this.
        # )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    # TODO (Jonahcb): would we even use this function in the MLX backend?
    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    # TODO (Jonahcb): would we even use this function in the MLX backend?
    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: mx.array,
        cache_k: mx.array,
        cache_v: mx.array,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.astype(self.dtype)
            cache_v = cache_v.astype(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)
        
        print(type(loc))

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            # row_dim=self.row_dim,
            # store_dtype=self.store_dtype,
            # device_module=self.device_module,
            # alt_stream=self.alt_stream,
            # same_kv_dim=self.same_kv_dim,
        )

    # def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
    #     if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
    #         move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
    #         return

    #     N = tgt_loc.numel()
    #     if N == 0:
    #         return

    #     assert (
    #         self._kv_copy_config is not None
    #     ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

    #     cfg = self._kv_copy_config
    #     cap = int(cfg.get("num_locs_upper", 256))
    #     grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

    #     if N <= cap:
    #         upper = next_power_of_2(N)
    #         copy_all_layer_kv_cache_tiled[grid](
    #             self.data_ptrs,
    #             self.data_strides,
    #             tgt_loc,
    #             src_loc,
    #             N,
    #             upper,
    #             BYTES_PER_TILE=cfg["bytes_per_tile"],
    #             num_warps=cfg["num_warps"],
    #             num_stages=2,
    #         )
    #         return

    #     # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
    #     for start in range(0, N, cap):
    #         end = min(start + cap, N)
    #         chunk_len = end - start
    #         upper = next_power_of_2(chunk_len)
    #         copy_all_layer_kv_cache_tiled[grid](
    #             self.data_ptrs,
    #             self.data_strides,
    #             tgt_loc[start:end],
    #             src_loc[start:end],
    #             chunk_len,
    #             upper,
    #             BYTES_PER_TILE=cfg["bytes_per_tile"],
    #             num_warps=cfg["num_warps"],
    #             num_stages=2,
    #         )
