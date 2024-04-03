

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/3j/c3jfeb5yzlijxizgsfjgx2sfxynorm7dfywckwo74tfjp7dkhh3h.py
# Source Nodes: [add_4, add_6, add_8, add_9, l__mod___model_encoder_block_0_layer_0_dropout, l__mod___model_encoder_block_0_layer__1__dropout, l__mod___model_encoder_block_1_layer_0_dropout, l__mod___model_encoder_embed_tokens, mean_3, mul_10, mul_9, pow_4, rsqrt_3], Original ATen: [aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_4 => add_6
# add_6 => add_8
# add_8 => add_11
# add_9 => add_12
# l__mod___model_encoder_block_0_layer_0_dropout => clone_3
# l__mod___model_encoder_block_0_layer__1__dropout => clone_5
# l__mod___model_encoder_block_1_layer_0_dropout => clone_8
# l__mod___model_encoder_embed_tokens => embedding
# mean_3 => mean_3
# mul_10 => mul_10
# mul_9 => mul_9
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*tmp1)), rmask & xmask, other=0.0)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 1024.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp21, rmask & xmask)
''')
