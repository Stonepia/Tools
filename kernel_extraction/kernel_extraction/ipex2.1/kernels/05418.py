

# Original file: ./llama___60.0/llama___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/yv/cyvozxaegyhyot4fv4or35zpwxcau67x7tzbu4o2pnawaxdzbebj.py
# Source Nodes: [add_1, add_3, add_4, l__mod___tok_embeddings, mean_2, mul_7, mul_8, pow_3, rsqrt_2], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_1 => add_1
# add_3 => add_3
# add_4 => add_4
# l__mod___tok_embeddings => embedding
# mean_2 => mean_2
# mul_7 => mul_8
# mul_8 => mul_9
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9', '''
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32000, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 32000)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32000")
    tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask & xmask, other=0.0)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = 512.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp6 * tmp16
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp19, rmask & xmask)
''')
