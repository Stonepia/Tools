

# Original file: ./llama___60.0/llama___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/yi/cyinatjjbgixns6dw5fu6awdyzzugbi6sdgnd37k6hwnj6rrm33t.py
# Source Nodes: [add_1, add_2, l__self___layers_0_feed_forward_w1, l__self___tok_embeddings, mean_1, mul_4, mul_5, pow_2, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_1 => add_1
# add_2 => add_2
# l__self___layers_0_feed_forward_w1 => convert_element_type_13
# l__self___tok_embeddings => embedding
# mean_1 => mean_1
# mul_4 => mul_4
# mul_5 => mul_5
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7', '''
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
    meta={'signature': {0: '*i32', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32000, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 32000)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32000")
    tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask & xmask, other=0.0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = 512.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp5 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp19, rmask & xmask)
''')
