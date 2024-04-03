

# Original file: ./maml__29_forward_88.14/maml__29_forward_88.14.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/zt/cztyse2y5tdmt2ntryusx7xcttbtw57ttdglpx7uuz2nk2koy5rj.py
# Source Nodes: [batch_norm_2, relu_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# batch_norm_2 => add_10, add_8, add_9, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
# relu_2 => relu_2
triton_red_fused__native_batch_norm_legit_functional_relu_8 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_relu_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_relu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_relu_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4
        r2 = (rindex // 4)
        tmp0 = tl.load(in_ptr0 + (r1 + (4*x0) + (256*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = triton_helpers.maximum(0, tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight,
        )
        tmp3_mean = tl.where(rmask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask & xmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = 4*ks0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.0
    tmp9 = tmp7 - tmp8
    tmp10 = tmp4 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 4*ks0*(1/((-1) + (4*ks0)))
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp10 * tmp15
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp3 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
''')
