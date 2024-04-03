

# Original file: ./maml__29_forward_88.14/maml__29_forward_88.14_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/6t/c6t6oao3xktdruetepuiqvgwu4rhaws3ca4hgipkwinjk4x2o7uv.py
# Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# batch_norm_1 => add_4, add_5, add_6, convert_element_type_7, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
# relu_1 => relu_1
triton_red_fused__native_batch_norm_legit_functional_relu_9 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_relu_9', '''
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
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_relu_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 36
        r2 = (rindex // 36)
        tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (2304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = triton_helpers.maximum(0, tmp0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = 36*ks0
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 0.0
    tmp10 = tmp8 - tmp9
    tmp11 = tmp5 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = 36*ks0*(1/((-1) + (36*ks0)))
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp11 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp4 * tmp18
    tmp26 = tmp25 * tmp21
    tmp27 = tmp24 + tmp26
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr3 + (x0), tmp23, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
''')
