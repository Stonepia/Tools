

# Original file: ./maml__23_forward_69.4/maml__23_forward_69.4.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/gt/cgtcvbqackrxvhvjhf35u5ftrws2eicgjvx4cmiak3jttt6n5yhe.py
# Source Nodes: [batch_norm, batch_norm_1, conv2d, conv2d_1, relu, relu_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution, aten.relu]
# batch_norm => add, add_3, mul, mul_6, rsqrt, sub, var_mean
# batch_norm_1 => add_5, add_6, mul_10, mul_11, mul_12, mul_8, mul_9, var_mean_1
# conv2d => convolution
# conv2d_1 => convolution_1
# relu => relu
# relu_1 => relu_1
triton_red_fused__native_batch_norm_legit_functional_convolution_relu_3 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_convolution_relu_3', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_convolution_relu_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_convolution_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2700
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 36
        r2 = (rindex // 36)
        tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (2304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = triton_helpers.maximum(0, tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask & xmask, tmp5_weight_next, tmp5_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = 0.1
    tmp9 = tmp5 * tmp8
    tmp11 = 0.9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2700.0
    tmp15 = tmp6 / tmp14
    tmp16 = 1.0003705075954057
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 * tmp8
    tmp20 = tmp19 * tmp11
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')
