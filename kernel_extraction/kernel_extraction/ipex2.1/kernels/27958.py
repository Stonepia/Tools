

# Original file: ./maml__23_forward_69.4/maml__23_forward_69.4_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/nf/cnfxsvrmjivdohttwdacater6fpg7vesauwy63rysczkvtzkug3j.py
# Source Nodes: [batch_norm, batch_norm_1, conv2d, conv2d_1, relu, relu_1], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.relu]
# batch_norm => add, add_3, convert_element_type_3, convert_element_type_4, mul, mul_6, rsqrt, sub, var_mean
# batch_norm_1 => add_5, add_6, convert_element_type_7, mul_10, mul_11, mul_12, mul_8, mul_9, var_mean_1
# conv2d => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# conv2d_1 => convert_element_type_5, convert_element_type_6, convolution_1
# relu => relu
# relu_1 => relu_1
triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_7 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_7', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2700
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 36
        r2 = (rindex // 36)
        tmp2 = tl.load(in_ptr1 + (r1 + (36*x0) + (2304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2 + tmp1
        tmp4 = triton_helpers.maximum(0, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight,
        )
        tmp7_mean = tl.where(rmask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask & xmask, tmp7_weight_next, tmp7_weight)
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tl.store(out_ptr2 + (x0), tmp8, xmask)
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = 0.1
    tmp11 = tmp7 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 2700.0
    tmp17 = tmp8 / tmp16
    tmp18 = 1.0003705075954057
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19 * tmp10
    tmp22 = tmp21 * tmp13
    tmp23 = tmp20 + tmp22
    tl.store(out_ptr3 + (x0), tmp15, xmask)
    tl.store(out_ptr4 + (x0), tmp23, xmask)
''')
