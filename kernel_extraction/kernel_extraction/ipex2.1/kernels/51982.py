

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/e2/ce25tmtfs3yxeuja3r2oqysysbshakb75pd6k3gtlnv7exsdhyre.py
# Source Nodes: [batch_norm, batch_norm_1, conv2d, conv2d_1, mul_5, relu, relu_1, sub_5], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.mul, aten.relu, aten.sub]
# batch_norm => add, add_3, convert_element_type, convert_element_type_1, mul_18, mul_24, rsqrt, sub_18, var_mean
# batch_norm_1 => add_5, add_6, convert_element_type_4, convert_element_type_6, convert_element_type_7, mul_26, mul_27, mul_28, mul_29, mul_30, var_mean_1
# conv2d => convolution
# conv2d_1 => convolution_1
# mul_5 => mul_5
# relu => relu
# relu_1 => relu_1
# sub_5 => sub_5
triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_6 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_6', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: '*bf16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2700
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = 0.4
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 36
        r2 = (rindex // 36)
        tmp5 = tl.load(in_ptr2 + (r1 + (36*x0) + (2304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tmp5 + tmp0
        tmp7 = triton_helpers.maximum(0, tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = 0.1
    tmp14 = tmp10 * tmp13
    tmp16 = 0.9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp14 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 2700.0
    tmp22 = tmp11 / tmp21
    tmp23 = 1.0003705075954057
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24 * tmp13
    tmp27 = tmp26 * tmp16
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp25 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')
