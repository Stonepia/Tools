

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/w2/cw2lwezvevpolnp6nis3k2prwdtaaqmd5y6j5hpuxktv5htttgyb.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, conv2d, conv2d_1, conv2d_2, mul_9, relu, relu_1, relu_2, sub_9], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.mul, aten.relu, aten.sub]
# batch_norm => add, add_3, convert_element_type_3, convert_element_type_4, mul_18, mul_24, rsqrt, sub_18, var_mean
# batch_norm_1 => add_4, add_7, convert_element_type_7, convert_element_type_8, mul_25, mul_31, rsqrt_1, sub_19, var_mean_1
# batch_norm_2 => add_10, add_9, convert_element_type_11, mul_33, mul_34, mul_35, mul_36, mul_37, var_mean_2
# conv2d => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# conv2d_1 => convert_element_type_5, convert_element_type_6, convolution_1
# conv2d_2 => convert_element_type_10, convert_element_type_9, convolution_2
# mul_9 => mul_9
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
# sub_9 => sub_9
triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_10 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_10', '''
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
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 300
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex % 4
    r2 = (rindex // 4)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r1 + (4*x0) + (256*r2)), rmask & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.4
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp5 = tmp0.to(tl.float32)
    tmp7 = tmp6 + tmp5
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 300, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = 0.1
    tmp27 = tmp19 * tmp26
    tmp29 = 0.9
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 + tmp30
    tmp32 = 300.0
    tmp33 = tmp25 / tmp32
    tmp34 = 1.0033444816053512
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp26
    tmp38 = tmp37 * tmp29
    tmp39 = tmp36 + tmp38
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tl.store(out_ptr4 + (x0), tmp31, xmask)
    tl.store(out_ptr5 + (x0), tmp39, xmask)
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr3 + (x0), tmp25, xmask)
''')
