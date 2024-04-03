

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/5j/c5jtdjmuge77lvx6igaoh4vqtd72pf64zejlsowlivfafi7utya4.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, batch_norm_3, conv2d, conv2d_1, conv2d_2, conv2d_3, mul_13, relu, relu_1, relu_2, relu_3, sub_13], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.mul, aten.relu, aten.sub]
# batch_norm => add, add_3, convert_element_type, convert_element_type_1, mul_18, mul_24, rsqrt, sub_18, var_mean
# batch_norm_1 => add_4, add_7, convert_element_type_4, convert_element_type_5, mul_25, mul_31, rsqrt_1, sub_19, var_mean_1
# batch_norm_2 => add_11, add_8, convert_element_type_8, convert_element_type_9, mul_32, mul_38, rsqrt_2, sub_20, var_mean_2
# batch_norm_3 => add_13, add_14, convert_element_type_12, convert_element_type_14, convert_element_type_15, mul_40, mul_41, mul_42, mul_43, mul_44, var_mean_3
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# conv2d_3 => convolution_3
# mul_13 => mul_13
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
# relu_3 => relu_3
# sub_13 => sub_13
triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_11 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_11', '''
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
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 75
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp38 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = 0.4
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp5 + tmp0
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 75, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = 0.1
    tmp26 = tmp18 * tmp25
    tmp28 = 0.9
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp26 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 75.0
    tmp34 = tmp24 / tmp33
    tmp35 = 1.0135135135135136
    tmp36 = tmp34 * tmp35
    tmp37 = tmp36 * tmp25
    tmp39 = tmp38 * tmp28
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp37 + tmp40
    tmp42 = tmp41.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
    tl.store(out_ptr4 + (x0), tmp42, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tl.store(out_ptr2 + (x0), tmp24, xmask)
''')
