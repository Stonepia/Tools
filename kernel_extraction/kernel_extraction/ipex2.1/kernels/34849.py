

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/iq/ciqxxlisu7m4afwvyigt5qa7mi6ju3qvdqvb6s4tk574qjyuqfqx.py
# Source Nodes: [abs_10, add_39, add_40, min_8, mul_100, mul_95, mul_99, neg_16, neg_17, setitem_27, sub_16, tanh_7, tensor, truediv_21, truediv_22], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.sub, aten.tanh]
# abs_10 => abs_10
# add_39 => add_46
# add_40 => add_47
# min_8 => minimum_7
# mul_100 => mul_100
# mul_95 => mul_95
# mul_99 => mul_99
# neg_16 => neg_16
# neg_17 => neg_17
# setitem_27 => copy_27, select_scatter_16
# sub_16 => sub_16
# tanh_7 => tanh_7
# tensor => full_default_1
# truediv_21 => div_21
# truediv_22 => div_22
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_26 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2090400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x3 = (xindex // 2)
    x1 = (xindex // 2) % 5226
    x2 = (xindex // 10452)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (10634 + x1 + (5304*x2)), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (42538 + x0 + (4*x1) + (21216*x2)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = -tmp3
    tmp6 = 0.0
    tmp7 = triton_helpers.minimum(tmp6, tmp5)
    tmp8 = 1e-20
    tmp9 = tmp7 - tmp8
    tmp10 = tmp4 / tmp9
    tmp11 = tl.abs(tmp10)
    tmp12 = -tmp11
    tmp13 = 0.001
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 / tmp13
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20 * tmp10
    tmp23 = tmp21 * tmp22
    tmp24 = 2 + x2
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp24 >= tmp25
    tmp27 = tl.full([1], 202, tl.int64)
    tmp28 = tmp24 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr3 + (106 + x0 + (4*x1) + (21216*x2)), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp29, tmp30, 0.0)
    tmp32 = tl.load(in_ptr4 + (106 + x0 + (4*x1) + (21216*x2)), tmp29 & xmask, other=0.0)
    tmp33 = tl.where(tmp29, tmp32, 0.0)
    tmp34 = tl.load(in_ptr5 + (106 + x0 + (4*x1) + (21216*x2)), tmp29 & xmask, other=0.0)
    tmp35 = tl.where(tmp29, tmp34, 0.0)
    tmp37 = tl.where(tmp29, tmp35, tmp36)
    tmp38 = tl.where(tmp29, tmp33, tmp37)
    tmp39 = tl.where(tmp29, tmp31, tmp38)
    tmp40 = tl.where(tmp2, tmp23, tmp39)
    tl.store(out_ptr0 + (x5), tmp40, xmask)
''')
