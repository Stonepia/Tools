

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jj/cjj7i4rpn257o2mvwxpzt5kvuwacixyrj37xiwpg5gyqyr2lx62l.py
# Source Nodes: [abs_7, add_27, add_28, min_5, mul_65, mul_69, mul_70, neg_10, neg_11, setitem_21, sub_13, tanh_4, tensor, truediv_15, truediv_16], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_7 => abs_7
# add_27 => add_31
# add_28 => add_32
# min_5 => minimum_4
# mul_65 => mul_65
# mul_69 => mul_69
# mul_70 => mul_70
# neg_10 => neg_10
# neg_11 => neg_11
# setitem_21 => copy_21, select_scatter_10, select_scatter_11, slice_scatter_93, slice_scatter_94
# sub_13 => sub_13
# tanh_4 => tanh_4
# tensor => full_default_1
# truediv_15 => div_15
# truediv_16 => div_16
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4243200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 104) % 204
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x4 = (xindex // 21216)
    x6 = (xindex // 4)
    x7 = xindex
    tmp43 = tl.load(in_ptr3 + (42432 + x7), xmask)
    tmp0 = x3
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp7 & tmp5
    tmp9 = x1
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp9 == tmp10
    tmp12 = x0
    tmp13 = tmp12 == tmp10
    tmp14 = tl.load(in_ptr0 + ((-26) + x2 + (25*x3) + (5025*x4)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = -tmp14
    tmp16 = tl.load(in_ptr1 + ((-26) + x2 + (25*x3) + (5025*x4)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 0.0
    tmp18 = triton_helpers.minimum(tmp17, tmp16)
    tmp19 = 1e-20
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 / tmp20
    tmp22 = tl.abs(tmp21)
    tmp23 = -tmp22
    tmp24 = 0.001
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 / tmp24
    tmp27 = libdevice.tanh(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = 0.5
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp21
    tmp33 = tl.load(in_ptr2 + (10608 + x6), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.load(in_ptr3 + (42432 + x0 + (4*x6)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.where(tmp13, tmp34, tmp35)
    tmp37 = tl.load(in_ptr3 + (42432 + x7), tmp8 & xmask, other=0.0)
    tmp38 = tl.where(tmp11, tmp36, tmp37)
    tmp39 = tl.where(tmp8, tmp38, 0.0)
    tmp40 = tl.load(in_ptr3 + (42432 + x7), tmp5 & xmask, other=0.0)
    tmp41 = tl.where(tmp7, tmp39, tmp40)
    tmp42 = tl.where(tmp5, tmp41, 0.0)
    tmp44 = tl.where(tmp5, tmp42, tmp43)
    tl.store(out_ptr0 + (x7), tmp44, xmask)
''')
