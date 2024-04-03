

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/iw/ciwdyowvsgtocqygri7afrjkb6s64yvc75otvwyvdeppwve6cjzz.py
# Source Nodes: [abs_9, add_35, add_36, min_7, mul_85, mul_89, mul_90, neg_14, neg_15, setitem_25, sub_15, tanh_6, tensor, truediv_19, truediv_20], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_9 => abs_9
# add_35 => add_41
# add_36 => add_42
# min_7 => minimum_6
# mul_85 => mul_85
# mul_89 => mul_89
# mul_90 => mul_90
# neg_14 => neg_14
# neg_15 => neg_15
# setitem_25 => copy_25, select_scatter_14, select_scatter_15, slice_scatter_119, slice_scatter_120
# sub_15 => sub_15
# tanh_6 => tanh_6
# tensor => full_default_1
# truediv_19 => div_19
# truediv_20 => div_20
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_25 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4243200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 104) % 204
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x4 = (xindex // 21216)
    x6 = (xindex // 4) % 5304
    x7 = (xindex // 4)
    x8 = xindex
    tmp38 = tl.load(in_ptr3 + (42432 + x8), xmask)
    tmp0 = x3
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = x0
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.load(in_ptr0 + ((-26) + x6 + (5226*x4)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = -tmp12
    tmp14 = tl.load(in_ptr1 + ((-26) + x6 + (5226*x4)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 0.0, tl.float64)
    tmp16 = triton_helpers.minimum(tmp15, tmp14)
    tmp17 = tl.full([1], 1e-20, tl.float64)
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 / tmp18
    tmp20 = tl.abs(tmp19)
    tmp21 = -tmp20
    tmp22 = tl.full([1], 0.001, tl.float64)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 / tmp22
    tmp25 = libdevice.tanh(tmp24)
    tmp26 = tl.full([1], 1.0, tl.float64)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full([1], 0.5, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29 * tmp19
    tmp31 = tl.load(in_ptr2 + (10608 + x7), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr3 + (42432 + x0 + (4*x7)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp11, tmp32, tmp33)
    tmp35 = tl.load(in_ptr3 + (42432 + x8), tmp5 & xmask, other=0.0)
    tmp36 = tl.where(tmp8, tmp34, tmp35)
    tmp37 = tl.where(tmp5, tmp36, 0.0)
    tmp39 = tl.where(tmp5, tmp37, tmp38)
    tl.store(out_ptr0 + (x8), tmp39, xmask)
''')
