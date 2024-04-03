

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/47/c47yfvvmkwcrxyi7ipgsbqdrvasghv6sk7cc7cve5wqfvo2owwg2.py
# Source Nodes: [abs_13, add_49, add_50, min_9, mul_125, mul_130, mul_131, neg_22, neg_23, setitem_31, sub_17, tanh_10, tensor, truediv_28, truediv_29], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_13 => abs_13
# add_49 => add_59
# add_50 => add_60
# min_9 => minimum_8
# mul_125 => mul_125
# mul_130 => mul_130
# mul_131 => mul_131
# neg_22 => neg_22
# neg_23 => neg_23
# setitem_31 => copy_31, select_scatter_22, select_scatter_23, slice_scatter_153, slice_scatter_154
# sub_17 => sub_17
# tanh_10 => tanh_10
# tensor => full_default_1
# truediv_28 => div_28
# truediv_29 => div_29
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_37 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_37', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_37', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp44 = tl.load(in_ptr3 + (42432 + x7), xmask)
    tmp0 = x3
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x2
    tmp7 = tl.full([1], 25, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = x1
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = x0
    tmp14 = tmp13 == tmp11
    tmp15 = tl.load(in_ptr0 + ((-50) + x2 + (25*x3) + (5000*x4)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = -tmp15
    tmp17 = tl.load(in_ptr1 + ((-50) + x2 + (25*x3) + (5000*x4)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full([1], 0.0, tl.float64)
    tmp19 = triton_helpers.minimum(tmp18, tmp17)
    tmp20 = tl.full([1], 1e-20, tl.float64)
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 / tmp21
    tmp23 = tl.abs(tmp22)
    tmp24 = -tmp23
    tmp25 = tl.full([1], 0.001, tl.float64)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 / tmp25
    tmp28 = libdevice.tanh(tmp27)
    tmp29 = tl.full([1], 1.0, tl.float64)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1], 0.5, tl.float64)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp32 * tmp22
    tmp34 = tl.load(in_ptr2 + (10608 + x6), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 * tmp34
    tmp36 = tl.load(in_ptr3 + (42432 + x0 + (4*x6)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.where(tmp14, tmp35, tmp36)
    tmp38 = tl.load(in_ptr3 + (42432 + x7), tmp9 & xmask, other=0.0)
    tmp39 = tl.where(tmp12, tmp37, tmp38)
    tmp40 = tl.where(tmp9, tmp39, 0.0)
    tmp41 = tl.load(in_ptr3 + (42432 + x7), tmp5 & xmask, other=0.0)
    tmp42 = tl.where(tmp8, tmp40, tmp41)
    tmp43 = tl.where(tmp5, tmp42, 0.0)
    tmp45 = tl.where(tmp5, tmp43, tmp44)
    tl.store(out_ptr0 + (x7), tmp45, xmask)
''')
