

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/yj/cyjjtd7rjj32atdb26mwzbbhrcnfzc5mwksre54erkw3nqgowrtz.py
# Source Nodes: [abs_10, add_39, add_40, iadd_7, max_8, min_8, mul_58, mul_86, mul_95, mul_97, mul_98, neg_16, neg_17, setitem_26, setitem_28, sub_16, tanh_7, tensor, tensor_1, truediv_21, truediv_22, truediv_23], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.slice_scatter, aten.sub, aten.tanh]
# abs_10 => abs_10
# add_39 => add_46
# add_40 => add_47
# iadd_7 => add_48, slice_scatter_122, slice_scatter_123, slice_scatter_124, slice_scatter_125
# max_8 => maximum_7
# min_8 => minimum_7
# mul_58 => mul_58
# mul_86 => mul_86
# mul_95 => mul_95
# mul_97 => mul_97
# mul_98 => mul_98
# neg_16 => neg_16
# neg_17 => neg_17
# setitem_26 => copy_26, slice_scatter_127, slice_scatter_128, slice_scatter_129, slice_scatter_130
# setitem_28 => copy_28, slice_scatter_135, slice_scatter_136
# sub_16 => sub_16
# tanh_7 => tanh_7
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_21 => div_21
# truediv_22 => div_22
# truediv_23 => div_23
triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_32 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_32', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x2 = (xindex // 5304)
    x4 = xindex
    x0 = xindex % 26
    x3 = xindex % 5304
    tmp54 = tl.load(in_ptr0 + (10608 + x4), xmask)
    tmp70 = tl.load(in_ptr7 + (10608 + x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2 + x2
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tmp5 & tmp11
    tmp13 = tl.load(in_ptr0 + (10608 + x4), tmp12 & xmask, other=0.0)
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = tl.load(in_ptr0 + (10608 + x4), tmp11 & xmask, other=0.0)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.where(tmp11, tmp16, 0.0)
    tmp18 = tl.load(in_ptr0 + (10608 + x4), tmp5 & xmask, other=0.0)
    tmp19 = tl.where(tmp10, tmp17, tmp18)
    tmp20 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr2 + (10608 + x4), tmp5 & xmask, other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr3 + (10608 + x4), tmp5 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr4 + ((-26) + x3 + (5226*x2)), tmp5 & xmask, other=0.0)
    tmp25 = -tmp24
    tmp26 = tl.load(in_ptr5 + ((-26) + x3 + (5226*x2)), tmp5 & xmask, other=0.0)
    tmp27 = tl.full([1], 0.0, tl.float64)
    tmp28 = triton_helpers.minimum(tmp27, tmp26)
    tmp29 = tl.full([1], 1e-20, tl.float64)
    tmp30 = tmp28 - tmp29
    tmp31 = tmp25 / tmp30
    tmp32 = tl.abs(tmp31)
    tmp33 = -tmp32
    tmp34 = tl.full([1], 0.001, tl.float64)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp35 / tmp34
    tmp37 = libdevice.tanh(tmp36)
    tmp38 = tl.full([1], 1.0, tl.float64)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full([1], 0.5, tl.float64)
    tmp41 = tmp39 * tmp40
    tmp42 = tmp23 * tmp41
    tmp43 = tl.full([1], 50.0, tl.float64)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tmp45 = tmp22 * tmp44
    tmp46 = tmp19 + tmp45
    tmp47 = tl.where(tmp5, tmp46, 0.0)
    tmp48 = tmp5 & tmp10
    tmp49 = tl.load(in_ptr0 + (10608 + x4), tmp48 & xmask, other=0.0)
    tmp50 = tl.where(tmp48, tmp49, 0.0)
    tmp51 = tl.load(in_ptr0 + (10608 + x4), tmp10 & xmask, other=0.0)
    tmp52 = tl.where(tmp5, tmp50, tmp51)
    tmp53 = tl.where(tmp10, tmp52, 0.0)
    tmp55 = tl.where(tmp10, tmp53, tmp54)
    tmp56 = tl.where(tmp5, tmp47, tmp55)
    tmp57 = tl.where(tmp11, tmp56, 0.0)
    tmp58 = tl.where(tmp10, tmp57, tmp19)
    tmp59 = tl.where(tmp5, tmp58, 0.0)
    tmp60 = tl.where(tmp10, tmp56, 0.0)
    tmp61 = tl.where(tmp10, tmp60, tmp55)
    tmp62 = tl.where(tmp5, tmp59, tmp61)
    tmp63 = tl.where(tmp11, tmp62, 0.0)
    tmp64 = tl.where(tmp10, tmp63, tmp58)
    tmp65 = tl.load(in_ptr6 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tl.full([1], 4.0, tl.float64)
    tmp67 = tmp65 * tmp66
    tmp68 = tmp64 / tmp67
    tmp69 = tl.where(tmp5, tmp68, 0.0)
    tmp71 = tl.where(tmp5, tmp69, tmp70)
    tl.store(in_out_ptr0 + (x4), tmp71, xmask)
''')
