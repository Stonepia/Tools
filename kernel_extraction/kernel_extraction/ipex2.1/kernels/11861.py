

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6a/c6axxfpsuxvmuibw446qc2vgbhzglkjqfgynr6vtfk553sch4rfp.py
# Source Nodes: [abs_7, abs_8, abs_9, add_14, add_15, gt_2, lt_3, maximum_6, maximum_7, minimum_4, minimum_5, mul_132, mul_133, mul_134, mul_135, mul_136, mul_137, mul_138, mul_139, mul_140, mul_141, mul_142, sub_40, sub_41, sub_42, sub_43, sub_44, tensor, tensor_3, tensor_4, truediv_71, truediv_72, where_5, where_8, where_9], Original ATen: [aten.abs, aten.add, aten.div, aten.gt, aten.lift_fresh, aten.lt, aten.maximum, aten.minimum, aten.mul, aten.rsub, aten.scalar_tensor, aten.sub, aten.where]
# abs_7 => abs_7
# abs_8 => abs_8
# abs_9 => abs_9
# add_14 => add_66
# add_15 => add_67
# gt_2 => gt_2
# lt_3 => lt_3
# maximum_6 => maximum_6
# maximum_7 => maximum_7
# minimum_4 => minimum_4
# minimum_5 => minimum_5
# mul_132 => mul_135
# mul_133 => mul_136
# mul_134 => mul_137
# mul_135 => mul_138
# mul_136 => mul_139
# mul_137 => mul_140
# mul_138 => mul_141
# mul_139 => mul_142
# mul_140 => mul_143
# mul_141 => mul_144
# mul_142 => mul_145
# sub_40 => sub_40
# sub_41 => sub_41
# sub_42 => sub_42
# sub_43 => sub_43
# sub_44 => sub_44
# tensor => full_default
# tensor_3 => full_default_7
# tensor_4 => full_default_8
# truediv_71 => div_68
# truediv_72 => div_69
# where_5 => full_default_5
# where_8 => where_8
# where_9 => where_9
triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_79 = async_compile.triton('triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_79', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_79', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_79(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x2 = (xindex // 5000)
    x4 = xindex
    tmp87 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 27, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tmp9 > tmp8
    tmp11 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp5 & xmask, other=0.0)
    tmp12 = tl.where(tmp5, tmp11, 0.0)
    tmp13 = tl.where(tmp5, tmp12, tmp8)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tl.load(in_ptr1 + (31977 + (3*x0) + (78*x1) + (15912*x2)), tmp17 & xmask, other=0.0)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.where(tmp17, tmp19, tmp8)
    tmp21 = tmp13 - tmp20
    tmp22 = (-1) + x0
    tmp23 = tl.full([1], 25, tl.int64)
    tmp24 = tmp22 < tmp23
    tmp25 = tmp24 & tmp17
    tmp26 = tl.load(in_ptr2 + (10660 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr2 + (10659 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.where(tmp25, tmp28, 0.0)
    tmp30 = tl.where(tmp24, tmp29, tmp8)
    tmp31 = tl.where(tmp17, tmp30, 0.0)
    tmp32 = tl.where(tmp17, tmp31, tmp8)
    tmp33 = tmp21 * tmp32
    tmp34 = 3 + x0
    tmp35 = tmp34 >= tmp1
    tmp36 = tmp34 < tmp3
    tmp37 = tmp35 & tmp36
    tmp38 = tl.load(in_ptr1 + (31986 + (3*x0) + (78*x1) + (15912*x2)), tmp37 & xmask, other=0.0)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tl.where(tmp37, tmp39, tmp8)
    tmp41 = 2 + x0
    tmp42 = tmp41 >= tmp1
    tmp43 = tmp41 < tmp3
    tmp44 = tmp42 & tmp43
    tmp45 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp44 & xmask, other=0.0)
    tmp46 = tl.where(tmp44, tmp45, 0.0)
    tmp47 = tl.where(tmp44, tmp46, tmp8)
    tmp48 = tmp40 - tmp47
    tmp49 = tmp0 < tmp23
    tmp50 = tmp49 & tmp44
    tmp51 = tl.load(in_ptr2 + (10662 + x0 + (26*x1) + (5304*x2)), tmp50 & xmask, other=0.0)
    tmp52 = tl.load(in_ptr2 + (10661 + x0 + (26*x1) + (5304*x2)), tmp50 & xmask, other=0.0)
    tmp53 = tmp51 * tmp52
    tmp54 = tl.where(tmp50, tmp53, 0.0)
    tmp55 = tl.where(tmp49, tmp54, tmp8)
    tmp56 = tl.where(tmp44, tmp55, 0.0)
    tmp57 = tl.where(tmp44, tmp56, tmp8)
    tmp58 = tmp48 * tmp57
    tmp59 = tl.where(tmp10, tmp33, tmp58)
    tmp60 = tmp47 - tmp13
    tmp61 = tmp14 < tmp23
    tmp62 = tmp61 & tmp5
    tmp63 = tl.load(in_ptr2 + (10661 + x0 + (26*x1) + (5304*x2)), tmp62 & xmask, other=0.0)
    tmp64 = tl.load(in_ptr2 + (10660 + x0 + (26*x1) + (5304*x2)), tmp62 & xmask, other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.where(tmp62, tmp65, 0.0)
    tmp67 = tl.where(tmp61, tmp66, tmp8)
    tmp68 = tl.where(tmp5, tmp67, 0.0)
    tmp69 = tl.where(tmp5, tmp68, tmp8)
    tmp70 = tmp60 * tmp69
    tmp71 = tl.abs(tmp70)
    tmp72 = 1e-20
    tmp73 = tmp71 < tmp72
    tmp74 = 9.999999682655225e-21
    tmp75 = tl.where(tmp73, tmp74, tmp70)
    tmp76 = tl.abs(tmp9)
    tmp77 = tmp59 / tmp75
    tmp78 = 2.0
    tmp79 = tmp77 * tmp78
    tmp80 = 1.0
    tmp81 = triton_helpers.minimum(tmp80, tmp79)
    tmp82 = triton_helpers.minimum(tmp78, tmp77)
    tmp83 = triton_helpers.maximum(tmp81, tmp82)
    tmp84 = triton_helpers.maximum(tmp8, tmp83)
    tmp85 = tmp80 - tmp84
    tmp86 = tmp9 * tmp80
    tmp88 = tmp86 / tmp87
    tmp89 = tl.abs(tmp88)
    tmp90 = tmp89 * tmp84
    tmp91 = tmp85 + tmp90
    tmp92 = tmp76 * tmp91
    tmp93 = tmp47 + tmp13
    tmp94 = tmp9 * tmp93
    tmp95 = 0.5
    tmp96 = tmp94 * tmp95
    tmp97 = tmp92 * tmp70
    tmp98 = tmp97 * tmp95
    tmp99 = tmp96 - tmp98
    tl.store(in_out_ptr0 + (x4), tmp99, xmask)
''')
