

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/r3/cr3kqsewvvn2gjhewwvhi6y3g7kvjvduprs3vrvr6ly5igt6szic.py
# Source Nodes: [abs_11, abs_15, abs_7, add_1, add_2, add_21, add_22, add_23, add_27, add_28, add_3, add_43, add_44, add_56, add_57, iadd_12, iadd_4, iadd_8, iadd_9, max_5, min_5, mul_106, mul_107, mul_108, mul_109, mul_110, mul_146, mul_147, mul_148, mul_149, mul_150, mul_16, mul_59, mul_65, mul_66, mul_67, mul_68, neg_10, neg_11, neg_19, neg_27, pow_1, pow_5, setitem_17, setitem_18, setitem_19, setitem_6, setitem_7, sub_13, tanh_12, tanh_4, tanh_8, tensor, tensor_1, truediv_15, truediv_16, truediv_25, truediv_33, zeros_like], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.pow, aten.slice_scatter, aten.sub, aten.tanh, aten.zeros_like]
# abs_11 => abs_11
# abs_15 => abs_15
# abs_7 => abs_7
# add_1 => add_1
# add_2 => add_2
# add_21 => add_25
# add_22 => add_26
# add_23 => add_27
# add_27 => add_31
# add_28 => add_32
# add_3 => add_3
# add_43 => add_51
# add_44 => add_52
# add_56 => add_68
# add_57 => add_69
# iadd_12 => add_70, slice_scatter_162, slice_scatter_163, slice_scatter_164
# iadd_4 => add_33, slice_scatter_83, slice_scatter_84, slice_scatter_85, slice_scatter_86, slice_scatter_87
# iadd_8 => add_53, slice_scatter_138, slice_scatter_139, slice_scatter_140
# iadd_9 => slice_scatter_146
# max_5 => maximum_4
# min_5 => minimum_4
# mul_106 => mul_106
# mul_107 => mul_107
# mul_108 => mul_108
# mul_109 => mul_109
# mul_110 => mul_110
# mul_146 => mul_146
# mul_147 => mul_147
# mul_148 => mul_148
# mul_149 => mul_149
# mul_150 => mul_150
# mul_16 => mul_16
# mul_59 => mul_59
# mul_65 => mul_65
# mul_66 => mul_66
# mul_67 => mul_67
# mul_68 => mul_68
# neg_10 => neg_10
# neg_11 => neg_11
# neg_19 => neg_19
# neg_27 => neg_27
# pow_1 => pow_1
# pow_5 => pow_5
# setitem_17 => copy_17
# setitem_18 => copy_18, slice_scatter_78, slice_scatter_79, slice_scatter_80
# setitem_19 => slice_scatter_82
# setitem_6 => copy_6, slice_scatter_18, slice_scatter_19, slice_scatter_20
# setitem_7 => slice_scatter_22
# sub_13 => sub_13
# tanh_12 => tanh_12
# tanh_4 => tanh_4
# tanh_8 => tanh_8
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_15 => div_15
# truediv_16 => div_16
# truediv_25 => div_25
# truediv_33 => div_33
# zeros_like => full
triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_13 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: '*fp64', 10: '*fp64', 11: '*fp64', 12: '*fp64', 13: '*fp64', 14: '*fp64', 15: '*fp64', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x3 = xindex
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-5304) + x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = x1
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tmp8 < tmp3
    tmp12 = tmp10 & tmp11
    tmp13 = tmp12 & tmp5
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp15 & tmp13
    tmp17 = tl.load(in_ptr1 + (x3), tmp16 & xmask, other=0.0)
    tmp18 = tl.load(in_ptr1 + ((-1) + x3), tmp16 & xmask, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr1 + (5304 + x3), tmp16 & xmask, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr1 + (5303 + x3), tmp16 & xmask, other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0.25, tl.float64)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.where(tmp16, tmp25, 0.0)
    tmp27 = tl.full([1], 0.0, tl.float64)
    tmp28 = tl.where(tmp15, tmp26, tmp27)
    tmp29 = tl.where(tmp13, tmp28, 0.0)
    tmp30 = tl.where(tmp12, tmp29, tmp27)
    tmp31 = tl.where(tmp5, tmp30, 0.0)
    tmp32 = tl.where(tmp5, tmp31, tmp27)
    tmp33 = tl.where(tmp5, tmp7, tmp32)
    tmp34 = tmp0 >= tmp9
    tmp35 = tmp34 & tmp4
    tmp36 = tl.load(in_ptr2 + ((-10608) + x3), tmp35 & xmask, other=0.0)
    tmp37 = tl.where(tmp35, tmp36, 0.0)
    tmp38 = tmp8 >= tmp1
    tmp39 = tmp38 & tmp11
    tmp40 = tmp39 & tmp35
    tmp41 = tmp15 & tmp40
    tmp42 = tl.load(in_ptr1 + (x3), tmp41 & xmask, other=0.0)
    tmp43 = tl.load(in_ptr1 + ((-1) + x3), tmp41 & xmask, other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.load(in_ptr1 + (26 + x3), tmp41 & xmask, other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = tl.load(in_ptr1 + (25 + x3), tmp41 & xmask, other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48 * tmp24
    tmp50 = tl.where(tmp41, tmp49, 0.0)
    tmp51 = tl.where(tmp15, tmp50, tmp27)
    tmp52 = tl.where(tmp40, tmp51, 0.0)
    tmp53 = tl.where(tmp39, tmp52, tmp27)
    tmp54 = tl.where(tmp35, tmp53, 0.0)
    tmp55 = tl.where(tmp35, tmp54, tmp27)
    tmp56 = tl.where(tmp35, tmp37, tmp55)
    tmp57 = tl.load(in_ptr3 + ((-1) + x0), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr4 + (x3), tmp41 & xmask, other=0.0)
    tmp59 = tmp57 * tmp58
    tmp60 = tl.load(in_ptr5 + ((-10076) + x0 + (25*x1) + (5025*x2)), tmp41 & xmask, other=0.0)
    tmp61 = -tmp60
    tmp62 = tl.load(in_ptr6 + ((-10076) + x0 + (25*x1) + (5025*x2)), tmp41 & xmask, other=0.0)
    tmp63 = triton_helpers.minimum(tmp27, tmp62)
    tmp64 = tl.full([1], 1e-20, tl.float64)
    tmp65 = tmp63 - tmp64
    tmp66 = tmp61 / tmp65
    tmp67 = tl.abs(tmp66)
    tmp68 = -tmp67
    tmp69 = tl.full([1], 0.001, tl.float64)
    tmp70 = tmp68 + tmp69
    tmp71 = tmp70 / tmp69
    tmp72 = libdevice.tanh(tmp71)
    tmp73 = tl.full([1], 1.0, tl.float64)
    tmp74 = tmp72 + tmp73
    tmp75 = tl.full([1], 0.5, tl.float64)
    tmp76 = tmp74 * tmp75
    tmp77 = tmp56 * tmp76
    tmp78 = tl.full([1], 50.0, tl.float64)
    tmp79 = triton_helpers.maximum(tmp78, tmp77)
    tmp80 = tmp59 * tmp79
    tmp81 = tmp27 + tmp80
    tmp82 = tl.where(tmp41, tmp81, 0.0)
    tmp83 = tl.where(tmp15, tmp82, tmp27)
    tmp84 = tl.where(tmp40, tmp83, 0.0)
    tmp85 = tl.where(tmp39, tmp84, tmp27)
    tmp86 = tl.where(tmp35, tmp85, 0.0)
    tmp87 = tl.where(tmp35, tmp86, tmp27)
    tmp88 = tl.load(in_ptr7 + ((-10608) + x3), tmp35 & xmask, other=0.0)
    tmp89 = tl.where(tmp35, tmp88, 0.0)
    tmp90 = tmp12 & tmp35
    tmp91 = tl.full([1], 25, tl.int64)
    tmp92 = tmp14 < tmp91
    tmp93 = tmp92 & tmp90
    tmp94 = tl.load(in_ptr8 + ((-1) + x2), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = tl.load(in_ptr1 + (x3), tmp93 & xmask, other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr9 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp93 & xmask, other=0.0)
    tmp98 = tl.abs(tmp97)
    tmp99 = -tmp98
    tmp100 = tmp99 + tmp69
    tmp101 = tmp100 / tmp69
    tmp102 = libdevice.tanh(tmp101)
    tmp103 = tmp102 + tmp73
    tmp104 = tmp103 * tmp75
    tmp105 = tmp96 * tmp104
    tmp106 = tmp97 * tmp97
    tmp107 = tmp105 * tmp106
    tmp108 = tl.load(in_ptr10 + (x3), tmp93 & xmask, other=0.0)
    tmp109 = tmp107 * tmp108
    tmp110 = tmp27 + tmp109
    tmp111 = tl.where(tmp93, tmp110, 0.0)
    tmp112 = tl.where(tmp92, tmp111, tmp27)
    tmp113 = tl.where(tmp90, tmp112, 0.0)
    tmp114 = tl.where(tmp12, tmp113, tmp27)
    tmp115 = tl.where(tmp35, tmp114, 0.0)
    tmp116 = tl.where(tmp35, tmp115, tmp27)
    tmp117 = tl.where(tmp35, tmp89, tmp116)
    tmp118 = tl.load(in_ptr11 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp93 & xmask, other=0.0)
    tmp119 = tl.abs(tmp118)
    tmp120 = -tmp119
    tmp121 = tmp120 + tmp69
    tmp122 = tmp121 / tmp69
    tmp123 = libdevice.tanh(tmp122)
    tmp124 = tmp123 + tmp73
    tmp125 = tmp124 * tmp75
    tmp126 = tmp96 * tmp125
    tmp127 = tmp118 * tmp118
    tmp128 = tmp126 * tmp127
    tmp129 = tmp128 * tmp108
    tmp130 = tmp117 + tmp129
    tmp131 = tl.where(tmp93, tmp130, 0.0)
    tmp132 = tl.where(tmp92, tmp131, tmp117)
    tmp133 = tl.where(tmp90, tmp132, 0.0)
    tmp134 = tl.where(tmp12, tmp133, tmp117)
    tmp135 = tl.where(tmp35, tmp134, 0.0)
    tmp136 = tl.where(tmp35, tmp135, tmp117)
    tl.store(out_ptr0 + (x3), tmp33, xmask)
    tl.store(out_ptr1 + (x3), tmp56, xmask)
    tl.store(out_ptr2 + (x3), tmp87, xmask)
    tl.store(in_out_ptr0 + (x3), tmp136, xmask)
''')
