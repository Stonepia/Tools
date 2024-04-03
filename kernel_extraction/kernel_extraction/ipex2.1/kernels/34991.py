

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/53/c53hn52776dtdjbuvcyl7mmhdf4b25iasebd3njrpdyzrhkwzp3u.py
# Source Nodes: [abs_11, abs_15, abs_16, abs_7, add_1, add_2, add_21, add_22, add_23, add_27, add_28, add_3, add_43, add_44, add_56, add_57, add_59, add_60, iadd_12, iadd_13, iadd_4, iadd_8, iadd_9, max_5, min_5, mul_106, mul_107, mul_108, mul_109, mul_110, mul_146, mul_147, mul_148, mul_149, mul_150, mul_155, mul_156, mul_157, mul_158, mul_159, mul_16, mul_59, mul_65, mul_66, mul_67, mul_68, neg_10, neg_11, neg_19, neg_27, neg_29, pow_1, pow_5, pow_6, setitem_17, setitem_18, setitem_19, setitem_6, setitem_7, sub_13, tanh_12, tanh_13, tanh_4, tanh_8, tensor, tensor_1, truediv_15, truediv_16, truediv_25, truediv_33, truediv_35, zeros_like], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.pow, aten.slice_scatter, aten.sub, aten.tanh, aten.zeros_like]
# abs_11 => abs_11
# abs_15 => abs_15
# abs_16 => abs_16
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
# add_59 => add_72
# add_60 => add_73
# iadd_12 => add_70, convert_element_type_12, slice_scatter_162, slice_scatter_163, slice_scatter_164
# iadd_13 => add_74, convert_element_type_13, slice_scatter_168, slice_scatter_169, slice_scatter_170
# iadd_4 => add_33, convert_element_type_4, slice_scatter_83, slice_scatter_84, slice_scatter_85, slice_scatter_86, slice_scatter_87
# iadd_8 => add_53, convert_element_type_8, slice_scatter_138, slice_scatter_139, slice_scatter_140
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
# mul_155 => mul_155
# mul_156 => mul_156
# mul_157 => mul_157
# mul_158 => mul_158
# mul_159 => mul_159
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
# neg_29 => neg_29
# pow_1 => pow_1
# pow_5 => pow_5
# pow_6 => pow_6
# setitem_17 => copy_17
# setitem_18 => copy_18, slice_scatter_78, slice_scatter_79, slice_scatter_80
# setitem_19 => slice_scatter_82
# setitem_6 => copy_6, slice_scatter_18, slice_scatter_19, slice_scatter_20
# setitem_7 => slice_scatter_22
# sub_13 => sub_13
# tanh_12 => tanh_12
# tanh_13 => tanh_13
# tanh_4 => tanh_4
# tanh_8 => tanh_8
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_15 => div_15
# truediv_16 => div_16
# truediv_25 => div_25
# truediv_33 => div_33
# truediv_35 => div_35
# zeros_like => full
triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_8 = async_compile.triton('triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*fp32', 11: '*bf16', 12: '*fp32', 13: '*fp32', 14: '*bf16', 15: '*bf16', 16: '*bf16', 17: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_pow_slice_scatter_sub_tanh_zeros_like_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr0 + ((-5304) + x3), tmp5 & xmask, other=0.0).to(tl.float32)
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
    tmp17 = tl.load(in_ptr1 + (x3), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr1 + ((-1) + x3), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr1 + (5304 + x3), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr1 + (5303 + x3), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = 0.25
    tmp25 = tmp23 * tmp24
    tmp26 = tl.where(tmp16, tmp25, 0.0)
    tmp27 = 0.0
    tmp28 = tl.where(tmp15, tmp26, tmp27)
    tmp29 = tl.where(tmp13, tmp28, 0.0)
    tmp30 = tl.where(tmp12, tmp29, tmp27)
    tmp31 = tl.where(tmp5, tmp30, 0.0)
    tmp32 = tl.where(tmp5, tmp31, tmp27)
    tmp33 = tl.where(tmp5, tmp7, tmp32)
    tmp34 = tmp0 >= tmp9
    tmp35 = tmp34 & tmp4
    tmp36 = tl.load(in_ptr2 + ((-10608) + x3), tmp35 & xmask, other=0.0).to(tl.float32)
    tmp37 = tl.where(tmp35, tmp36, 0.0)
    tmp38 = tmp8 >= tmp1
    tmp39 = tmp38 & tmp11
    tmp40 = tmp39 & tmp35
    tmp41 = tmp15 & tmp40
    tmp42 = tl.load(in_ptr1 + (x3), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr1 + ((-1) + x3), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.load(in_ptr1 + (26 + x3), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp46 = tmp44 + tmp45
    tmp47 = tl.load(in_ptr1 + (25 + x3), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48 * tmp24
    tmp50 = tl.where(tmp41, tmp49, 0.0)
    tmp51 = tl.where(tmp15, tmp50, tmp27)
    tmp52 = tl.where(tmp40, tmp51, 0.0)
    tmp53 = tl.where(tmp39, tmp52, tmp27)
    tmp54 = tl.where(tmp35, tmp53, 0.0)
    tmp55 = tl.where(tmp35, tmp54, tmp27)
    tmp56 = tl.where(tmp35, tmp37, tmp55)
    tmp57 = tl.load(in_ptr3 + ((-1) + x0), tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp58 = tl.load(in_ptr4 + (x3), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp59 = tmp57 * tmp58
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp56.to(tl.float32)
    tmp62 = tl.load(in_ptr5 + ((-10076) + x0 + (25*x1) + (5025*x2)), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp63 = -tmp62
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tl.load(in_ptr6 + ((-10076) + x0 + (25*x1) + (5025*x2)), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = triton_helpers.minimum(tmp27, tmp66)
    tmp68 = 1e-20
    tmp69 = tmp67 - tmp68
    tmp70 = tmp64 / tmp69
    tmp71 = tl.abs(tmp70)
    tmp72 = -tmp71
    tmp73 = 0.001
    tmp74 = tmp72 + tmp73
    tmp75 = tmp74 / tmp73
    tmp76 = libdevice.tanh(tmp75)
    tmp77 = 1.0
    tmp78 = tmp76 + tmp77
    tmp79 = 0.5
    tmp80 = tmp78 * tmp79
    tmp81 = tmp61 * tmp80
    tmp82 = 50.0
    tmp83 = triton_helpers.maximum(tmp82, tmp81)
    tmp84 = tmp60 * tmp83
    tmp85 = tmp27 + tmp84
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tl.where(tmp41, tmp86, 0.0)
    tmp88 = tl.where(tmp15, tmp87, tmp27)
    tmp89 = tl.where(tmp40, tmp88, 0.0)
    tmp90 = tl.where(tmp39, tmp89, tmp27)
    tmp91 = tl.where(tmp35, tmp90, 0.0)
    tmp92 = tl.where(tmp35, tmp91, tmp27)
    tmp93 = tl.load(in_ptr7 + ((-10608) + x3), tmp35 & xmask, other=0.0).to(tl.float32)
    tmp94 = tl.where(tmp35, tmp93, 0.0)
    tmp95 = tmp12 & tmp35
    tmp96 = tl.full([1], 25, tl.int64)
    tmp97 = tmp14 < tmp96
    tmp98 = tmp97 & tmp95
    tmp99 = tl.load(in_ptr8 + ((-1) + x2), tmp98 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp100 = tl.load(in_ptr1 + (x3), tmp98 & xmask, other=0.0).to(tl.float32)
    tmp101 = tmp99 * tmp100
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tl.load(in_ptr9 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp98 & xmask, other=0.0)
    tmp104 = tl.abs(tmp103)
    tmp105 = -tmp104
    tmp106 = tmp105 + tmp73
    tmp107 = tmp106 / tmp73
    tmp108 = libdevice.tanh(tmp107)
    tmp109 = tmp108 + tmp77
    tmp110 = tmp109 * tmp79
    tmp111 = tmp102 * tmp110
    tmp112 = tmp103 * tmp103
    tmp113 = tmp111 * tmp112
    tmp114 = tl.load(in_ptr10 + (x3), tmp98 & xmask, other=0.0).to(tl.float32)
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tmp113 * tmp115
    tmp117 = tmp27 + tmp116
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tl.where(tmp98, tmp118, 0.0)
    tmp120 = tl.where(tmp97, tmp119, tmp27)
    tmp121 = tl.where(tmp95, tmp120, 0.0)
    tmp122 = tl.where(tmp12, tmp121, tmp27)
    tmp123 = tl.where(tmp35, tmp122, 0.0)
    tmp124 = tl.where(tmp35, tmp123, tmp27)
    tmp125 = tl.where(tmp35, tmp94, tmp124)
    tmp126 = tmp125.to(tl.float32)
    tmp127 = tl.load(in_ptr11 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp98 & xmask, other=0.0)
    tmp128 = tl.abs(tmp127)
    tmp129 = -tmp128
    tmp130 = tmp129 + tmp73
    tmp131 = tmp130 / tmp73
    tmp132 = libdevice.tanh(tmp131)
    tmp133 = tmp132 + tmp77
    tmp134 = tmp133 * tmp79
    tmp135 = tmp102 * tmp134
    tmp136 = tmp127 * tmp127
    tmp137 = tmp135 * tmp136
    tmp138 = tmp137 * tmp115
    tmp139 = tmp126 + tmp138
    tmp140 = tmp139.to(tl.float32)
    tmp141 = tl.where(tmp98, tmp140, 0.0)
    tmp142 = tl.where(tmp97, tmp141, tmp125)
    tmp143 = tl.where(tmp95, tmp142, 0.0)
    tmp144 = tl.where(tmp12, tmp143, tmp125)
    tmp145 = tl.where(tmp35, tmp144, 0.0)
    tmp146 = tl.where(tmp35, tmp145, tmp125)
    tmp147 = tmp146.to(tl.float32)
    tmp148 = tl.load(in_ptr8 + (x2), tmp98 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp149 = tmp148 * tmp100
    tmp150 = tmp149.to(tl.float32)
    tmp151 = tl.load(in_ptr12 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp98 & xmask, other=0.0)
    tmp152 = tl.abs(tmp151)
    tmp153 = -tmp152
    tmp154 = tmp153 + tmp73
    tmp155 = tmp154 / tmp73
    tmp156 = libdevice.tanh(tmp155)
    tmp157 = tmp156 + tmp77
    tmp158 = tmp157 * tmp79
    tmp159 = tmp150 * tmp158
    tmp160 = tmp151 * tmp151
    tmp161 = tmp159 * tmp160
    tmp162 = tmp161 * tmp115
    tmp163 = tmp147 + tmp162
    tmp164 = tmp163.to(tl.float32)
    tmp165 = tl.where(tmp98, tmp164, 0.0)
    tmp166 = tl.where(tmp97, tmp165, tmp146)
    tmp167 = tl.where(tmp95, tmp166, 0.0)
    tmp168 = tl.where(tmp12, tmp167, tmp146)
    tmp169 = tl.where(tmp35, tmp168, 0.0)
    tmp170 = tl.where(tmp35, tmp169, tmp146)
    tl.store(out_ptr0 + (x3), tmp33, xmask)
    tl.store(out_ptr1 + (x3), tmp56, xmask)
    tl.store(out_ptr2 + (x3), tmp92, xmask)
    tl.store(in_out_ptr0 + (x3), tmp170, xmask)
''')
