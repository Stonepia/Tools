

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/33/c33ujmcwoxixcy54owv37izncu5ap723oae5565ekwbtlk2wazic.py
# Source Nodes: [abs_12, add_41, add_42, add_45, add_46, add_47, add_48, add_51, add_54, add_55, add_58, add_61, add_64, iadd_9, min_13, min_9, mul_102, mul_103, mul_104, mul_105, mul_113, mul_114, mul_115, mul_116, mul_117, mul_118, mul_119, mul_123, mul_124, mul_133, mul_134, mul_142, mul_143, mul_144, mul_145, mul_153, mul_154, mul_163, mul_164, mul_173, mul_174, neg_18, neg_20, neg_21, neg_26, neg_28, pow_2, sub_17, sub_21, tanh_9, tensor, truediv_24, truediv_26, truediv_27, truediv_32, truediv_34], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.sub, aten.tanh]
# abs_12 => abs_12
# add_41 => add_49
# add_42 => add_50
# add_45 => add_54
# add_46 => add_55
# add_47 => add_56
# add_48 => add_58
# add_51 => add_62
# add_54 => add_66
# add_55 => add_67
# add_58 => add_71
# add_61 => add_75
# add_64 => add_79
# iadd_9 => add_57
# min_13 => minimum_12
# min_9 => minimum_8
# mul_102 => mul_102
# mul_103 => mul_103
# mul_104 => mul_104
# mul_105 => mul_105
# mul_113 => mul_113
# mul_114 => mul_114
# mul_115 => mul_115
# mul_116 => mul_116
# mul_117 => mul_117
# mul_118 => mul_118
# mul_119 => mul_119
# mul_123 => mul_123
# mul_124 => mul_124
# mul_133 => mul_133
# mul_134 => mul_134
# mul_142 => mul_142
# mul_143 => mul_143
# mul_144 => mul_144
# mul_145 => mul_145
# mul_153 => mul_153
# mul_154 => mul_154
# mul_163 => mul_163
# mul_164 => mul_164
# mul_173 => mul_173
# mul_174 => mul_174
# neg_18 => neg_18
# neg_20 => neg_20
# neg_21 => neg_21
# neg_26 => neg_26
# neg_28 => neg_28
# pow_2 => pow_2
# sub_17 => sub_17
# sub_21 => sub_21
# tanh_9 => tanh_9
# tensor => full_default_1
# truediv_24 => div_24
# truediv_26 => div_26
# truediv_27 => div_27
# truediv_32 => div_32
# truediv_34 => div_34
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_10 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: '*fp64', 10: '*fp64', 11: '*fp64', 12: '*fp64', 13: '*fp64', 14: '*fp64', 15: '*fp64', 16: '*fp64', 17: '*fp64', 18: '*fp64', 19: '*fp64', 20: '*fp64', 21: '*fp64', 22: '*fp64', 23: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x2 = (xindex // 5000)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (10661 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp48 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), xmask)
    tmp51 = tl.load(in_ptr2 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr6 + (5356 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp69 = tl.load(in_ptr7 + (5356 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp77 = tl.load(in_ptr6 + (5357 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp79 = tl.load(in_ptr7 + (5357 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp86 = tl.load(in_ptr6 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp88 = tl.load(in_ptr7 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp93 = tl.load(in_ptr6 + (10661 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp95 = tl.load(in_ptr7 + (10661 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp214 = tl.load(in_ptr10 + (2 + x2), xmask, eviction_policy='evict_last')
    tmp215 = tl.load(in_ptr11 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp227 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp2 = tl.full([1], 9.850000000000023, tl.float64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1], 1e-05, tl.float64)
    tmp5 = tmp3 * tmp4
    tmp7 = tl.abs(tmp6)
    tmp8 = -tmp7
    tmp9 = tl.full([1], 0.0, tl.float64)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.full([1], 1.0790999999999999e-07, tl.float64)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full([1], 1024.0, tl.float64)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.full([1], 1.0, tl.float64)
    tmp16 = tmp15 - tmp14
    tmp17 = tl.full([1], 0.000167, tl.float64)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 + tmp18
    tmp20 = -tmp19
    tmp21 = tmp20 * tmp13
    tmp22 = tmp0 * tmp21
    tmp23 = x0
    tmp24 = tl.full([1], 25, tl.int64)
    tmp25 = tmp23 < tmp24
    tmp26 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = tl.load(in_ptr4 + (x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 / tmp31
    tmp33 = tl.where(tmp25, tmp32, 0.0)
    tmp34 = tl.where(tmp25, tmp33, tmp9)
    tmp35 = tmp22 * tmp34
    tmp36 = tl.full([1], 0.79872, tl.float64)
    tmp37 = tmp0 * tmp36
    tmp38 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp39 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp40 = tmp38 - tmp39
    tmp41 = tmp26 * tmp40
    tmp42 = tmp41 / tmp31
    tmp43 = tl.where(tmp25, tmp42, 0.0)
    tmp44 = tl.where(tmp25, tmp43, tmp9)
    tmp45 = tmp37 * tmp44
    tmp46 = tmp35 + tmp45
    tmp49 = tmp48 - tmp2
    tmp50 = tmp49 * tmp4
    tmp52 = tl.abs(tmp51)
    tmp53 = -tmp52
    tmp54 = tmp53 - tmp9
    tmp55 = tmp54 * tmp11
    tmp56 = tmp55 * tmp13
    tmp57 = tmp15 - tmp56
    tmp58 = tmp57 * tmp17
    tmp59 = tmp50 + tmp58
    tmp60 = -tmp59
    tmp61 = tmp60 * tmp13
    tmp62 = tmp47 * tmp61
    tmp63 = tmp62 * tmp34
    tmp64 = tmp47 * tmp36
    tmp65 = tmp64 * tmp44
    tmp66 = tmp63 + tmp65
    tmp68 = tmp22 * tmp67
    tmp70 = tmp37 * tmp69
    tmp71 = tmp68 + tmp70
    tmp72 = -tmp71
    tmp73 = triton_helpers.minimum(tmp9, tmp46)
    tmp74 = tl.full([1], 1e-20, tl.float64)
    tmp75 = tmp73 - tmp74
    tmp76 = tmp72 / tmp75
    tmp78 = tmp62 * tmp77
    tmp80 = tmp64 * tmp79
    tmp81 = tmp78 + tmp80
    tmp82 = -tmp81
    tmp83 = triton_helpers.minimum(tmp9, tmp66)
    tmp84 = tmp83 - tmp74
    tmp85 = tmp82 / tmp84
    tmp87 = tmp22 * tmp86
    tmp89 = tmp37 * tmp88
    tmp90 = tmp87 + tmp89
    tmp91 = -tmp90
    tmp92 = tmp91 / tmp75
    tmp94 = tmp62 * tmp93
    tmp96 = tmp64 * tmp95
    tmp97 = tmp94 + tmp96
    tmp98 = -tmp97
    tmp99 = tmp98 / tmp84
    tmp100 = 1 + x1
    tmp101 = tl.full([1], 203, tl.int64)
    tmp102 = tmp100 < tmp101
    tmp103 = tl.load(in_ptr8 + (10634 + x0 + (26*x1) + (5304*x2)), tmp102 & xmask, other=0.0)
    tmp104 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp105 = tl.load(in_ptr1 + (31902 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp106 = tmp104 - tmp105
    tmp107 = tmp103 * tmp106
    tmp108 = tl.load(in_ptr9 + (1 + x1), tmp102 & xmask, eviction_policy='evict_last', other=0.0)
    tmp109 = tmp107 / tmp108
    tmp110 = tl.where(tmp102, tmp109, 0.0)
    tmp111 = tl.where(tmp102, tmp110, tmp9)
    tmp112 = tmp22 * tmp111
    tmp113 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp114 = tl.load(in_ptr5 + (31902 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp115 = tmp113 - tmp114
    tmp116 = tmp103 * tmp115
    tmp117 = tmp116 / tmp108
    tmp118 = tl.where(tmp102, tmp117, 0.0)
    tmp119 = tl.where(tmp102, tmp118, tmp9)
    tmp120 = tmp37 * tmp119
    tmp121 = tmp112 + tmp120
    tmp122 = 2 + x1
    tmp123 = tmp122 < tmp101
    tmp124 = tl.load(in_ptr8 + (10660 + x0 + (26*x1) + (5304*x2)), tmp123 & xmask, other=0.0)
    tmp125 = tl.load(in_ptr1 + (32058 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp126 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp127 = tmp125 - tmp126
    tmp128 = tmp124 * tmp127
    tmp129 = tl.load(in_ptr9 + (2 + x1), tmp123 & xmask, eviction_policy='evict_last', other=0.0)
    tmp130 = tmp128 / tmp129
    tmp131 = tl.where(tmp123, tmp130, 0.0)
    tmp132 = tl.where(tmp123, tmp131, tmp9)
    tmp133 = tmp22 * tmp132
    tmp134 = tl.load(in_ptr5 + (32058 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp135 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp136 = tmp134 - tmp135
    tmp137 = tmp124 * tmp136
    tmp138 = tmp137 / tmp129
    tmp139 = tl.where(tmp123, tmp138, 0.0)
    tmp140 = tl.where(tmp123, tmp139, tmp9)
    tmp141 = tmp37 * tmp140
    tmp142 = tmp133 + tmp141
    tmp143 = tl.load(in_ptr8 + (10635 + x0 + (26*x1) + (5304*x2)), tmp102 & xmask, other=0.0)
    tmp144 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp145 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp146 = tmp144 - tmp145
    tmp147 = tmp143 * tmp146
    tmp148 = tmp147 / tmp108
    tmp149 = tl.where(tmp102, tmp148, 0.0)
    tmp150 = tl.where(tmp102, tmp149, tmp9)
    tmp151 = tmp62 * tmp150
    tmp152 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp153 = tl.load(in_ptr5 + (31905 + (3*x0) + (78*x1) + (15912*x2)), tmp102 & xmask, other=0.0)
    tmp154 = tmp152 - tmp153
    tmp155 = tmp143 * tmp154
    tmp156 = tmp155 / tmp108
    tmp157 = tl.where(tmp102, tmp156, 0.0)
    tmp158 = tl.where(tmp102, tmp157, tmp9)
    tmp159 = tmp64 * tmp158
    tmp160 = tmp151 + tmp159
    tmp161 = tl.load(in_ptr8 + (10661 + x0 + (26*x1) + (5304*x2)), tmp123 & xmask, other=0.0)
    tmp162 = tl.load(in_ptr1 + (32061 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp163 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp164 = tmp162 - tmp163
    tmp165 = tmp161 * tmp164
    tmp166 = tmp165 / tmp129
    tmp167 = tl.where(tmp123, tmp166, 0.0)
    tmp168 = tl.where(tmp123, tmp167, tmp9)
    tmp169 = tmp62 * tmp168
    tmp170 = tl.load(in_ptr5 + (32061 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp171 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp123 & xmask, other=0.0)
    tmp172 = tmp170 - tmp171
    tmp173 = tmp161 * tmp172
    tmp174 = tmp173 / tmp129
    tmp175 = tl.where(tmp123, tmp174, 0.0)
    tmp176 = tl.where(tmp123, tmp175, tmp9)
    tmp177 = tmp64 * tmp176
    tmp178 = tmp169 + tmp177
    tmp179 = 2 + x2
    tmp180 = tl.full([1], 2, tl.int64)
    tmp181 = tmp179 >= tmp180
    tmp182 = tl.full([1], 202, tl.int64)
    tmp183 = tmp179 < tmp182
    tmp184 = tmp181 & tmp183
    tmp185 = tmp122 >= tmp180
    tmp186 = tmp122 < tmp182
    tmp187 = tmp185 & tmp186
    tmp188 = tmp187 & tmp184
    tmp189 = tmp25 & tmp188
    tmp190 = tl.load(in_ptr10 + (1 + x2), tmp189 & xmask, eviction_policy='evict_last', other=0.0)
    tmp191 = tl.load(in_ptr11 + (10660 + x0 + (26*x1) + (5304*x2)), tmp189 & xmask, other=0.0)
    tmp192 = tmp190 * tmp191
    tmp193 = tl.abs(tmp76)
    tmp194 = -tmp193
    tmp195 = tl.full([1], 0.001, tl.float64)
    tmp196 = tmp194 + tmp195
    tmp197 = tmp196 / tmp195
    tmp198 = libdevice.tanh(tmp197)
    tmp199 = tmp198 + tmp15
    tmp200 = tl.full([1], 0.5, tl.float64)
    tmp201 = tmp199 * tmp200
    tmp202 = tmp192 * tmp201
    tmp203 = tmp76 * tmp76
    tmp204 = tmp202 * tmp203
    tmp205 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp189 & xmask, other=0.0)
    tmp206 = tmp204 * tmp205
    tmp207 = tmp9 + tmp206
    tmp208 = tl.where(tmp189, tmp207, 0.0)
    tmp209 = tl.where(tmp25, tmp208, tmp9)
    tmp210 = tl.where(tmp188, tmp209, 0.0)
    tmp211 = tl.where(tmp187, tmp210, tmp9)
    tmp212 = tl.where(tmp184, tmp211, 0.0)
    tmp213 = tl.where(tmp184, tmp212, tmp9)
    tmp216 = tmp214 * tmp215
    tmp217 = tl.abs(tmp92)
    tmp218 = -tmp217
    tmp219 = tmp218 + tmp195
    tmp220 = tmp219 / tmp195
    tmp221 = libdevice.tanh(tmp220)
    tmp222 = tmp221 + tmp15
    tmp223 = tmp222 * tmp200
    tmp224 = tmp216 * tmp223
    tmp225 = tmp92 * tmp92
    tmp226 = tmp224 * tmp225
    tmp228 = tmp226 * tmp227
    tmp229 = tmp213 + tmp228
    tl.store(out_ptr0 + (x3), tmp46, xmask)
    tl.store(out_ptr1 + (x3), tmp66, xmask)
    tl.store(out_ptr2 + (x3), tmp76, xmask)
    tl.store(out_ptr3 + (x3), tmp85, xmask)
    tl.store(out_ptr4 + (x3), tmp92, xmask)
    tl.store(out_ptr5 + (x3), tmp99, xmask)
    tl.store(out_ptr6 + (x3), tmp121, xmask)
    tl.store(out_ptr7 + (x3), tmp142, xmask)
    tl.store(out_ptr8 + (x3), tmp160, xmask)
    tl.store(out_ptr9 + (x3), tmp178, xmask)
    tl.store(out_ptr10 + (x3), tmp229, xmask)
''')
