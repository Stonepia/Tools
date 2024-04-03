

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/vm/cvmycg3nif6kqmtbe2lsrvzoumoe4is37q3myb6lohnhkuht7a35.py
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
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_5 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*bf16', 19: '*bf16', 20: '*bf16', 21: '*bf16', 22: '*fp32', 23: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x2 = (xindex // 5000)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tl.load(in_ptr0 + (10661 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp48 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), xmask).to(tl.float32)
    tmp51 = tl.load(in_ptr2 + (1 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp67 = tl.load(in_ptr6 + (5356 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp69 = tl.load(in_ptr7 + (5356 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp79 = tl.load(in_ptr6 + (5357 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp81 = tl.load(in_ptr7 + (5357 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp90 = tl.load(in_ptr6 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp92 = tl.load(in_ptr7 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp98 = tl.load(in_ptr6 + (10661 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp100 = tl.load(in_ptr7 + (10661 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp224 = tl.load(in_ptr10 + (2 + x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp225 = tl.load(in_ptr11 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp238 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp2 = 9.850000000000023
    tmp3 = tmp1 - tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 * tmp4
    tmp7 = tl.abs(tmp6)
    tmp8 = -tmp7
    tmp9 = 0.0
    tmp10 = tmp8 - tmp9
    tmp11 = 1.0790999999999999e-07
    tmp12 = tmp10 * tmp11
    tmp13 = 1024.0
    tmp14 = tmp12 * tmp13
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = 0.000167
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 + tmp18
    tmp20 = -tmp19
    tmp21 = tmp20 * tmp13
    tmp22 = tmp0 * tmp21
    tmp23 = x0
    tmp24 = tl.full([1], 25, tl.int64)
    tmp25 = tmp23 < tmp24
    tmp26 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp29 = tmp27 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = tl.load(in_ptr4 + (x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 / tmp31
    tmp33 = tl.where(tmp25, tmp32, 0.0)
    tmp34 = tl.where(tmp25, tmp33, tmp9)
    tmp35 = tmp22 * tmp34
    tmp36 = 0.796875
    tmp37 = tmp0 * tmp36
    tmp38 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
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
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp46.to(tl.float32)
    tmp75 = triton_helpers.minimum(tmp9, tmp74)
    tmp76 = 1e-20
    tmp77 = tmp75 - tmp76
    tmp78 = tmp73 / tmp77
    tmp80 = tmp62 * tmp79
    tmp82 = tmp64 * tmp81
    tmp83 = tmp80 + tmp82
    tmp84 = -tmp83
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp66.to(tl.float32)
    tmp87 = triton_helpers.minimum(tmp9, tmp86)
    tmp88 = tmp87 - tmp76
    tmp89 = tmp85 / tmp88
    tmp91 = tmp22 * tmp90
    tmp93 = tmp37 * tmp92
    tmp94 = tmp91 + tmp93
    tmp95 = -tmp94
    tmp96 = tmp95.to(tl.float32)
    tmp97 = tmp96 / tmp77
    tmp99 = tmp62 * tmp98
    tmp101 = tmp64 * tmp100
    tmp102 = tmp99 + tmp101
    tmp103 = -tmp102
    tmp104 = tmp103.to(tl.float32)
    tmp105 = tmp104 / tmp88
    tmp106 = 1 + x1
    tmp107 = tl.full([1], 203, tl.int64)
    tmp108 = tmp106 < tmp107
    tmp109 = tl.load(in_ptr8 + (10634 + x0 + (26*x1) + (5304*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp110 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp111 = tl.load(in_ptr1 + (31902 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp112 = tmp110 - tmp111
    tmp113 = tmp109 * tmp112
    tmp114 = tl.load(in_ptr9 + (1 + x1), tmp108 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp115 = tmp113 / tmp114
    tmp116 = tl.where(tmp108, tmp115, 0.0)
    tmp117 = tl.where(tmp108, tmp116, tmp9)
    tmp118 = tmp22 * tmp117
    tmp119 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp120 = tl.load(in_ptr5 + (31902 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp121 = tmp119 - tmp120
    tmp122 = tmp109 * tmp121
    tmp123 = tmp122 / tmp114
    tmp124 = tl.where(tmp108, tmp123, 0.0)
    tmp125 = tl.where(tmp108, tmp124, tmp9)
    tmp126 = tmp37 * tmp125
    tmp127 = tmp118 + tmp126
    tmp128 = 2 + x1
    tmp129 = tmp128 < tmp107
    tmp130 = tl.load(in_ptr8 + (10660 + x0 + (26*x1) + (5304*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp131 = tl.load(in_ptr1 + (32058 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp132 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp133 = tmp131 - tmp132
    tmp134 = tmp130 * tmp133
    tmp135 = tl.load(in_ptr9 + (2 + x1), tmp129 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp136 = tmp134 / tmp135
    tmp137 = tl.where(tmp129, tmp136, 0.0)
    tmp138 = tl.where(tmp129, tmp137, tmp9)
    tmp139 = tmp22 * tmp138
    tmp140 = tl.load(in_ptr5 + (32058 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp141 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp142 = tmp140 - tmp141
    tmp143 = tmp130 * tmp142
    tmp144 = tmp143 / tmp135
    tmp145 = tl.where(tmp129, tmp144, 0.0)
    tmp146 = tl.where(tmp129, tmp145, tmp9)
    tmp147 = tmp37 * tmp146
    tmp148 = tmp139 + tmp147
    tmp149 = tl.load(in_ptr8 + (10635 + x0 + (26*x1) + (5304*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp150 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp151 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp152 = tmp150 - tmp151
    tmp153 = tmp149 * tmp152
    tmp154 = tmp153 / tmp114
    tmp155 = tl.where(tmp108, tmp154, 0.0)
    tmp156 = tl.where(tmp108, tmp155, tmp9)
    tmp157 = tmp62 * tmp156
    tmp158 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp159 = tl.load(in_ptr5 + (31905 + (3*x0) + (78*x1) + (15912*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp160 = tmp158 - tmp159
    tmp161 = tmp149 * tmp160
    tmp162 = tmp161 / tmp114
    tmp163 = tl.where(tmp108, tmp162, 0.0)
    tmp164 = tl.where(tmp108, tmp163, tmp9)
    tmp165 = tmp64 * tmp164
    tmp166 = tmp157 + tmp165
    tmp167 = tl.load(in_ptr8 + (10661 + x0 + (26*x1) + (5304*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp168 = tl.load(in_ptr1 + (32061 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp169 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp170 = tmp168 - tmp169
    tmp171 = tmp167 * tmp170
    tmp172 = tmp171 / tmp135
    tmp173 = tl.where(tmp129, tmp172, 0.0)
    tmp174 = tl.where(tmp129, tmp173, tmp9)
    tmp175 = tmp62 * tmp174
    tmp176 = tl.load(in_ptr5 + (32061 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp177 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp129 & xmask, other=0.0).to(tl.float32)
    tmp178 = tmp176 - tmp177
    tmp179 = tmp167 * tmp178
    tmp180 = tmp179 / tmp135
    tmp181 = tl.where(tmp129, tmp180, 0.0)
    tmp182 = tl.where(tmp129, tmp181, tmp9)
    tmp183 = tmp64 * tmp182
    tmp184 = tmp175 + tmp183
    tmp185 = 2 + x2
    tmp186 = tl.full([1], 2, tl.int64)
    tmp187 = tmp185 >= tmp186
    tmp188 = tl.full([1], 202, tl.int64)
    tmp189 = tmp185 < tmp188
    tmp190 = tmp187 & tmp189
    tmp191 = tmp128 >= tmp186
    tmp192 = tmp128 < tmp188
    tmp193 = tmp191 & tmp192
    tmp194 = tmp193 & tmp190
    tmp195 = tmp25 & tmp194
    tmp196 = tl.load(in_ptr10 + (1 + x2), tmp195 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp197 = tl.load(in_ptr11 + (10660 + x0 + (26*x1) + (5304*x2)), tmp195 & xmask, other=0.0).to(tl.float32)
    tmp198 = tmp196 * tmp197
    tmp199 = tmp198.to(tl.float32)
    tmp200 = tl.abs(tmp78)
    tmp201 = -tmp200
    tmp202 = 0.001
    tmp203 = tmp201 + tmp202
    tmp204 = tmp203 / tmp202
    tmp205 = libdevice.tanh(tmp204)
    tmp206 = tmp205 + tmp15
    tmp207 = 0.5
    tmp208 = tmp206 * tmp207
    tmp209 = tmp199 * tmp208
    tmp210 = tmp78 * tmp78
    tmp211 = tmp209 * tmp210
    tmp212 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp195 & xmask, other=0.0).to(tl.float32)
    tmp213 = tmp212.to(tl.float32)
    tmp214 = tmp211 * tmp213
    tmp215 = tmp9 + tmp214
    tmp216 = tmp215.to(tl.float32)
    tmp217 = tl.where(tmp195, tmp216, 0.0)
    tmp218 = tl.where(tmp25, tmp217, tmp9)
    tmp219 = tl.where(tmp194, tmp218, 0.0)
    tmp220 = tl.where(tmp193, tmp219, tmp9)
    tmp221 = tl.where(tmp190, tmp220, 0.0)
    tmp222 = tl.where(tmp190, tmp221, tmp9)
    tmp223 = tmp222.to(tl.float32)
    tmp226 = tmp224 * tmp225
    tmp227 = tmp226.to(tl.float32)
    tmp228 = tl.abs(tmp97)
    tmp229 = -tmp228
    tmp230 = tmp229 + tmp202
    tmp231 = tmp230 / tmp202
    tmp232 = libdevice.tanh(tmp231)
    tmp233 = tmp232 + tmp15
    tmp234 = tmp233 * tmp207
    tmp235 = tmp227 * tmp234
    tmp236 = tmp97 * tmp97
    tmp237 = tmp235 * tmp236
    tmp239 = tmp238.to(tl.float32)
    tmp240 = tmp237 * tmp239
    tmp241 = tmp223 + tmp240
    tl.store(out_ptr0 + (x3), tmp46, xmask)
    tl.store(out_ptr1 + (x3), tmp66, xmask)
    tl.store(out_ptr2 + (x3), tmp78, xmask)
    tl.store(out_ptr3 + (x3), tmp89, xmask)
    tl.store(out_ptr4 + (x3), tmp97, xmask)
    tl.store(out_ptr5 + (x3), tmp105, xmask)
    tl.store(out_ptr6 + (x3), tmp127, xmask)
    tl.store(out_ptr7 + (x3), tmp148, xmask)
    tl.store(out_ptr8 + (x3), tmp166, xmask)
    tl.store(out_ptr9 + (x3), tmp184, xmask)
    tl.store(out_ptr10 + (x3), tmp241, xmask)
''')
