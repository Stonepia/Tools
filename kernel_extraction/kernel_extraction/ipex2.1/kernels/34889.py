

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/vh/cvh7zugyfxuxcogpcbxfwr7vlth5sybdjztm7vyidlvwnydojekh.py
# Source Nodes: [abs_5, add_13, add_14, add_15, add_16, add_17, add_18, add_33, add_34, add_37, add_38, iadd_2, max_3, min_3, min_4, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_48, mul_49, mul_50, mul_51, mul_81, mul_82, mul_83, mul_84, mul_91, mul_92, mul_93, mul_94, neg_6, neg_7, neg_8, sub_11, sub_12, tanh_2, tensor, tensor_1, truediv_10, truediv_11, truediv_12], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.sub, aten.tanh]
# abs_5 => abs_5
# add_13 => add_15
# add_14 => add_16
# add_15 => add_17
# add_16 => add_18
# add_17 => add_20
# add_18 => add_21
# add_33 => add_39
# add_34 => add_40
# add_37 => add_44
# add_38 => add_45
# iadd_2 => add_19, convert_element_type_2
# max_3 => maximum_2
# min_3 => minimum_2
# min_4 => minimum_3
# mul_38 => mul_38
# mul_39 => mul_39
# mul_40 => mul_40
# mul_41 => mul_41
# mul_42 => mul_42
# mul_43 => mul_43
# mul_44 => mul_44
# mul_45 => mul_45
# mul_48 => mul_48
# mul_49 => mul_49
# mul_50 => mul_50
# mul_51 => mul_51
# mul_81 => mul_81
# mul_82 => mul_82
# mul_83 => mul_83
# mul_84 => mul_84
# mul_91 => mul_91
# mul_92 => mul_92
# mul_93 => mul_93
# mul_94 => mul_94
# neg_6 => neg_6
# neg_7 => neg_7
# neg_8 => neg_8
# sub_11 => sub_11
# sub_12 => sub_12
# tanh_2 => tanh_2
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_10 => div_10
# truediv_11 => div_11
# truediv_12 => div_12
triton_poi_fused__to_copy_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_12 = async_compile.triton('triton_poi_fused__to_copy_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp32', 14: '*fp32', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr1, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5200)
    x3 = xindex % 5200
    x0 = xindex % 26
    x4 = xindex
    x1 = (xindex // 26) % 200
    x6 = (xindex // 5226)
    x7 = xindex % 5226
    x5 = (xindex // 26) % 201
    tmp0 = tl.load(in_ptr0 + (5356 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (16068 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tl.load(in_ptr6 + (5356 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp49 = tl.load(in_ptr7 + (5356 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp59 = tl.load(in_ptr0 + (10660 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp60 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp117 = tl.load(in_ptr8 + (5356 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp120 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp121 = tl.load(in_ptr9 + (5356 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp124 = tl.load(in_ptr10 + (5356 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp141 = tl.load(in_ptr0 + (10634 + x7 + (5304*x6)), xmask).to(tl.float32)
    tmp142 = tl.load(in_ptr1 + (31902 + (3*x7) + (15912*x6)), xmask).to(tl.float32)
    tmp172 = tl.load(in_ptr0 + (10660 + x7 + (5304*x6)), xmask).to(tl.float32)
    tmp173 = tl.load(in_ptr1 + (31980 + (3*x7) + (15912*x6)), xmask).to(tl.float32)
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
    tmp26 = tl.load(in_ptr3 + (5356 + x3 + (5304*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (16071 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (16068 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp29 = tmp27 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = tl.load(in_ptr4 + (x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 / tmp31
    tmp33 = tl.where(tmp25, tmp32, 0.0)
    tmp34 = tl.where(tmp25, tmp33, tmp9)
    tmp35 = tmp22 * tmp34
    tmp36 = 0.798828125
    tmp37 = tmp0 * tmp36
    tmp38 = tl.load(in_ptr5 + (16071 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr5 + (16068 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp40 = tmp38 - tmp39
    tmp41 = tmp26 * tmp40
    tmp42 = tmp41 / tmp31
    tmp43 = tl.where(tmp25, tmp42, 0.0)
    tmp44 = tl.where(tmp25, tmp43, tmp9)
    tmp45 = tmp37 * tmp44
    tmp46 = tmp35 + tmp45
    tmp48 = tmp22 * tmp47
    tmp50 = tmp37 * tmp49
    tmp51 = tmp48 + tmp50
    tmp52 = -tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp46.to(tl.float32)
    tmp55 = triton_helpers.minimum(tmp54, tmp9)
    tmp56 = 1e-20
    tmp57 = tmp55 - tmp56
    tmp58 = tmp53 / tmp57
    tmp61 = tmp60 - tmp2
    tmp62 = tmp61 * tmp4
    tmp63 = tmp62 + tmp18
    tmp64 = -tmp63
    tmp65 = tmp64 * tmp13
    tmp66 = tmp59 * tmp65
    tmp67 = tl.load(in_ptr3 + (10660 + x3 + (5304*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp68 = tl.load(in_ptr1 + (31983 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp69 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp70 = tmp68 - tmp69
    tmp71 = tmp67 * tmp70
    tmp72 = tmp71 / tmp31
    tmp73 = tl.where(tmp25, tmp72, 0.0)
    tmp74 = tl.where(tmp25, tmp73, tmp9)
    tmp75 = tmp66 * tmp74
    tmp76 = tmp59 * tmp36
    tmp77 = tl.load(in_ptr5 + (31983 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp78 = tl.load(in_ptr5 + (31980 + (3*x3) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp79 = tmp77 - tmp78
    tmp80 = tmp67 * tmp79
    tmp81 = tmp80 / tmp31
    tmp82 = tl.where(tmp25, tmp81, 0.0)
    tmp83 = tl.where(tmp25, tmp82, tmp9)
    tmp84 = tmp76 * tmp83
    tmp85 = tmp75 + tmp84
    tmp86 = tmp66 * tmp47
    tmp87 = tmp76 * tmp49
    tmp88 = tmp86 + tmp87
    tmp89 = -tmp88
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tmp85.to(tl.float32)
    tmp92 = triton_helpers.minimum(tmp91, tmp9)
    tmp93 = tmp92 - tmp56
    tmp94 = tmp90 / tmp93
    tmp95 = 1 + x2
    tmp96 = tl.full([1], 1, tl.int64)
    tmp97 = tmp95 >= tmp96
    tmp98 = tl.full([1], 202, tl.int64)
    tmp99 = tmp95 < tmp98
    tmp100 = tmp97 & tmp99
    tmp101 = 2 + x1
    tmp102 = tl.full([1], 2, tl.int64)
    tmp103 = tmp101 >= tmp102
    tmp104 = tmp101 < tmp98
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp100
    tmp107 = tmp23 >= tmp96
    tmp108 = tmp107 & tmp106
    tmp109 = tl.load(in_ptr8 + (5356 + x3 + (5304*x2)), tmp108 & xmask, other=0.0).to(tl.float32)
    tmp110 = tl.where(tmp108, tmp109, 0.0)
    tmp111 = tl.load(in_ptr8 + (5356 + x3 + (5304*x2)), tmp106 & xmask, other=0.0).to(tl.float32)
    tmp112 = tl.where(tmp107, tmp110, tmp111)
    tmp113 = tl.where(tmp106, tmp112, 0.0)
    tmp114 = tl.load(in_ptr8 + (5356 + x3 + (5304*x2)), tmp100 & xmask, other=0.0).to(tl.float32)
    tmp115 = tl.where(tmp105, tmp113, tmp114)
    tmp116 = tl.where(tmp100, tmp115, 0.0)
    tmp118 = tl.where(tmp100, tmp116, tmp117)
    tmp119 = tmp118.to(tl.float32)
    tmp122 = tmp120 * tmp121
    tmp123 = tmp122.to(tl.float32)
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tl.abs(tmp58)
    tmp127 = -tmp126
    tmp128 = 0.001
    tmp129 = tmp127 + tmp128
    tmp130 = tmp129 / tmp128
    tmp131 = libdevice.tanh(tmp130)
    tmp132 = tmp131 + tmp15
    tmp133 = 0.5
    tmp134 = tmp132 * tmp133
    tmp135 = tmp125 * tmp134
    tmp136 = 50.0
    tmp137 = triton_helpers.maximum(tmp136, tmp135)
    tmp138 = tmp123 * tmp137
    tmp139 = tmp119 + tmp138
    tmp140 = tmp139.to(tl.float32)
    tmp143 = tmp142 - tmp2
    tmp144 = tmp143 * tmp4
    tmp145 = tmp144 + tmp18
    tmp146 = -tmp145
    tmp147 = tmp146 * tmp13
    tmp148 = tmp141 * tmp147
    tmp149 = 1 + x5
    tmp150 = tl.full([1], 203, tl.int64)
    tmp151 = tmp149 < tmp150
    tmp152 = tl.load(in_ptr11 + (10634 + x7 + (5304*x6)), tmp151 & xmask, other=0.0).to(tl.float32)
    tmp153 = tl.load(in_ptr1 + (31980 + (3*x7) + (15912*x6)), tmp151 & xmask, other=0.0).to(tl.float32)
    tmp154 = tl.load(in_ptr1 + (31902 + (3*x7) + (15912*x6)), tmp151 & xmask, other=0.0).to(tl.float32)
    tmp155 = tmp153 - tmp154
    tmp156 = tmp152 * tmp155
    tmp157 = tl.load(in_ptr12 + (1 + x5), tmp151 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp158 = tmp156 / tmp157
    tmp159 = tl.where(tmp151, tmp158, 0.0)
    tmp160 = tl.where(tmp151, tmp159, tmp9)
    tmp161 = tmp148 * tmp160
    tmp162 = tmp141 * tmp36
    tmp163 = tl.load(in_ptr5 + (31980 + (3*x7) + (15912*x6)), tmp151 & xmask, other=0.0).to(tl.float32)
    tmp164 = tl.load(in_ptr5 + (31902 + (3*x7) + (15912*x6)), tmp151 & xmask, other=0.0).to(tl.float32)
    tmp165 = tmp163 - tmp164
    tmp166 = tmp152 * tmp165
    tmp167 = tmp166 / tmp157
    tmp168 = tl.where(tmp151, tmp167, 0.0)
    tmp169 = tl.where(tmp151, tmp168, tmp9)
    tmp170 = tmp162 * tmp169
    tmp171 = tmp161 + tmp170
    tmp174 = tmp173 - tmp2
    tmp175 = tmp174 * tmp4
    tmp176 = tmp175 + tmp18
    tmp177 = -tmp176
    tmp178 = tmp177 * tmp13
    tmp179 = tmp172 * tmp178
    tmp180 = tmp179 * tmp160
    tmp181 = tmp172 * tmp36
    tmp182 = tmp181 * tmp169
    tmp183 = tmp180 + tmp182
    tmp184 = tl.load(in_ptr3 + (10634 + x7 + (5304*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp185 = tl.load(in_ptr1 + (31905 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp186 = tl.load(in_ptr1 + (31902 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp187 = tmp185 - tmp186
    tmp188 = tmp184 * tmp187
    tmp189 = tmp188 / tmp31
    tmp190 = tl.where(tmp25, tmp189, 0.0)
    tmp191 = tl.where(tmp25, tmp190, tmp9)
    tmp192 = tmp148 * tmp191
    tmp193 = tl.load(in_ptr5 + (31905 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp194 = tl.load(in_ptr5 + (31902 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp195 = tmp193 - tmp194
    tmp196 = tmp184 * tmp195
    tmp197 = tmp196 / tmp31
    tmp198 = tl.where(tmp25, tmp197, 0.0)
    tmp199 = tl.where(tmp25, tmp198, tmp9)
    tmp200 = tmp162 * tmp199
    tmp201 = tmp192 + tmp200
    tmp202 = tl.load(in_ptr3 + (10660 + x7 + (5304*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp203 = tl.load(in_ptr1 + (31983 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp204 = tl.load(in_ptr1 + (31980 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp205 = tmp203 - tmp204
    tmp206 = tmp202 * tmp205
    tmp207 = tmp206 / tmp31
    tmp208 = tl.where(tmp25, tmp207, 0.0)
    tmp209 = tl.where(tmp25, tmp208, tmp9)
    tmp210 = tmp179 * tmp209
    tmp211 = tl.load(in_ptr5 + (31983 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp212 = tl.load(in_ptr5 + (31980 + (3*x7) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp213 = tmp211 - tmp212
    tmp214 = tmp202 * tmp213
    tmp215 = tmp214 / tmp31
    tmp216 = tl.where(tmp25, tmp215, 0.0)
    tmp217 = tl.where(tmp25, tmp216, tmp9)
    tmp218 = tmp181 * tmp217
    tmp219 = tmp210 + tmp218
    tl.store(out_ptr1 + (x4), tmp58, xmask)
    tl.store(out_ptr3 + (x4), tmp94, xmask)
    tl.store(out_ptr4 + (x4), tmp140, xmask)
    tl.store(out_ptr5 + (x4), tmp171, xmask)
    tl.store(out_ptr6 + (x4), tmp183, xmask)
    tl.store(out_ptr7 + (x4), tmp201, xmask)
    tl.store(out_ptr8 + (x4), tmp219, xmask)
''')
