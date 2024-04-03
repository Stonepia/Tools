

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/4p/c4pzfp334vuvpqphwp6yywyz2u45qmguow4m62ecfqxekodvbxgq.py
# Source Nodes: [add_10, add_25, add_26, add_29, add_30, add_5, add_6, add_9, min_1, min_2, mul_18, mul_19, mul_20, mul_21, mul_28, mul_29, mul_30, mul_31, mul_61, mul_62, mul_63, mul_64, mul_71, mul_72, mul_73, mul_74, neg_2, neg_4, sub_10, sub_9, tensor, truediv_6, truediv_8], Original ATen: [aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.sub]
# add_10 => add_11
# add_25 => add_29
# add_26 => add_30
# add_29 => add_34
# add_30 => add_35
# add_5 => add_5
# add_6 => add_6
# add_9 => add_10
# min_1 => minimum
# min_2 => minimum_1
# mul_18 => mul_18
# mul_19 => mul_19
# mul_20 => mul_20
# mul_21 => mul_21
# mul_28 => mul_28
# mul_29 => mul_29
# mul_30 => mul_30
# mul_31 => mul_31
# mul_61 => mul_61
# mul_62 => mul_62
# mul_63 => mul_63
# mul_64 => mul_64
# mul_71 => mul_71
# mul_72 => mul_72
# mul_73 => mul_73
# mul_74 => mul_74
# neg_2 => neg_2
# neg_4 => neg_4
# sub_10 => sub_10
# sub_9 => sub_9
# tensor => full_default_1
# truediv_6 => div_6
# truediv_8 => div_8
triton_poi_fused_add_div_lift_fresh_minimum_mul_neg_sub_1 = async_compile.triton('triton_poi_fused_add_div_lift_fresh_minimum_mul_neg_sub_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_lift_fresh_minimum_mul_neg_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_lift_fresh_minimum_mul_neg_sub_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1005000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x2 = (xindex // 5000)
    x4 = xindex
    x5 = (xindex // 25) % 201
    x6 = (xindex // 5025)
    tmp0 = tl.load(in_ptr0 + (5357 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (16071 + (3*x0) + (78*x1) + (15912*x2)), xmask)
    tmp6 = tl.load(in_ptr2 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr6 + (5357 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp49 = tl.load(in_ptr7 + (5357 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp57 = tl.load(in_ptr0 + (10661 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp58 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), xmask)
    tmp91 = tl.load(in_ptr0 + (10635 + x0 + (26*x5) + (5304*x6)), xmask)
    tmp92 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x5) + (15912*x6)), xmask)
    tmp122 = tl.load(in_ptr0 + (10661 + x0 + (26*x5) + (5304*x6)), xmask)
    tmp123 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x5) + (15912*x6)), xmask)
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
    tmp26 = tl.load(in_ptr3 + (5356 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr1 + (16071 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr1 + (16068 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = tl.load(in_ptr4 + (x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 / tmp31
    tmp33 = tl.where(tmp25, tmp32, 0.0)
    tmp34 = tl.where(tmp25, tmp33, tmp9)
    tmp35 = tmp22 * tmp34
    tmp36 = 0.7987200021743774
    tmp37 = tmp0 * tmp36
    tmp38 = tl.load(in_ptr5 + (16071 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp39 = tl.load(in_ptr5 + (16068 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
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
    tmp53 = triton_helpers.minimum(tmp46, tmp9)
    tmp54 = 1e-20
    tmp55 = tmp53 - tmp54
    tmp56 = tmp52 / tmp55
    tmp59 = tmp58 - tmp2
    tmp60 = tmp59 * tmp4
    tmp61 = tmp60 + tmp18
    tmp62 = -tmp61
    tmp63 = tmp62 * tmp13
    tmp64 = tmp57 * tmp63
    tmp65 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0)
    tmp66 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp67 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp68 = tmp66 - tmp67
    tmp69 = tmp65 * tmp68
    tmp70 = tmp69 / tmp31
    tmp71 = tl.where(tmp25, tmp70, 0.0)
    tmp72 = tl.where(tmp25, tmp71, tmp9)
    tmp73 = tmp64 * tmp72
    tmp74 = tmp57 * tmp36
    tmp75 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp76 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tmp65 * tmp77
    tmp79 = tmp78 / tmp31
    tmp80 = tl.where(tmp25, tmp79, 0.0)
    tmp81 = tl.where(tmp25, tmp80, tmp9)
    tmp82 = tmp74 * tmp81
    tmp83 = tmp73 + tmp82
    tmp84 = tmp64 * tmp47
    tmp85 = tmp74 * tmp49
    tmp86 = tmp84 + tmp85
    tmp87 = -tmp86
    tmp88 = triton_helpers.minimum(tmp83, tmp9)
    tmp89 = tmp88 - tmp54
    tmp90 = tmp87 / tmp89
    tmp93 = tmp92 - tmp2
    tmp94 = tmp93 * tmp4
    tmp95 = tmp94 + tmp18
    tmp96 = -tmp95
    tmp97 = tmp96 * tmp13
    tmp98 = tmp91 * tmp97
    tmp99 = 1 + x5
    tmp100 = tl.full([1], 203, tl.int64)
    tmp101 = tmp99 < tmp100
    tmp102 = tl.load(in_ptr8 + (10635 + x0 + (26*x5) + (5304*x6)), tmp101 & xmask, other=0.0)
    tmp103 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp101 & xmask, other=0.0)
    tmp104 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp101 & xmask, other=0.0)
    tmp105 = tmp103 - tmp104
    tmp106 = tmp102 * tmp105
    tmp107 = tl.load(in_ptr9 + (1 + x5), tmp101 & xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tmp106 / tmp107
    tmp109 = tl.where(tmp101, tmp108, 0.0)
    tmp110 = tl.where(tmp101, tmp109, tmp9)
    tmp111 = tmp98 * tmp110
    tmp112 = tmp91 * tmp36
    tmp113 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp101 & xmask, other=0.0)
    tmp114 = tl.load(in_ptr5 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp101 & xmask, other=0.0)
    tmp115 = tmp113 - tmp114
    tmp116 = tmp102 * tmp115
    tmp117 = tmp116 / tmp107
    tmp118 = tl.where(tmp101, tmp117, 0.0)
    tmp119 = tl.where(tmp101, tmp118, tmp9)
    tmp120 = tmp112 * tmp119
    tmp121 = tmp111 + tmp120
    tmp124 = tmp123 - tmp2
    tmp125 = tmp124 * tmp4
    tmp126 = tmp125 + tmp18
    tmp127 = -tmp126
    tmp128 = tmp127 * tmp13
    tmp129 = tmp122 * tmp128
    tmp130 = tmp129 * tmp110
    tmp131 = tmp122 * tmp36
    tmp132 = tmp131 * tmp119
    tmp133 = tmp130 + tmp132
    tmp134 = tl.load(in_ptr3 + (10634 + x0 + (26*x5) + (5304*x6)), tmp25 & xmask, other=0.0)
    tmp135 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp136 = tl.load(in_ptr1 + (31902 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tmp134 * tmp137
    tmp139 = tmp138 / tmp31
    tmp140 = tl.where(tmp25, tmp139, 0.0)
    tmp141 = tl.where(tmp25, tmp140, tmp9)
    tmp142 = tmp98 * tmp141
    tmp143 = tl.load(in_ptr5 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp144 = tl.load(in_ptr5 + (31902 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp145 = tmp143 - tmp144
    tmp146 = tmp134 * tmp145
    tmp147 = tmp146 / tmp31
    tmp148 = tl.where(tmp25, tmp147, 0.0)
    tmp149 = tl.where(tmp25, tmp148, tmp9)
    tmp150 = tmp112 * tmp149
    tmp151 = tmp142 + tmp150
    tmp152 = tl.load(in_ptr3 + (10660 + x0 + (26*x5) + (5304*x6)), tmp25 & xmask, other=0.0)
    tmp153 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp154 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp155 = tmp153 - tmp154
    tmp156 = tmp152 * tmp155
    tmp157 = tmp156 / tmp31
    tmp158 = tl.where(tmp25, tmp157, 0.0)
    tmp159 = tl.where(tmp25, tmp158, tmp9)
    tmp160 = tmp129 * tmp159
    tmp161 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp162 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tmp152 * tmp163
    tmp165 = tmp164 / tmp31
    tmp166 = tl.where(tmp25, tmp165, 0.0)
    tmp167 = tl.where(tmp25, tmp166, tmp9)
    tmp168 = tmp131 * tmp167
    tmp169 = tmp160 + tmp168
    tl.store(in_out_ptr0 + (x4), tmp56, xmask)
    tl.store(in_out_ptr1 + (x4), tmp90, xmask)
    tl.store(out_ptr0 + (x4), tmp121, xmask)
    tl.store(out_ptr1 + (x4), tmp133, xmask)
    tl.store(out_ptr2 + (x4), tmp151, xmask)
    tl.store(out_ptr3 + (x4), tmp169, xmask)
''')
