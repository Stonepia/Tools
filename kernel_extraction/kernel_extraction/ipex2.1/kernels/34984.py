

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ec/cechpitbbphxmyo7amu6gmnkld67oh6yqabsfswbeqhjnmb76tvm.py
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

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*fp32', 11: '*fp32', 12: '*bf16', 13: '*bf16', 14: '*bf16', 15: '*bf16', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_lift_fresh_minimum_mul_neg_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_lift_fresh_minimum_mul_neg_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (5357 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (16071 + (3*x0) + (78*x1) + (15912*x2)), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (1 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tl.load(in_ptr6 + (5357 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp49 = tl.load(in_ptr7 + (5357 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp59 = tl.load(in_ptr0 + (10661 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp60 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), xmask).to(tl.float32)
    tmp95 = tl.load(in_ptr0 + (10635 + x0 + (26*x5) + (5304*x6)), xmask).to(tl.float32)
    tmp96 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x5) + (15912*x6)), xmask).to(tl.float32)
    tmp126 = tl.load(in_ptr0 + (10661 + x0 + (26*x5) + (5304*x6)), xmask).to(tl.float32)
    tmp127 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x5) + (15912*x6)), xmask).to(tl.float32)
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
    tmp26 = tl.load(in_ptr3 + (5356 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (16071 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (16068 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp29 = tmp27 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = tl.load(in_ptr4 + (x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 / tmp31
    tmp33 = tl.where(tmp25, tmp32, 0.0)
    tmp34 = tl.where(tmp25, tmp33, tmp9)
    tmp35 = tmp22 * tmp34
    tmp36 = 0.796875
    tmp37 = tmp0 * tmp36
    tmp38 = tl.load(in_ptr5 + (16071 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr5 + (16068 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
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
    tmp67 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp68 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp69 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp70 = tmp68 - tmp69
    tmp71 = tmp67 * tmp70
    tmp72 = tmp71 / tmp31
    tmp73 = tl.where(tmp25, tmp72, 0.0)
    tmp74 = tl.where(tmp25, tmp73, tmp9)
    tmp75 = tmp66 * tmp74
    tmp76 = tmp59 * tmp36
    tmp77 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp78 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x1) + (15912*x2)), tmp25 & xmask, other=0.0).to(tl.float32)
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
    tmp97 = tmp96 - tmp2
    tmp98 = tmp97 * tmp4
    tmp99 = tmp98 + tmp18
    tmp100 = -tmp99
    tmp101 = tmp100 * tmp13
    tmp102 = tmp95 * tmp101
    tmp103 = 1 + x5
    tmp104 = tl.full([1], 203, tl.int64)
    tmp105 = tmp103 < tmp104
    tmp106 = tl.load(in_ptr8 + (10635 + x0 + (26*x5) + (5304*x6)), tmp105 & xmask, other=0.0).to(tl.float32)
    tmp107 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp105 & xmask, other=0.0).to(tl.float32)
    tmp108 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp105 & xmask, other=0.0).to(tl.float32)
    tmp109 = tmp107 - tmp108
    tmp110 = tmp106 * tmp109
    tmp111 = tl.load(in_ptr9 + (1 + x5), tmp105 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp112 = tmp110 / tmp111
    tmp113 = tl.where(tmp105, tmp112, 0.0)
    tmp114 = tl.where(tmp105, tmp113, tmp9)
    tmp115 = tmp102 * tmp114
    tmp116 = tmp95 * tmp36
    tmp117 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp105 & xmask, other=0.0).to(tl.float32)
    tmp118 = tl.load(in_ptr5 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp105 & xmask, other=0.0).to(tl.float32)
    tmp119 = tmp117 - tmp118
    tmp120 = tmp106 * tmp119
    tmp121 = tmp120 / tmp111
    tmp122 = tl.where(tmp105, tmp121, 0.0)
    tmp123 = tl.where(tmp105, tmp122, tmp9)
    tmp124 = tmp116 * tmp123
    tmp125 = tmp115 + tmp124
    tmp128 = tmp127 - tmp2
    tmp129 = tmp128 * tmp4
    tmp130 = tmp129 + tmp18
    tmp131 = -tmp130
    tmp132 = tmp131 * tmp13
    tmp133 = tmp126 * tmp132
    tmp134 = tmp133 * tmp114
    tmp135 = tmp126 * tmp36
    tmp136 = tmp135 * tmp123
    tmp137 = tmp134 + tmp136
    tmp138 = tl.load(in_ptr3 + (10634 + x0 + (26*x5) + (5304*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp139 = tl.load(in_ptr1 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp140 = tl.load(in_ptr1 + (31902 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp141 = tmp139 - tmp140
    tmp142 = tmp138 * tmp141
    tmp143 = tmp142 / tmp31
    tmp144 = tl.where(tmp25, tmp143, 0.0)
    tmp145 = tl.where(tmp25, tmp144, tmp9)
    tmp146 = tmp102 * tmp145
    tmp147 = tl.load(in_ptr5 + (31905 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp148 = tl.load(in_ptr5 + (31902 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp149 = tmp147 - tmp148
    tmp150 = tmp138 * tmp149
    tmp151 = tmp150 / tmp31
    tmp152 = tl.where(tmp25, tmp151, 0.0)
    tmp153 = tl.where(tmp25, tmp152, tmp9)
    tmp154 = tmp116 * tmp153
    tmp155 = tmp146 + tmp154
    tmp156 = tl.load(in_ptr3 + (10660 + x0 + (26*x5) + (5304*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp157 = tl.load(in_ptr1 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp158 = tl.load(in_ptr1 + (31980 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp159 = tmp157 - tmp158
    tmp160 = tmp156 * tmp159
    tmp161 = tmp160 / tmp31
    tmp162 = tl.where(tmp25, tmp161, 0.0)
    tmp163 = tl.where(tmp25, tmp162, tmp9)
    tmp164 = tmp133 * tmp163
    tmp165 = tl.load(in_ptr5 + (31983 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp166 = tl.load(in_ptr5 + (31980 + (3*x0) + (78*x5) + (15912*x6)), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp167 = tmp165 - tmp166
    tmp168 = tmp156 * tmp167
    tmp169 = tmp168 / tmp31
    tmp170 = tl.where(tmp25, tmp169, 0.0)
    tmp171 = tl.where(tmp25, tmp170, tmp9)
    tmp172 = tmp135 * tmp171
    tmp173 = tmp164 + tmp172
    tl.store(out_ptr1 + (x4), tmp58, xmask)
    tl.store(out_ptr3 + (x4), tmp94, xmask)
    tl.store(out_ptr4 + (x4), tmp125, xmask)
    tl.store(out_ptr5 + (x4), tmp137, xmask)
    tl.store(out_ptr6 + (x4), tmp155, xmask)
    tl.store(out_ptr7 + (x4), tmp173, xmask)
''')
