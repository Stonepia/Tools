

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/es/cesqqzarepylh4dx3m753kkt2afv5vv6qoqtucux43mw377orfvn.py
# Source Nodes: [abs_17, add_62, add_63, iadd_14, min_13, mul_165, mul_166, mul_167, mul_168, mul_169, mul_182, neg_30, neg_31, pow_7, sub_21, tanh_14, tensor, truediv_36, truediv_37, truediv_40], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.sub, aten.tanh]
# abs_17 => abs_17
# add_62 => add_76
# add_63 => add_77
# iadd_14 => add_78
# min_13 => minimum_12
# mul_165 => mul_165
# mul_166 => mul_166
# mul_167 => mul_167
# mul_168 => mul_168
# mul_169 => mul_169
# mul_182 => mul_182
# neg_30 => neg_30
# neg_31 => neg_31
# pow_7 => pow_7
# sub_21 => sub_21
# tanh_14 => tanh_14
# tensor => full_default_1
# truediv_36 => div_36
# truediv_37 => div_37
# truediv_40 => div_40
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: '*fp64', 10: '*fp64', 11: '*fp64', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5000)
    x1 = (xindex // 25) % 200
    x0 = xindex % 25
    x4 = xindex
    x5 = (xindex // 25)
    tmp43 = tl.load(in_ptr0 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp45 = tl.load(in_ptr4 + (2 + x2), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr7 + (1 + x1), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr8 + (1 + x1), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr2 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp67 = tl.load(in_out_ptr1 + (x4), xmask)
    tmp69 = tl.load(in_ptr9 + (x4), xmask)
    tmp84 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), xmask)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2 + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = x0
    tmp12 = tl.full([1], 25, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + (10660 + x0 + (26*x1) + (5304*x2)), tmp14 & xmask, other=0.0)
    tmp16 = tl.load(in_ptr1 + (2 + x2), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr2 + (10660 + x0 + (26*x1) + (5304*x2)), tmp14 & xmask, other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_out_ptr0 + (x4), tmp14 & xmask, other=0.0)
    tmp20 = tl.abs(tmp19)
    tmp21 = -tmp20
    tmp22 = tl.full([1], 0.001, tl.float64)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 / tmp22
    tmp25 = libdevice.tanh(tmp24)
    tmp26 = tl.full([1], 1.0, tl.float64)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full([1], 0.5, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp18 * tmp29
    tmp31 = tmp19 * tmp19
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), tmp14 & xmask, other=0.0)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp15 + tmp34
    tmp36 = tl.where(tmp14, tmp35, 0.0)
    tmp37 = tl.load(in_ptr0 + (10660 + x0 + (26*x1) + (5304*x2)), tmp10 & xmask, other=0.0)
    tmp38 = tl.where(tmp13, tmp36, tmp37)
    tmp39 = tl.where(tmp10, tmp38, 0.0)
    tmp40 = tl.load(in_ptr0 + (10660 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp41 = tl.where(tmp9, tmp39, tmp40)
    tmp42 = tl.where(tmp5, tmp41, 0.0)
    tmp44 = tl.where(tmp5, tmp42, tmp43)
    tmp46 = tl.full([1], 4.0, tl.float64)
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 / tmp47
    tmp49 = tl.load(in_ptr5 + (x0 + (26*x5)), tmp10 & xmask, other=0.0)
    tmp50 = tl.where(tmp10, tmp49, 0.0)
    tmp51 = tmp5 & tmp5
    tmp52 = tl.load(in_ptr6 + (52 + x0 + (26*x1) + (5304*x2)), tmp51 & xmask, other=0.0)
    tmp53 = tl.where(tmp51, tmp52, 0.0)
    tmp54 = tl.full([1], 0.0, tl.float64)
    tmp55 = tl.where(tmp5, tmp53, tmp54)
    tmp56 = tl.where(tmp9, tmp50, tmp55)
    tmp57 = tl.where(tmp5, tmp56, 0.0)
    tmp58 = tl.load(in_ptr6 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp59 = tl.where(tmp5, tmp58, 0.0)
    tmp60 = tl.where(tmp5, tmp59, tmp54)
    tmp61 = tl.where(tmp5, tmp57, tmp60)
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 * tmp65
    tmp68 = -tmp67
    tmp70 = triton_helpers.minimum(tmp54, tmp69)
    tmp71 = tl.full([1], 1e-20, tl.float64)
    tmp72 = tmp70 - tmp71
    tmp73 = tmp68 / tmp72
    tmp74 = tl.abs(tmp73)
    tmp75 = -tmp74
    tmp76 = tmp75 + tmp22
    tmp77 = tmp76 / tmp22
    tmp78 = libdevice.tanh(tmp77)
    tmp79 = tmp78 + tmp26
    tmp80 = tmp79 * tmp28
    tmp81 = tmp66 * tmp80
    tmp82 = tmp73 * tmp73
    tmp83 = tmp81 * tmp82
    tmp85 = tmp83 * tmp84
    tmp86 = tmp61 + tmp85
    tl.store(in_out_ptr0 + (x4), tmp48, xmask)
    tl.store(in_out_ptr1 + (x4), tmp86, xmask)
''')
