

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/c2/cc25zgx7t2ka7p6xen2bbqeqvhvishwdxnvilkxcdjuljbsjqqyw.py
# Source Nodes: [abs_13, add_24, add_49, add_50, iadd_10, iadd_9, min_9, mul_125, mul_126, mul_127, mul_128, mul_129, mul_60, neg_22, neg_23, pow_3, setitem_19, sub_17, tanh_10, tensor, truediv_28, truediv_29], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_13 => abs_13
# add_24 => add_28
# add_49 => add_59
# add_50 => add_60
# iadd_10 => add_61, slice_scatter_150, slice_scatter_151
# iadd_9 => slice_scatter_145
# min_9 => minimum_8
# mul_125 => mul_125
# mul_126 => mul_126
# mul_127 => mul_127
# mul_128 => mul_128
# mul_129 => mul_129
# mul_60 => mul_60
# neg_22 => neg_22
# neg_23 => neg_23
# pow_3 => pow_3
# setitem_19 => copy_19, select_scatter_9, slice_scatter_81
# sub_17 => sub_17
# tanh_10 => tanh_10
# tensor => full_default_1
# truediv_28 => div_28
# truediv_29 => div_29
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_pow_select_scatter_slice_scatter_sub_tanh_12 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_pow_select_scatter_slice_scatter_sub_tanh_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: '*fp64', 10: '*fp64', 11: '*fp64', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_pow_select_scatter_slice_scatter_sub_tanh_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_pow_select_scatter_slice_scatter_sub_tanh_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x3 = (xindex // 26)
    x2 = (xindex // 5304)
    x5 = xindex
    x4 = xindex % 5304
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + (10608 + (26*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (10634 + (26*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0.5, tl.float64)
    tmp13 = tmp11 * tmp12
    tmp14 = 2 + x2
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tmp14 < tmp3
    tmp18 = tmp16 & tmp17
    tmp19 = tmp18 & tmp5
    tmp20 = tmp5 & tmp19
    tmp21 = tmp6 >= tmp1
    tmp22 = tmp21 & tmp20
    tmp23 = tl.load(in_ptr0 + (10608 + x5), tmp22 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr0 + (10607 + x5), tmp22 & xmask, other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr0 + (10634 + x5), tmp22 & xmask, other=0.0)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.load(in_ptr0 + (10633 + x5), tmp22 & xmask, other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0.25, tl.float64)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.where(tmp22, tmp31, 0.0)
    tmp33 = tl.full([1], 0.0, tl.float64)
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tl.where(tmp20, tmp34, 0.0)
    tmp36 = tl.where(tmp5, tmp35, tmp33)
    tmp37 = tl.where(tmp19, tmp36, 0.0)
    tmp38 = tl.where(tmp18, tmp37, tmp33)
    tmp39 = tl.where(tmp8, tmp13, tmp38)
    tmp40 = tl.where(tmp5, tmp39, 0.0)
    tmp41 = tmp5 & tmp18
    tmp42 = tmp21 & tmp41
    tmp43 = tl.load(in_ptr0 + (10608 + x5), tmp42 & xmask, other=0.0)
    tmp44 = tl.load(in_ptr0 + (10607 + x5), tmp42 & xmask, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr0 + (10634 + x5), tmp42 & xmask, other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr0 + (10633 + x5), tmp42 & xmask, other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tmp49 * tmp30
    tmp51 = tl.where(tmp42, tmp50, 0.0)
    tmp52 = tl.where(tmp21, tmp51, tmp33)
    tmp53 = tl.where(tmp41, tmp52, 0.0)
    tmp54 = tl.where(tmp5, tmp53, tmp33)
    tmp55 = tl.where(tmp18, tmp54, 0.0)
    tmp56 = tl.where(tmp18, tmp55, tmp33)
    tmp57 = tl.where(tmp5, tmp40, tmp56)
    tmp58 = tmp0 >= tmp15
    tmp59 = tmp58 & tmp4
    tmp60 = tl.load(in_ptr1 + ((-52) + x4 + (5200*x2)), tmp59 & xmask, other=0.0)
    tmp61 = tl.where(tmp59, tmp60, 0.0)
    tmp62 = tmp59 & tmp18
    tmp63 = tl.full([1], 25, tl.int64)
    tmp64 = tmp6 < tmp63
    tmp65 = tmp64 & tmp62
    tmp66 = tl.load(in_ptr2 + (1 + x2), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr0 + (10608 + x5), tmp65 & xmask, other=0.0)
    tmp68 = tmp66 * tmp67
    tmp69 = tl.load(in_ptr3 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp65 & xmask, other=0.0)
    tmp70 = tl.abs(tmp69)
    tmp71 = -tmp70
    tmp72 = tl.full([1], 0.001, tl.float64)
    tmp73 = tmp71 + tmp72
    tmp74 = tmp73 / tmp72
    tmp75 = libdevice.tanh(tmp74)
    tmp76 = tl.full([1], 1.0, tl.float64)
    tmp77 = tmp75 + tmp76
    tmp78 = tmp77 * tmp12
    tmp79 = tmp68 * tmp78
    tmp80 = tmp69 * tmp69
    tmp81 = tmp79 * tmp80
    tmp82 = tl.load(in_ptr4 + (10608 + x5), tmp65 & xmask, other=0.0)
    tmp83 = tmp81 * tmp82
    tmp84 = tmp33 + tmp83
    tmp85 = tl.where(tmp65, tmp84, 0.0)
    tmp86 = tl.where(tmp64, tmp85, tmp33)
    tmp87 = tl.where(tmp62, tmp86, 0.0)
    tmp88 = tl.where(tmp59, tmp87, tmp33)
    tmp89 = tl.where(tmp18, tmp88, 0.0)
    tmp90 = tl.where(tmp18, tmp89, tmp33)
    tmp91 = tl.where(tmp59, tmp61, tmp90)
    tmp92 = tmp64 & tmp59
    tmp93 = tl.load(in_ptr5 + ((-1) + x1), tmp92 & xmask, eviction_policy='evict_last', other=0.0)
    tmp94 = tl.load(in_ptr6 + ((-1) + x1), tmp92 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = tmp93 * tmp94
    tmp96 = tl.load(in_ptr0 + (10608 + x5), tmp92 & xmask, other=0.0)
    tmp97 = tmp95 * tmp96
    tmp98 = tl.load(in_ptr7 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp92 & xmask, other=0.0)
    tmp99 = -tmp98
    tmp100 = tl.load(in_ptr8 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp92 & xmask, other=0.0)
    tmp101 = triton_helpers.minimum(tmp33, tmp100)
    tmp102 = tl.full([1], 1e-20, tl.float64)
    tmp103 = tmp101 - tmp102
    tmp104 = tmp99 / tmp103
    tmp105 = tl.abs(tmp104)
    tmp106 = -tmp105
    tmp107 = tmp106 + tmp72
    tmp108 = tmp107 / tmp72
    tmp109 = libdevice.tanh(tmp108)
    tmp110 = tmp109 + tmp76
    tmp111 = tmp110 * tmp12
    tmp112 = tmp97 * tmp111
    tmp113 = tmp104 * tmp104
    tmp114 = tmp112 * tmp113
    tmp115 = tl.load(in_ptr4 + (10608 + x5), tmp92 & xmask, other=0.0)
    tmp116 = tmp114 * tmp115
    tmp117 = tmp33 + tmp116
    tmp118 = tl.where(tmp92, tmp117, 0.0)
    tmp119 = tl.where(tmp64, tmp118, tmp33)
    tmp120 = tl.where(tmp59, tmp119, 0.0)
    tmp121 = tl.where(tmp59, tmp120, tmp33)
    tl.store(out_ptr0 + (x5), tmp57, xmask)
    tl.store(out_ptr1 + (x5), tmp91, xmask)
    tl.store(out_ptr2 + (x5), tmp121, xmask)
''')
