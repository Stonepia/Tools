

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/vg/cvgbeujxnzyf6pb674sakxvr2ogggqwwwweyqw4uligtx5mltojf.py
# Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, and_, and__1, eq, ge_1, iadd, mul, mul_1, mul_13, mul_2, mul_3, mul_4, mul_6, mul_7, mul_8, mul_9, neg, neg_1, neg_2, setitem, setitem_1, setitem_2, setitem_3, setitem_4, setitem_5, setitem_6, truediv, truediv_1, truediv_10, truediv_2, truediv_3, truediv_4, truediv_5, truediv_6, truediv_7, truediv_8, truediv_9, where, where_1, zeros_like_3], Original ATen: [aten.add, aten.bitwise_and, aten.copy, aten.div, aten.eq, aten.ge, aten.mul, aten.neg, aten.reciprocal, aten.scalar_tensor, aten.select_scatter, aten.slice, aten.slice_scatter, aten.where, aten.zeros_like]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# add_4 => add_4
# add_5 => add_5
# add_6 => add_6
# add_7 => add_7
# add_8 => add_8
# and_ => bitwise_and
# and__1 => bitwise_and_1
# eq => eq
# ge_1 => ge_1
# iadd => add_9, select_scatter_2, slice_133, slice_134, slice_scatter_16, slice_scatter_17
# mul => mul_1
# mul_1 => mul_2
# mul_13 => mul_16
# mul_2 => mul_3
# mul_3 => mul_4
# mul_4 => mul_5
# mul_6 => mul_8
# mul_7 => mul_10
# mul_8 => mul_11
# mul_9 => mul_12
# neg => neg
# neg_1 => neg_1
# neg_2 => neg_2
# setitem => copy, slice_23, slice_24, slice_scatter, slice_scatter_1, slice_scatter_2
# setitem_1 => copy_1, slice_scatter_3, slice_scatter_4, slice_scatter_5
# setitem_2 => copy_2, select_scatter, slice_52, slice_53
# setitem_3 => copy_3, slice_scatter_10, slice_scatter_8, slice_scatter_9
# setitem_4 => copy_4, select_scatter_1, slice_95, slice_96, slice_scatter_11, slice_scatter_12
# setitem_5 => copy_5, slice_scatter_13, slice_scatter_14, slice_scatter_15
# setitem_6 => copy_6
# truediv => mul, reciprocal
# truediv_1 => div
# truediv_10 => div_7
# truediv_2 => div_1
# truediv_3 => div_2
# truediv_4 => div_3
# truediv_5 => div_4
# truediv_6 => mul_7, reciprocal_1
# truediv_7 => div_5
# truediv_8 => mul_9, reciprocal_2
# truediv_9 => div_6
# where => full_default_2, where
# where_1 => where_1
# zeros_like_3 => full_3
triton_poi_fused_add_bitwise_and_copy_div_eq_ge_mul_neg_reciprocal_scalar_tensor_select_scatter_slice_slice_scatter_where_zeros_like_1 = async_compile.triton('triton_poi_fused_add_bitwise_and_copy_div_eq_ge_mul_neg_reciprocal_scalar_tensor_select_scatter_slice_slice_scatter_where_zeros_like_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_copy_div_eq_ge_mul_neg_reciprocal_scalar_tensor_select_scatter_slice_slice_scatter_where_zeros_like_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_bitwise_and_copy_div_eq_ge_mul_neg_reciprocal_scalar_tensor_select_scatter_slice_slice_scatter_where_zeros_like_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26) % 200
    x2 = (xindex // 5200)
    x3 = xindex % 5200
    x4 = xindex
    x5 = (xindex // 26)
    tmp3 = tl.load(in_ptr0 + (32055 + (78*x1) + (15912*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (10685 + (26*x1) + (5304*x2)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (410 + x1 + (204*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (25))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (31980 + (3*x3) + (15912*x2)), xmask)
    tmp17 = tl.load(in_ptr1 + (10660 + x3 + (5304*x2)), xmask)
    tmp24 = tl.load(in_ptr4 + (25))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp88 = tl.load(in_ptr7 + (410 + x1 + (204*x2)), xmask, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr6 + (10660 + x3 + (5304*x2)), xmask)
    tmp117 = tl.load(in_ptr8 + (x5), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 25, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 * tmp5
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 / tmp13
    tmp15 = tmp7 + tmp14
    tmp18 = tmp17 * tmp5
    tmp19 = tmp16 + tmp18
    tmp20 = tl.where(tmp2, tmp15, tmp19)
    tmp21 = tl.full([1], 24, tl.int64)
    tmp22 = tl.full([1], 25, tl.int64)
    tmp23 = tmp21 < tmp22
    tmp26 = 1 / tmp25
    tmp27 = tmp26 * tmp5
    tmp28 = tmp27 * tmp5
    tmp29 = tmp28 * tmp12
    tmp30 = tl.load(in_ptr5 + (10684 + (26*x1) + (5304*x2)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr5 + (10685 + (26*x1) + (5304*x2)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp29 * tmp32
    tmp34 = tl.where(tmp23, tmp33, 0.0)
    tmp35 = 0.0
    tmp36 = tl.where(tmp23, tmp34, tmp35)
    tmp37 = -tmp36
    tmp38 = tmp37 / tmp13
    tmp39 = tl.full([1], 1, tl.int64)
    tmp40 = tmp0 >= tmp39
    tmp41 = tmp0 < tmp22
    tmp42 = tmp40 & tmp41
    tmp43 = (-1) + x0
    tmp44 = tmp43 < tmp22
    tmp45 = tmp44 & tmp42
    tmp46 = tl.load(in_ptr4 + (x0), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = 1 / tmp46
    tmp48 = tmp47 * tmp5
    tmp49 = tmp48 * tmp5
    tmp50 = tmp49 * tmp12
    tmp51 = tl.load(in_ptr5 + (10659 + x3 + (5304*x2)), tmp45 & xmask, other=0.0)
    tmp52 = tl.load(in_ptr5 + (10660 + x3 + (5304*x2)), tmp45 & xmask, other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp50 * tmp53
    tmp55 = tl.where(tmp45, tmp54, 0.0)
    tmp56 = tl.where(tmp44, tmp55, tmp35)
    tmp57 = -tmp56
    tmp58 = tl.load(in_ptr3 + (x0), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 / tmp58
    tmp60 = tl.where(tmp42, tmp59, 0.0)
    tmp61 = tl.where(tmp42, tmp60, tmp35)
    tmp62 = tl.where(tmp2, tmp38, tmp61)
    tmp63 = tmp41 & tmp42
    tmp64 = tl.load(in_ptr4 + (1 + x0), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = 1 / tmp64
    tmp66 = tmp65 * tmp5
    tmp67 = tmp66 * tmp5
    tmp68 = tmp67 * tmp12
    tmp69 = tl.load(in_ptr5 + (10660 + x3 + (5304*x2)), tmp63 & xmask, other=0.0)
    tmp70 = tl.load(in_ptr5 + (10661 + x3 + (5304*x2)), tmp63 & xmask, other=0.0)
    tmp71 = tmp69 + tmp70
    tmp72 = tmp68 * tmp71
    tmp73 = tl.where(tmp63, tmp72, 0.0)
    tmp74 = tl.where(tmp41, tmp73, tmp35)
    tmp75 = tmp74 + tmp56
    tmp76 = tmp75 / tmp58
    tmp77 = tmp76 + tmp5
    tmp78 = tl.load(in_ptr0 + (31980 + (3*x3) + (15912*x2)), tmp42 & xmask, other=0.0)
    tmp79 = triton_helpers.maximum(tmp35, tmp78)
    tmp80 = tl.sqrt(tmp79)
    tmp81 = 0.7
    tmp82 = tmp80 * tmp81
    tmp83 = tl.load(in_ptr6 + (10660 + x3 + (5304*x2)), tmp42 & xmask, other=0.0)
    tmp84 = tmp82 / tmp83
    tmp85 = tmp77 + tmp84
    tmp86 = tl.where(tmp42, tmp85, 0.0)
    tmp87 = tl.where(tmp42, tmp86, tmp35)
    tmp89 = tmp88 - tmp39
    tmp90 = tl.full([1], 0, tl.int64)
    tmp91 = tmp89 >= tmp90
    tmp92 = tmp0 == tmp89
    tmp93 = tmp91 & tmp92
    tmp94 = tl.load(in_ptr4 + (1 + x0), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = 1 / tmp94
    tmp96 = tmp95 * tmp5
    tmp97 = tmp96 * tmp5
    tmp98 = tmp97 * tmp12
    tmp99 = tl.load(in_ptr5 + (10660 + x3 + (5304*x2)), tmp41 & xmask, other=0.0)
    tmp100 = tl.load(in_ptr5 + (10661 + x3 + (5304*x2)), tmp41 & xmask, other=0.0)
    tmp101 = tmp99 + tmp100
    tmp102 = tmp98 * tmp101
    tmp103 = tl.where(tmp41, tmp102, 0.0)
    tmp104 = tl.where(tmp41, tmp103, tmp35)
    tmp106 = tmp104 / tmp105
    tmp107 = tmp106 + tmp5
    tmp109 = 1 / tmp108
    tmp110 = tmp109 * tmp81
    tmp111 = triton_helpers.maximum(tmp35, tmp16)
    tmp112 = tl.sqrt(tmp111)
    tmp113 = tmp110 * tmp112
    tmp114 = tmp107 + tmp113
    tmp115 = tmp0 >= tmp89
    tmp116 = tmp91 & tmp115
    tmp118 = tl.where(tmp2, tmp117, tmp87)
    tmp119 = tl.where(tmp116, tmp118, tmp5)
    tmp120 = tl.where(tmp93, tmp114, tmp119)
    tmp121 = tmp116.to(tl.float32)
    tmp122 = tmp41 & tmp41
    tmp123 = tl.load(in_ptr4 + (1 + x0), tmp122 & xmask, eviction_policy='evict_last', other=0.0)
    tmp124 = 1 / tmp123
    tmp125 = tmp124 * tmp5
    tmp126 = tmp125 * tmp5
    tmp127 = tmp126 * tmp12
    tmp128 = tl.load(in_ptr5 + (10660 + x3 + (5304*x2)), tmp122 & xmask, other=0.0)
    tmp129 = tl.load(in_ptr5 + (10661 + x3 + (5304*x2)), tmp122 & xmask, other=0.0)
    tmp130 = tmp128 + tmp129
    tmp131 = tmp127 * tmp130
    tmp132 = tl.where(tmp122, tmp131, 0.0)
    tmp133 = tl.where(tmp41, tmp132, tmp35)
    tmp134 = -tmp133
    tmp135 = tl.load(in_ptr3 + (x0), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tmp134 / tmp135
    tmp137 = tl.where(tmp41, tmp136, 0.0)
    tmp138 = tl.where(tmp41, tmp137, tmp35)
    tmp139 = tmp121 * tmp138
    tl.store(out_ptr0 + (x4), tmp20, xmask)
    tl.store(out_ptr1 + (x4), tmp62, xmask)
    tl.store(in_out_ptr0 + (x4), tmp120, xmask)
    tl.store(out_ptr2 + (x4), tmp139, xmask)
''')
