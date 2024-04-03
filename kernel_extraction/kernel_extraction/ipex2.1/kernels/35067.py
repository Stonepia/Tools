

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/qx/cqxai3ukcn3bkllt4jzihvzd57tac2bkk6dtmro4vqfewmcvwpee.py
# Source Nodes: [abs_9, add_35, add_36, iadd_5, iadd_6, max_7, min_7, mul_85, mul_86, mul_87, mul_88, neg_14, neg_15, setitem_20, setitem_22, sub_15, tanh_6, tensor, tensor_1, truediv_19, truediv_20], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.slice_scatter, aten.sub, aten.tanh]
# abs_9 => abs_9
# add_35 => add_41
# add_36 => add_42
# iadd_5 => slice_scatter_100
# iadd_6 => add_43, slice_scatter_109, slice_scatter_110, slice_scatter_111, slice_scatter_112, slice_scatter_113
# max_7 => maximum_6
# min_7 => minimum_6
# mul_85 => mul_85
# mul_86 => mul_86
# mul_87 => mul_87
# mul_88 => mul_88
# neg_14 => neg_14
# neg_15 => neg_15
# setitem_20 => copy_20, slice_scatter_88, slice_scatter_89, slice_scatter_90, slice_scatter_91, slice_scatter_92
# setitem_22 => slice_scatter_102, slice_scatter_103, slice_scatter_104, slice_scatter_105
# sub_15 => sub_15
# tanh_6 => tanh_6
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_19 => div_19
# truediv_20 => div_20
triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_31 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_minimum_mul_neg_slice_scatter_sub_tanh_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x1 = (xindex // 26) % 204
    x3 = xindex % 5304
    x4 = xindex
    x0 = xindex % 26
    tmp44 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + ((-10478) + x3 + (5226*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.where(tmp11, tmp12, 0.0)
    tmp14 = tmp5 & tmp5
    tmp15 = tl.load(in_ptr1 + ((-10608) + x4), tmp14 & xmask, other=0.0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tmp10 & tmp14
    tmp18 = x0
    tmp19 = tmp18 >= tmp7
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_out_ptr0 + (x4), tmp20 & xmask, other=0.0)
    tmp22 = tl.where(tmp20, tmp21, 0.0)
    tmp23 = tl.load(in_out_ptr0 + (x4), tmp17 & xmask, other=0.0)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp17, tmp24, 0.0)
    tmp26 = tl.load(in_out_ptr0 + (x4), tmp14 & xmask, other=0.0)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tl.where(tmp14, tmp27, 0.0)
    tmp29 = tl.load(in_out_ptr0 + (x4), tmp5 & xmask, other=0.0)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tl.where(tmp5, tmp16, tmp30)
    tmp32 = tl.where(tmp10, tmp13, tmp31)
    tmp33 = tl.where(tmp5, tmp32, 0.0)
    tmp34 = tl.load(in_ptr1 + ((-10608) + x4), tmp5 & xmask, other=0.0)
    tmp35 = tl.where(tmp5, tmp34, 0.0)
    tmp36 = tmp19 & tmp11
    tmp37 = tl.load(in_out_ptr0 + (x4), tmp36 & xmask, other=0.0)
    tmp38 = tl.where(tmp36, tmp37, 0.0)
    tmp39 = tl.load(in_out_ptr0 + (x4), tmp11 & xmask, other=0.0)
    tmp40 = tl.where(tmp19, tmp38, tmp39)
    tmp41 = tl.where(tmp11, tmp40, 0.0)
    tmp42 = tl.where(tmp10, tmp41, tmp29)
    tmp43 = tl.where(tmp5, tmp42, 0.0)
    tmp45 = tl.where(tmp5, tmp43, tmp44)
    tmp46 = tl.where(tmp5, tmp35, tmp45)
    tmp47 = tl.where(tmp5, tmp33, tmp46)
    tmp48 = tl.load(in_ptr2 + (x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.load(in_ptr3 + (x4), tmp11 & xmask, other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tl.load(in_ptr4 + (x4), tmp11 & xmask, other=0.0)
    tmp52 = tl.load(in_ptr5 + ((-10478) + x3 + (5226*x2)), tmp11 & xmask, other=0.0)
    tmp53 = -tmp52
    tmp54 = tl.load(in_ptr6 + ((-10478) + x3 + (5226*x2)), tmp11 & xmask, other=0.0)
    tmp55 = tl.full([1], 0.0, tl.float64)
    tmp56 = triton_helpers.minimum(tmp55, tmp54)
    tmp57 = tl.full([1], 1e-20, tl.float64)
    tmp58 = tmp56 - tmp57
    tmp59 = tmp53 / tmp58
    tmp60 = tl.abs(tmp59)
    tmp61 = -tmp60
    tmp62 = tl.full([1], 0.001, tl.float64)
    tmp63 = tmp61 + tmp62
    tmp64 = tmp63 / tmp62
    tmp65 = libdevice.tanh(tmp64)
    tmp66 = tl.full([1], 1.0, tl.float64)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.full([1], 0.5, tl.float64)
    tmp69 = tmp67 * tmp68
    tmp70 = tmp51 * tmp69
    tmp71 = tl.full([1], 50.0, tl.float64)
    tmp72 = triton_helpers.maximum(tmp71, tmp70)
    tmp73 = tmp50 * tmp72
    tmp74 = tmp47 + tmp73
    tmp75 = tl.where(tmp11, tmp74, 0.0)
    tmp76 = tl.where(tmp10, tmp75, tmp47)
    tmp77 = tl.where(tmp5, tmp76, 0.0)
    tmp78 = tl.where(tmp5, tmp77, tmp47)
    tl.store(in_out_ptr0 + (x4), tmp78, xmask)
''')
