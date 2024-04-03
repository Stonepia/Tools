

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/xz/cxznvvsfuxaknliy5zabwat7gswzm75txrrvqhhpu36hdnzvcphi.py
# Source Nodes: [abs_3, abs_4, add_11, add_12, add_7, add_8, iadd, iadd_1, max_1, max_2, mul_22, mul_23, mul_24, mul_25, mul_32, mul_34, mul_35, neg_3, neg_5, setitem_8, tanh, tanh_1, tensor_1, truediv_7, truediv_9, zeros_like], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.maximum, aten.mul, aten.neg, aten.slice_scatter, aten.tanh, aten.zeros_like]
# abs_3 => abs_3
# abs_4 => abs_4
# add_11 => add_12
# add_12 => add_13
# add_7 => add_7
# add_8 => add_8
# iadd => add_9, slice_scatter_23, slice_scatter_24, slice_scatter_25, slice_scatter_26, slice_scatter_27
# iadd_1 => add_14, slice_scatter_36, slice_scatter_37, slice_scatter_38, slice_scatter_39, slice_scatter_40
# max_1 => maximum
# max_2 => maximum_1
# mul_22 => mul_22
# mul_23 => mul_23
# mul_24 => mul_24
# mul_25 => mul_25
# mul_32 => mul_32
# mul_34 => mul_34
# mul_35 => mul_35
# neg_3 => neg_3
# neg_5 => neg_5
# setitem_8 => slice_scatter_32
# tanh => tanh
# tanh_1 => tanh_1
# tensor_1 => full_default_2
# truediv_7 => div_7
# truediv_9 => div_9
# zeros_like => full
triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_zeros_like_16 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_zeros_like_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_zeros_like_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_zeros_like_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr0 + ((-5304) + x3), tmp5 & xmask, other=0.0)
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
    tmp17 = tl.load(in_ptr1 + ((-1) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr2 + (x3), tmp16 & xmask, other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr3 + (x3), tmp16 & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + ((-5051) + x0 + (25*x1) + (5000*x2)), tmp16 & xmask, other=0.0)
    tmp22 = tl.abs(tmp21)
    tmp23 = -tmp22
    tmp24 = tl.full([1], 0.001, tl.float64)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 / tmp24
    tmp27 = libdevice.tanh(tmp26)
    tmp28 = tl.full([1], 1.0, tl.float64)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0.5, tl.float64)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp20 * tmp31
    tmp33 = tl.full([1], 50.0, tl.float64)
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp35 = tmp19 * tmp34
    tmp36 = tl.full([1], 0.0, tl.float64)
    tmp37 = tmp36 + tmp35
    tmp38 = tl.where(tmp16, tmp37, 0.0)
    tmp39 = tl.where(tmp15, tmp38, tmp36)
    tmp40 = tl.where(tmp13, tmp39, 0.0)
    tmp41 = tl.where(tmp12, tmp40, tmp36)
    tmp42 = tl.where(tmp5, tmp41, 0.0)
    tmp43 = tl.where(tmp5, tmp42, tmp36)
    tmp44 = tl.where(tmp5, tmp7, tmp43)
    tmp45 = tl.load(in_ptr5 + ((-5051) + x0 + (25*x1) + (5000*x2)), tmp16 & xmask, other=0.0)
    tmp46 = tl.abs(tmp45)
    tmp47 = -tmp46
    tmp48 = tmp47 + tmp24
    tmp49 = tmp48 / tmp24
    tmp50 = libdevice.tanh(tmp49)
    tmp51 = tmp50 + tmp28
    tmp52 = tmp51 * tmp30
    tmp53 = tmp20 * tmp52
    tmp54 = triton_helpers.maximum(tmp33, tmp53)
    tmp55 = tmp19 * tmp54
    tmp56 = tmp44 + tmp55
    tmp57 = tl.where(tmp16, tmp56, 0.0)
    tmp58 = tl.where(tmp15, tmp57, tmp44)
    tmp59 = tl.where(tmp13, tmp58, 0.0)
    tmp60 = tl.where(tmp12, tmp59, tmp44)
    tmp61 = tl.where(tmp5, tmp60, 0.0)
    tmp62 = tl.where(tmp5, tmp61, tmp44)
    tl.store(in_out_ptr0 + (x3), tmp62, xmask)
''')
