

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/xo/cxotqmx3i353wvriqgb63dq7xzqhcpiu6oy4edtajzkqwas2f5ov.py
# Source Nodes: [abs_14, add_52, add_53, iadd_11, min_9, mul_135, mul_136, mul_137, mul_138, mul_139, neg_24, neg_25, pow_4, sub_17, tanh_11, tensor, truediv_30, truediv_31], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.slice_scatter, aten.sub, aten.tanh]
# abs_14 => abs_14
# add_52 => add_63
# add_53 => add_64
# iadd_11 => add_65, slice_scatter_156
# min_9 => minimum_8
# mul_135 => mul_135
# mul_136 => mul_136
# mul_137 => mul_137
# mul_138 => mul_138
# mul_139 => mul_139
# neg_24 => neg_24
# neg_25 => neg_25
# pow_4 => pow_4
# sub_17 => sub_17
# tanh_11 => tanh_11
# tensor => full_default_1
# truediv_30 => div_30
# truediv_31 => div_31
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_44 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x2 = (xindex // 5200)
    x3 = xindex % 5200
    x1 = (xindex // 26) % 200
    x4 = (xindex // 26)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2 + x2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 202, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = tl.load(in_ptr0 + (52 + x3 + (5304*x2)), tmp9 & xmask, other=0.0)
    tmp11 = tl.where(tmp9, tmp10, 0.0)
    tmp12 = tl.full([1], 0.0, tl.float64)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (2 + x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (2 + x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.load(in_ptr3 + (10660 + x3 + (5304*x2)), tmp2 & xmask, other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0 + (25*x4)), tmp2 & xmask, other=0.0)
    tmp20 = -tmp19
    tmp21 = tl.load(in_ptr5 + (x0 + (25*x4)), tmp2 & xmask, other=0.0)
    tmp22 = triton_helpers.minimum(tmp12, tmp21)
    tmp23 = tl.full([1], 1e-20, tl.float64)
    tmp24 = tmp22 - tmp23
    tmp25 = tmp20 / tmp24
    tmp26 = tl.abs(tmp25)
    tmp27 = -tmp26
    tmp28 = tl.full([1], 0.001, tl.float64)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp29 / tmp28
    tmp31 = libdevice.tanh(tmp30)
    tmp32 = tl.full([1], 1.0, tl.float64)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1], 0.5, tl.float64)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp18 * tmp35
    tmp37 = tmp25 * tmp25
    tmp38 = tmp36 * tmp37
    tmp39 = tl.load(in_ptr6 + (10660 + x3 + (5304*x2)), tmp2 & xmask, other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tmp13 + tmp40
    tmp42 = tl.where(tmp2, tmp41, 0.0)
    tmp43 = tl.load(in_ptr0 + (52 + x3 + (5304*x2)), tmp8 & xmask, other=0.0)
    tmp44 = tl.where(tmp8, tmp43, 0.0)
    tmp45 = tl.where(tmp8, tmp44, tmp12)
    tmp46 = tl.where(tmp2, tmp42, tmp45)
    tl.store(out_ptr0 + (x5), tmp46, xmask)
''')
