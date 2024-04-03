

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/7v/c7vketra55su23uo5keqfzxwnyzodsm32vflexucpxyfdxn6bkhh.py
# Source Nodes: [abs_7, add_27, add_28, min_5, mul_65, mul_69, mul_70, neg_10, neg_11, setitem_21, sub_13, tanh_4, tensor, truediv_15, truediv_16], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_7 => abs_7
# add_27 => add_31
# add_28 => add_32
# min_5 => minimum_4
# mul_65 => mul_65
# mul_69 => mul_69
# mul_70 => mul_70
# neg_10 => neg_10
# neg_11 => neg_11
# setitem_21 => copy_21, select_scatter_10, select_scatter_11, slice_scatter_93
# sub_13 => sub_13
# tanh_4 => tanh_4
# tensor => full_default_1
# truediv_15 => div_15
# truediv_16 => div_16
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4180800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x6 = (xindex // 104)
    x4 = (xindex // 20904)
    x7 = (xindex // 4) % 5226
    x8 = xindex % 20904
    x9 = xindex
    tmp34 = tl.load(in_ptr3 + (42536 + x8 + (21216*x4)), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = x0
    tmp7 = tmp6 == tmp4
    tmp8 = tl.load(in_ptr0 + ((-1) + x2 + (25*x6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = -tmp8
    tmp10 = tl.load(in_ptr1 + ((-1) + x2 + (25*x6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([1], 0.0, tl.float64)
    tmp12 = triton_helpers.minimum(tmp11, tmp10)
    tmp13 = tl.full([1], 1e-20, tl.float64)
    tmp14 = tmp12 - tmp13
    tmp15 = tmp9 / tmp14
    tmp16 = tl.abs(tmp15)
    tmp17 = -tmp16
    tmp18 = tl.full([1], 0.001, tl.float64)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19 / tmp18
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tl.full([1], 1.0, tl.float64)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0.5, tl.float64)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 * tmp15
    tmp27 = tl.load(in_ptr2 + (10634 + x7 + (5304*x4)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr3 + (42536 + x0 + (4*x7) + (21216*x4)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp7, tmp28, tmp29)
    tmp31 = tl.load(in_ptr3 + (42536 + x8 + (21216*x4)), tmp2 & xmask, other=0.0)
    tmp32 = tl.where(tmp5, tmp30, tmp31)
    tmp33 = tl.where(tmp2, tmp32, 0.0)
    tmp35 = tl.where(tmp2, tmp33, tmp34)
    tl.store(out_ptr0 + (x9), tmp35, xmask)
''')
