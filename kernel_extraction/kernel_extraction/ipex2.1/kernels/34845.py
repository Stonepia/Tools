

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/n3/cn3fcpzzrtci7qxc6dtaqs55d7ww2pztymzqyjyzcgte6lvzdd6f.py
# Source Nodes: [abs_8, add_31, add_32, min_6, mul_75, mul_79, mul_80, neg_12, neg_13, setitem_23, sub_14, tanh_5, tensor, truediv_17, truediv_18], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.sub, aten.tanh]
# abs_8 => abs_8
# add_31 => add_36
# add_32 => add_37
# min_6 => minimum_5
# mul_75 => mul_75
# mul_79 => mul_79
# mul_80 => mul_80
# neg_12 => neg_12
# neg_13 => neg_13
# setitem_23 => copy_23, select_scatter_12
# sub_14 => sub_14
# tanh_5 => tanh_5
# tensor => full_default_1
# truediv_17 => div_17
# truediv_18 => div_18
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_22 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_22', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2010000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x4 = (xindex // 2)
    x1 = (xindex // 2) % 25
    x2 = (xindex // 50) % 201
    x3 = (xindex // 10050)
    x6 = xindex
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (10635 + x1 + (26*x2) + (5304*x3)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (42542 + x0 + (4*x1) + (104*x2) + (21216*x3)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = -tmp3
    tmp6 = 0.0
    tmp7 = triton_helpers.minimum(tmp6, tmp5)
    tmp8 = 1e-20
    tmp9 = tmp7 - tmp8
    tmp10 = tmp4 / tmp9
    tmp11 = tl.abs(tmp10)
    tmp12 = -tmp11
    tmp13 = 0.001
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 / tmp13
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20 * tmp10
    tmp23 = tmp21 * tmp22
    tmp24 = 2 + x3
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp24 >= tmp25
    tmp27 = tl.full([1], 202, tl.int64)
    tmp28 = tmp24 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr3 + (110 + x0 + (4*x1) + (104*x2) + (21216*x3)), tmp29 & xmask, other=0.0)
    tmp31 = tl.where(tmp29, tmp30, 0.0)
    tmp33 = tl.where(tmp29, tmp31, tmp32)
    tmp34 = tl.where(tmp2, tmp23, tmp33)
    tl.store(out_ptr0 + (x6), tmp34, xmask)
''')
