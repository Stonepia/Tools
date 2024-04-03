

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/sz/cszzddqiqxlni3uo5dwmcm5n34z2oxo3x6r3eqtv7zznm7rn6vml.py
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
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_20 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2010000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x4 = (xindex // 2)
    x1 = (xindex // 2) % 25
    x2 = (xindex // 50) % 201
    x3 = (xindex // 10050)
    x7 = (xindex // 50)
    x8 = xindex
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (10635 + x1 + (26*x2) + (5304*x3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp45 = tl.load(in_ptr4 + (42542 + x0 + (4*x1) + (104*x2) + (21216*x3)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = -tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.0
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tmp10 = 1e-20
    tmp11 = tmp9 - tmp10
    tmp12 = tmp5 / tmp11
    tmp13 = tl.abs(tmp12)
    tmp14 = -tmp13
    tmp15 = 0.001
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 / tmp15
    tmp18 = libdevice.tanh(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 + tmp19
    tmp21 = 0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp12
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 2 + x3
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp28 >= tmp29
    tmp31 = tl.full([1], 202, tl.int64)
    tmp32 = tmp28 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = 1 + x2
    tmp35 = tl.full([1], 1, tl.int64)
    tmp36 = tmp34 >= tmp35
    tmp37 = tmp34 < tmp31
    tmp38 = tmp36 & tmp37
    tmp39 = tmp38 & tmp33
    tmp40 = tl.load(in_ptr3 + (6 + x0 + (4*x1) + (104*x7)), tmp39 & xmask, other=0.0).to(tl.float32)
    tmp41 = tl.where(tmp39, tmp40, 0.0)
    tmp42 = tl.load(in_ptr4 + (42542 + x0 + (4*x1) + (104*x2) + (21216*x3)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp38, tmp41, tmp42)
    tmp44 = tl.where(tmp33, tmp43, 0.0)
    tmp46 = tl.where(tmp33, tmp44, tmp45)
    tmp47 = tl.where(tmp2, tmp27, tmp46)
    tl.store(out_ptr0 + (x8), tmp47, xmask)
''')
