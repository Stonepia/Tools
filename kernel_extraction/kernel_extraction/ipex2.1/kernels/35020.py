

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ws/cwsiiknmtasp5dgplcjvmskwfimmpzpc4kfw43rs5ilzt35dziov.py
# Source Nodes: [abs_14, add_52, add_53, min_9, mul_135, mul_140, mul_141, neg_24, neg_25, setitem_32, sub_17, tanh_11, tensor, truediv_30, truediv_31], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.sub, aten.tanh]
# abs_14 => abs_14
# add_52 => add_63
# add_53 => add_64
# min_9 => minimum_8
# mul_135 => mul_135
# mul_140 => mul_140
# mul_141 => mul_141
# neg_24 => neg_24
# neg_25 => neg_25
# setitem_32 => copy_32, select_scatter_24
# sub_17 => sub_17
# tanh_11 => tanh_11
# tensor => full_default_1
# truediv_30 => div_30
# truediv_31 => div_31
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_37 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_37', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_37', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x4 = (xindex // 2)
    x1 = (xindex // 2) % 25
    x2 = (xindex // 50) % 200
    x3 = (xindex // 10000)
    x7 = (xindex // 50)
    x8 = xindex
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (10660 + x1 + (26*x2) + (5304*x3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr4 + (42642 + x0 + (4*x1) + (104*x2) + (21216*x3)), xmask).to(tl.float32)
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
    tmp34 = 2 + x2
    tmp35 = tmp34 >= tmp29
    tmp36 = tmp34 < tmp31
    tmp37 = tmp35 & tmp36
    tmp38 = tmp37 & tmp33
    tmp39 = tl.load(in_ptr3 + (2 + x0 + (4*x1) + (104*x7)), tmp38 & xmask, other=0.0).to(tl.float32)
    tmp40 = tl.where(tmp38, tmp39, 0.0)
    tmp41 = tl.load(in_ptr4 + (42642 + x0 + (4*x1) + (104*x2) + (21216*x3)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.where(tmp33, tmp42, 0.0)
    tmp45 = tl.where(tmp33, tmp43, tmp44)
    tmp46 = tl.where(tmp2, tmp27, tmp45)
    tl.store(out_ptr0 + (x8), tmp46, xmask)
''')
