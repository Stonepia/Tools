

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/e2/ce2f36coziti2wwx3k2iiexr6ziqi6hifc2luw76ot3vtlr2cl6y.py
# Source Nodes: [abs_10, add_39, add_40, min_8, mul_100, mul_95, mul_99, neg_16, neg_17, setitem_27, sub_16, tanh_7, tensor, truediv_21, truediv_22], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.sub, aten.tanh]
# abs_10 => abs_10
# add_39 => add_46
# add_40 => add_47
# min_8 => minimum_7
# mul_100 => mul_100
# mul_95 => mul_95
# mul_99 => mul_99
# neg_16 => neg_16
# neg_17 => neg_17
# setitem_27 => copy_27, select_scatter_16
# sub_16 => sub_16
# tanh_7 => tanh_7
# tensor => full_default_1
# truediv_21 => div_21
# truediv_22 => div_22
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_24 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_sub_tanh_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2090400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x3 = (xindex // 2)
    x1 = (xindex // 2) % 5226
    x2 = (xindex // 10452)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (10634 + x1 + (5304*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp36 = tl.load(in_ptr4 + (42538 + x0 + (4*x1) + (21216*x2)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
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
    tmp28 = 2 + x2
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp28 >= tmp29
    tmp31 = tl.full([1], 202, tl.int64)
    tmp32 = tmp28 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr3 + (106 + x0 + (4*x1) + (21216*x2)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp37 = tl.where(tmp33, tmp35, tmp36)
    tmp38 = tl.where(tmp2, tmp27, tmp37)
    tl.store(out_ptr0 + (x5), tmp38, xmask)
''')
