

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/nr/cnr6o25la7vumlirus52b3kbvxhazyki63q32fut4fkih7pu4qxd.py
# Source Nodes: [abs_17, add_62, add_63, min_13, mul_165, mul_170, mul_171, neg_30, neg_31, setitem_35, sub_21, tanh_14, tensor, truediv_36, truediv_37], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_17 => abs_17
# add_62 => add_76
# add_63 => add_77
# min_13 => minimum_12
# mul_165 => mul_165
# mul_170 => mul_170
# mul_171 => mul_171
# neg_30 => neg_30
# neg_31 => neg_31
# setitem_35 => copy_35, select_scatter_30, select_scatter_31, slice_scatter_177
# sub_21 => sub_21
# tanh_14 => tanh_14
# tensor => full_default_1
# truediv_36 => div_36
# truediv_37 => div_37
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_40 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_40', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4160000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x6 = (xindex // 104)
    x4 = (xindex // 20800)
    x7 = (xindex // 4) % 5200
    x8 = xindex % 20800
    x9 = xindex
    tmp39 = tl.load(in_ptr3 + (42640 + x8 + (21216*x4)), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + (x2 + (25*x6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = -tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (x2 + (25*x6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 0.0
    tmp15 = triton_helpers.minimum(tmp14, tmp13)
    tmp16 = 1e-20
    tmp17 = tmp15 - tmp16
    tmp18 = tmp11 / tmp17
    tmp19 = tl.abs(tmp18)
    tmp20 = -tmp19
    tmp21 = 0.001
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 / tmp21
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = 1.0
    tmp26 = tmp24 + tmp25
    tmp27 = 0.5
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28 * tmp18
    tmp30 = tl.load(in_ptr2 + (10660 + x7 + (5304*x4)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (42640 + x0 + (4*x7) + (21216*x4)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp8, tmp33, tmp34)
    tmp36 = tl.load(in_ptr3 + (42640 + x8 + (21216*x4)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tl.where(tmp2, tmp37, 0.0)
    tmp40 = tl.where(tmp2, tmp38, tmp39)
    tl.store(out_ptr0 + (x9), tmp40, xmask)
''')
