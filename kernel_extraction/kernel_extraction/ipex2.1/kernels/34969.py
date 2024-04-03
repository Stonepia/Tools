

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/a4/ca4tfpn46m2apajl77qvwtsllmmzy7o6esbyin4zisy7r32w3eng.py
# Source Nodes: [abs_14, add_52, add_53, min_9, mul_135, mul_140, mul_141, neg_24, neg_25, setitem_32, sub_17, tanh_11, tensor, truediv_30, truediv_31], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.sub, aten.tanh]
# abs_14 => abs_14
# add_52 => add_63
# add_53 => add_64
# min_9 => minimum_8
# mul_135 => mul_135
# mul_140 => mul_140
# mul_141 => mul_141
# neg_24 => neg_24
# neg_25 => neg_25
# setitem_32 => copy_32, select_scatter_24, select_scatter_25, slice_scatter_159, slice_scatter_160
# sub_17 => sub_17
# tanh_11 => tanh_11
# tensor => full_default_1
# truediv_30 => div_30
# truediv_31 => div_31
triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_39 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_39', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_minimum_mul_neg_select_scatter_slice_scatter_sub_tanh_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4243200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 104) % 204
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x4 = (xindex // 21216)
    x6 = xindex
    tmp34 = tl.load(in_ptr2 + (42432 + x6), xmask)
    tmp0 = x3
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x2
    tmp7 = tl.full([1], 25, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = x1
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.load(in_ptr0 + ((-100) + x0 + (2*x2) + (50*x3) + (10000*x4)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 2 + x4
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17 & tmp9
    tmp19 = tl.load(in_ptr1 + (x6), tmp18 & xmask, other=0.0)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.load(in_ptr2 + (42432 + x6), tmp9 & xmask, other=0.0)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp12, tmp13, tmp22)
    tmp24 = tl.where(tmp9, tmp23, 0.0)
    tmp25 = tmp17 & tmp5
    tmp26 = tl.load(in_ptr1 + (x6), tmp25 & xmask, other=0.0)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tl.load(in_ptr2 + (42432 + x6), tmp5 & xmask, other=0.0)
    tmp29 = tl.where(tmp17, tmp27, tmp28)
    tmp30 = tl.where(tmp8, tmp24, tmp29)
    tmp31 = tl.where(tmp5, tmp30, 0.0)
    tmp32 = tl.load(in_ptr1 + (x6), tmp17 & xmask, other=0.0)
    tmp33 = tl.where(tmp17, tmp32, 0.0)
    tmp35 = tl.where(tmp17, tmp33, tmp34)
    tmp36 = tl.where(tmp5, tmp31, tmp35)
    tl.store(out_ptr0 + (x6), tmp36, xmask)
''')
