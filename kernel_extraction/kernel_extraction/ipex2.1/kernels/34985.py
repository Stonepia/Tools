

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/th/cthh5dtva7rf2fp4wxsll35edddoycwlqmeievv7bb6ggiksh3tf.py
# Source Nodes: [abs_3, add_7, add_8, mul_22, mul_26, mul_27, neg_3, setitem_9, tanh, truediv_7], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.tanh]
# abs_3 => abs_3
# add_7 => add_7
# add_8 => add_8
# mul_22 => mul_22
# mul_26 => mul_26
# mul_27 => mul_27
# neg_3 => neg_3
# setitem_9 => copy_9, select_scatter_1, select_scatter_2, slice_scatter_33, slice_scatter_34, slice_scatter_35
# tanh => tanh
# truediv_7 => div_7
triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_2 = async_compile.triton('triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 21216)
    x3 = (xindex // 104) % 204
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x6 = (xindex // 4)
    x7 = xindex
    tmp47 = tl.load(in_ptr2 + (x7), xmask).to(tl.float32)
    tmp0 = x4
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = x2
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp13 & tmp11
    tmp15 = x1
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tmp15 == tmp16
    tmp18 = x0
    tmp19 = tmp18 == tmp16
    tmp20 = tl.load(in_ptr0 + ((-5051) + x2 + (25*x3) + (5000*x4)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.abs(tmp20)
    tmp22 = -tmp21
    tmp23 = 0.001
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp23
    tmp26 = libdevice.tanh(tmp25)
    tmp27 = 1.0
    tmp28 = tmp26 + tmp27
    tmp29 = 0.5
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30 * tmp20
    tmp32 = tl.load(in_ptr1 + (x6), tmp14 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr2 + (x0 + (4*x6)), tmp14 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.where(tmp19, tmp35, tmp36)
    tmp38 = tl.load(in_ptr2 + (x7), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp17, tmp37, tmp38)
    tmp40 = tl.where(tmp14, tmp39, 0.0)
    tmp41 = tl.load(in_ptr2 + (x7), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp42 = tl.where(tmp13, tmp40, tmp41)
    tmp43 = tl.where(tmp11, tmp42, 0.0)
    tmp44 = tl.load(in_ptr2 + (x7), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp45 = tl.where(tmp10, tmp43, tmp44)
    tmp46 = tl.where(tmp5, tmp45, 0.0)
    tmp48 = tl.where(tmp5, tmp46, tmp47)
    tl.store(out_ptr0 + (x7), tmp48, xmask)
''')
