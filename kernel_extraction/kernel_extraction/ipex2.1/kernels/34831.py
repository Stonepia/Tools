

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/37/c375gyibgubyme64g4z7hqo2t5rlusbkjd5etiaqzlwmmv5rtbze.py
# Source Nodes: [abs_5, add_15, add_16, mul_42, mul_46, mul_47, neg_7, setitem_13, setitem_15, tanh_2, truediv_11], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.tanh]
# abs_5 => abs_5
# add_15 => add_17
# add_16 => add_18
# mul_42 => mul_42
# mul_46 => mul_46
# mul_47 => mul_47
# neg_7 => neg_7
# setitem_13 => copy_13, select_scatter_5, select_scatter_6, slice_scatter_59, slice_scatter_60, slice_scatter_61
# setitem_15 => slice_scatter_74
# tanh_2 => tanh_2
# truediv_11 => div_11
triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_8 = async_compile.triton('triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 21216)
    x5 = xindex
    x3 = (xindex // 104) % 204
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x7 = (xindex // 4) % 5304
    x8 = (xindex // 4)
    tmp42 = tl.load(in_ptr3 + (x5), xmask)
    tmp0 = x4
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-21216) + x5), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = x3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tmp8 < tmp3
    tmp12 = tmp10 & tmp11
    tmp13 = tmp12 & tmp5
    tmp14 = x1
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = tmp14 == tmp15
    tmp17 = x0
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp17 == tmp18
    tmp20 = tl.load(in_ptr1 + ((-5252) + x7 + (5200*x4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp32 = tl.load(in_ptr2 + (x8), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tl.load(in_ptr3 + (x0 + (4*x8)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.load(in_ptr3 + (x5), tmp13 & xmask, other=0.0)
    tmp37 = tl.where(tmp16, tmp35, tmp36)
    tmp38 = tl.where(tmp13, tmp37, 0.0)
    tmp39 = tl.load(in_ptr3 + (x5), tmp5 & xmask, other=0.0)
    tmp40 = tl.where(tmp12, tmp38, tmp39)
    tmp41 = tl.where(tmp5, tmp40, 0.0)
    tmp43 = tl.where(tmp5, tmp41, tmp42)
    tmp44 = tl.where(tmp5, tmp7, tmp43)
    tl.store(out_ptr0 + (x5), tmp44, xmask)
''')
