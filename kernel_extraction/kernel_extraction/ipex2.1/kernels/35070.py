

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/gu/cgubfvweefmvhfq4gieprqd6j44h2gf7kki456flyxejfzwefx5i.py
# Source Nodes: [abs_12, add_46, add_47, mul_115, mul_120, mul_121, neg_21, setitem_30, tanh_9, truediv_27], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice_scatter, aten.tanh]
# abs_12 => abs_12
# add_46 => add_55
# add_47 => add_56
# mul_115 => mul_115
# mul_120 => mul_120
# mul_121 => mul_121
# neg_21 => neg_21
# setitem_30 => copy_30, select_scatter_20, select_scatter_21, slice_scatter_147, slice_scatter_148, slice_scatter_149
# tanh_9 => tanh_9
# truediv_27 => div_27
triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_34 = async_compile.triton('triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_slice_scatter_tanh_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp46 = tl.load(in_ptr2 + (x7), xmask)
    tmp0 = x4
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x3
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = x2
    tmp12 = tl.full([1], 25, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = x1
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp15 == tmp16
    tmp18 = x0
    tmp19 = tl.full([1], 0, tl.int32)
    tmp20 = tmp18 == tmp19
    tmp21 = tl.load(in_ptr0 + ((-10050) + x2 + (25*x3) + (5000*x4)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.abs(tmp21)
    tmp23 = -tmp22
    tmp24 = tl.full([1], 0.001, tl.float64)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 / tmp24
    tmp27 = libdevice.tanh(tmp26)
    tmp28 = tl.full([1], 1.0, tl.float64)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0.5, tl.float64)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp21
    tmp33 = tl.load(in_ptr1 + (x6), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.load(in_ptr2 + (2 + x0 + (4*x6)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tl.load(in_ptr2 + (x7), tmp14 & xmask, other=0.0)
    tmp38 = tl.where(tmp17, tmp36, tmp37)
    tmp39 = tl.where(tmp14, tmp38, 0.0)
    tmp40 = tl.load(in_ptr2 + (x7), tmp10 & xmask, other=0.0)
    tmp41 = tl.where(tmp13, tmp39, tmp40)
    tmp42 = tl.where(tmp10, tmp41, 0.0)
    tmp43 = tl.load(in_ptr2 + (x7), tmp5 & xmask, other=0.0)
    tmp44 = tl.where(tmp9, tmp42, tmp43)
    tmp45 = tl.where(tmp5, tmp44, 0.0)
    tmp47 = tl.where(tmp5, tmp45, tmp46)
    tl.store(out_ptr0 + (x7), tmp47, xmask)
''')
