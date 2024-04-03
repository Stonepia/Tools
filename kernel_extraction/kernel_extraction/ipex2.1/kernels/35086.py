

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/y2/cy2avkephjr6ugdzldrr4ioff5nfp7yp2gyty67pgmmgnqzjsp6g.py
# Source Nodes: [add_67, iadd_10, iadd_11, iadd_14, iadd_15, mul_183, mul_184, setitem_37, setitem_38, truediv_41, zeros_like], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.select_scatter, aten.slice_scatter, aten.zeros_like]
# add_67 => add_83
# iadd_10 => slice_scatter_152
# iadd_11 => slice_scatter_157, slice_scatter_158
# iadd_14 => slice_scatter_176
# iadd_15 => slice_scatter_182
# mul_183 => mul_183
# mul_184 => mul_184
# setitem_37 => copy_37, slice_scatter_186, slice_scatter_187, slice_scatter_188
# setitem_38 => copy_38, select_scatter_34, slice_scatter_189, slice_scatter_190
# truediv_41 => div_41
# zeros_like => full
triton_poi_fused_add_copy_div_mul_select_scatter_slice_scatter_zeros_like_50 = async_compile.triton('triton_poi_fused_add_copy_div_mul_select_scatter_slice_scatter_zeros_like_50', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr7', 'out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_select_scatter_slice_scatter_zeros_like_50', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_select_scatter_slice_scatter_zeros_like_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x3 = xindex
    x1 = (xindex // 26) % 204
    x4 = xindex % 5304
    x0 = xindex % 26
    tmp49 = tl.load(in_ptr7 + (x3), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-10608) + x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + ((-10608) + x3), tmp5 & xmask, other=0.0)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = x1
    tmp11 = tmp10 >= tmp1
    tmp12 = tmp10 < tmp3
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13 & tmp5
    tmp15 = tl.load(in_ptr2 + ((-10452) + x4 + (5200*x2)), tmp14 & xmask, other=0.0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tmp5 & tmp5
    tmp18 = tl.load(in_ptr3 + ((-10608) + x3), tmp17 & xmask, other=0.0)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.full([1], 0.0, tl.float64)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tmp22 = tl.where(tmp13, tmp16, tmp21)
    tmp23 = tl.where(tmp5, tmp22, 0.0)
    tmp24 = tl.load(in_ptr3 + ((-10608) + x3), tmp5 & xmask, other=0.0)
    tmp25 = tl.where(tmp5, tmp24, 0.0)
    tmp26 = tl.where(tmp5, tmp25, tmp20)
    tmp27 = tl.where(tmp5, tmp23, tmp26)
    tmp28 = tl.where(tmp5, tmp9, tmp27)
    tmp29 = tl.where(tmp5, tmp7, tmp28)
    tmp30 = x0
    tmp31 = tl.full([1], 25, tl.int64)
    tmp32 = tmp30 < tmp31
    tmp33 = tmp32 & tmp14
    tmp34 = tl.load(in_ptr4 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp33 & xmask, other=0.0)
    tmp35 = tl.load(in_ptr5 + (x1), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full([1], 4.0, tl.float64)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr6 + (x1), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tmp29 / tmp39
    tmp41 = tmp34 + tmp40
    tmp42 = tl.where(tmp33, tmp41, 0.0)
    tmp43 = tl.load(in_ptr7 + (x3), tmp14 & xmask, other=0.0)
    tmp44 = tl.where(tmp32, tmp42, tmp43)
    tmp45 = tl.where(tmp14, tmp44, 0.0)
    tmp46 = tl.load(in_ptr7 + (x3), tmp5 & xmask, other=0.0)
    tmp47 = tl.where(tmp13, tmp45, tmp46)
    tmp48 = tl.where(tmp5, tmp47, 0.0)
    tmp50 = tl.where(tmp5, tmp48, tmp49)
    tmp51 = tl.full([1], 25, tl.int32)
    tmp52 = tmp30 == tmp51
    tmp53 = tl.where(tmp52, tmp20, tmp50)
    tmp54 = tl.where(tmp14, tmp53, 0.0)
    tmp55 = tl.where(tmp13, tmp54, tmp50)
    tmp56 = tl.where(tmp5, tmp55, 0.0)
    tmp57 = tl.where(tmp5, tmp56, tmp50)
    tl.store(out_ptr2 + (x3), tmp57, xmask)
''')
