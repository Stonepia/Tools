

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/jh/cjhymmhask327cdx5gxg2lvcpbzuocwwetoyx7hc57eehusf2sus.py
# Source Nodes: [iadd_2, iadd_3, mul_58, setitem_10, setitem_12, setitem_16, truediv_14], Original ATen: [aten.copy, aten.div, aten.mul, aten.slice_scatter]
# iadd_2 => slice_scatter_53
# iadd_3 => slice_scatter_66
# mul_58 => mul_58
# setitem_10 => copy_10, slice_scatter_41, slice_scatter_42, slice_scatter_43, slice_scatter_44, slice_scatter_45
# setitem_12 => slice_scatter_58
# setitem_16 => copy_16, slice_scatter_75, slice_scatter_76, slice_scatter_77
# truediv_14 => div_14
triton_poi_fused_copy_div_mul_slice_scatter_20 = async_compile.triton('triton_poi_fused_copy_div_mul_slice_scatter_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_ptr4', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_slice_scatter_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_slice_scatter_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x3 = xindex
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    tmp29 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp48 = tl.load(in_ptr4 + (x3), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-5304) + x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + ((-5304) + x3), tmp5 & xmask, other=0.0)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = tl.load(in_ptr2 + ((-5304) + x3), tmp5 & xmask, other=0.0)
    tmp11 = tl.where(tmp5, tmp10, 0.0)
    tmp12 = x1
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tmp12 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp16 & tmp5
    tmp18 = x0
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_out_ptr0 + (x3), tmp20 & xmask, other=0.0)
    tmp22 = tl.where(tmp20, tmp21, 0.0)
    tmp23 = tl.load(in_out_ptr0 + (x3), tmp17 & xmask, other=0.0)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp17, tmp24, 0.0)
    tmp26 = tl.load(in_out_ptr0 + (x3), tmp5 & xmask, other=0.0)
    tmp27 = tl.where(tmp16, tmp25, tmp26)
    tmp28 = tl.where(tmp5, tmp27, 0.0)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tl.where(tmp5, tmp11, tmp30)
    tmp32 = tl.where(tmp5, tmp9, tmp31)
    tmp33 = tl.where(tmp5, tmp7, tmp32)
    tmp34 = tmp5 & tmp17
    tmp35 = tmp16 & tmp34
    tmp36 = tl.where(tmp35, tmp33, 0.0)
    tmp37 = tl.where(tmp16, tmp36, tmp33)
    tmp38 = tl.where(tmp34, tmp37, 0.0)
    tmp39 = tl.where(tmp5, tmp38, tmp33)
    tmp40 = tl.load(in_ptr3 + (x0), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([1], 4.0, tl.float64)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp39 / tmp42
    tmp44 = tl.where(tmp17, tmp43, 0.0)
    tmp45 = tl.load(in_ptr4 + (x3), tmp5 & xmask, other=0.0)
    tmp46 = tl.where(tmp16, tmp44, tmp45)
    tmp47 = tl.where(tmp5, tmp46, 0.0)
    tmp49 = tl.where(tmp5, tmp47, tmp48)
    tl.store(out_ptr0 + (x3), tmp49, xmask)
''')
