

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/fl/cfld6ze3aqplc7azuteedx2vwmv4kfpwilmfisewlge7sepj73jr.py
# Source Nodes: [clone_1, iadd_52, iadd_53, mul_100, mul_145, mul_99, neg_54, neg_56, setitem_104, setitem_105, sub_46, sub_47, truediv_73, truediv_74, truediv_76], Original ATen: [aten.add, aten.clone, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# clone_1 => clone_1
# iadd_52 => slice_761, slice_765, slice_766, slice_scatter_73, slice_scatter_74
# iadd_53 => add_69, select_scatter_142, slice_791, slice_792, slice_scatter_77
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# neg_56 => neg_56
# setitem_104 => copy_104, select_scatter_137, slice_scatter_70, slice_scatter_71, slice_scatter_72
# setitem_105 => copy_105, select_scatter_140, select_scatter_141, slice_scatter_75, slice_scatter_76
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
# truediv_76 => div_73
triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_83 = async_compile.triton('triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_83', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_83', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_83(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x2 = (xindex // 78)
    x4 = (xindex // 3)
    x5 = xindex
    tmp26 = tl.load(in_ptr0 + (78*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (x0 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (x5), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 25, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tmp0 == tmp7
    tmp10 = tmp7 == tmp7
    tmp11 = tl.load(in_ptr0 + (78*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.where(tmp10, tmp11, tmp11)
    tmp13 = tl.load(in_ptr0 + (3*x4), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.load(in_ptr1 + ((-1) + x1 + (24*x2)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = -tmp15
    tmp17 = tl.load(in_ptr2 + (x1), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 / tmp17
    tmp19 = tmp14 + tmp18
    tmp20 = tl.load(in_ptr0 + (x0 + (78*x2)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp8, tmp11, tmp20)
    tmp22 = tl.load(in_ptr0 + (x5), tmp5 & xmask, other=0.0)
    tmp23 = tl.where(tmp9, tmp21, tmp22)
    tmp24 = tl.where(tmp8, tmp19, tmp23)
    tmp25 = tl.where(tmp5, tmp24, 0.0)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp30 = tl.where(tmp9, tmp28, tmp29)
    tmp31 = tl.where(tmp5, tmp25, tmp30)
    tl.store(out_ptr0 + (x5), tmp31, xmask)
''')
