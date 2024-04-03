

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/wa/cwaijqeyycj7545ynielylevw4ct7eyaswbe5r7bheajyohy6l2b.py
# Source Nodes: [clone_1, iadd_52, iadd_53, iadd_54, mul_100, mul_145, mul_3, mul_99, neg_54, neg_57, setitem_104, setitem_105, setitem_106, sub_46, sub_47, truediv_73, truediv_74, truediv_77], Original ATen: [aten.add, aten.clone, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# clone_1 => clone_1
# iadd_52 => slice_761, slice_765, slice_766, slice_scatter_73, slice_scatter_74
# iadd_53 => slice_791, slice_797, slice_798, slice_scatter_78, slice_scatter_79
# iadd_54 => add_70, select_scatter_144, select_scatter_145, slice_822, slice_823
# mul_100 => mul_103
# mul_145 => mul_148
# mul_3 => mul_4
# mul_99 => mul_102
# neg_54 => neg_54
# neg_57 => neg_57
# setitem_104 => copy_104, select_scatter_137, slice_scatter_70, slice_scatter_71, slice_scatter_72
# setitem_105 => copy_105, select_scatter_140, select_scatter_141, slice_scatter_75, slice_scatter_76
# setitem_106 => copy_106, select_scatter_143, slice_scatter_80, slice_scatter_81, slice_scatter_82
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
# truediv_77 => div_74
triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_85 = async_compile.triton('triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_85', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_85', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_85(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x2 = (xindex // 78)
    x4 = (xindex // 3)
    x5 = xindex
    tmp15 = tl.load(in_ptr0 + (75 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (25))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (75 + x0 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (x5), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 25, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = tl.full([1], 25, tl.int64)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp6
    tmp10 = tmp8 & tmp9
    tmp11 = tmp4 == tmp4
    tmp12 = tl.load(in_ptr0 + (75 + (78*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp11, tmp12, tmp12)
    tmp14 = tl.where(tmp10, tmp13, 0.0)
    tmp16 = tl.where(tmp10, tmp14, tmp15)
    tmp18 = -tmp17
    tmp21 = 0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tmp18 / tmp22
    tmp24 = tmp16 + tmp23
    tmp25 = tl.load(in_ptr0 + (75 + x0 + (78*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp5, tmp12, tmp25)
    tmp27 = tl.where(tmp10, tmp26, 0.0)
    tmp29 = tl.where(tmp10, tmp27, tmp28)
    tmp30 = tl.where(tmp5, tmp24, tmp29)
    tmp31 = tmp0 >= tmp7
    tmp32 = tmp0 < tmp6
    tmp33 = tmp31 & tmp32
    tmp34 = tl.load(in_ptr0 + (3*x4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr0 + (x5), tmp33 & xmask, other=0.0)
    tmp36 = tl.where(tmp5, tmp34, tmp35)
    tmp37 = tl.where(tmp33, tmp36, 0.0)
    tmp39 = tl.where(tmp33, tmp37, tmp38)
    tmp40 = tl.where(tmp2, tmp30, tmp39)
    tl.store(out_ptr0 + (x5), tmp40, xmask)
''')
