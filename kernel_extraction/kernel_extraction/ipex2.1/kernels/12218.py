

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/cp/ccp4ruah6winpl3alxk7vm6jmpfaukoaj63q6eschfimjf72gugi.py
# Source Nodes: [clone_1, iadd_52, iadd_53, iadd_54, mul_100, mul_145, mul_99, neg_54, setitem_104, setitem_105, setitem_106, setitem_107, sub_46, sub_47, truediv_73, truediv_74], Original ATen: [aten.clone, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# clone_1 => clone_1
# iadd_52 => select_scatter_139, slice_761, slice_762, slice_765, slice_766, slice_scatter_73, slice_scatter_74
# iadd_53 => slice_791, slice_797, slice_798, slice_scatter_78, slice_scatter_79
# iadd_54 => select_scatter_145, slice_822, slice_823, slice_826, slice_827, slice_scatter_83, slice_scatter_84
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# setitem_104 => copy_104, select_scatter_137, slice_scatter_70, slice_scatter_71, slice_scatter_72
# setitem_105 => select_scatter_141, slice_scatter_75, slice_scatter_76
# setitem_106 => copy_106, select_scatter_143, slice_scatter_80, slice_scatter_81, slice_scatter_82
# setitem_107 => copy_107, select_scatter_146, select_scatter_147, slice_scatter_85, slice_scatter_86
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_87 = async_compile.triton('triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_87', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_87', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_87(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x2 = (xindex // 78)
    x4 = (xindex // 3)
    x5 = xindex
    tmp7 = tl.load(in_ptr0 + (3*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (75 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (x0 + (3*x2)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (75 + x0 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (x5), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 25, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp1 == tmp1
    tmp8 = tl.full([1], 25, tl.int64)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tmp8 < tmp8
    tmp12 = tmp10 & tmp11
    tmp13 = tmp4 == tmp4
    tmp14 = tl.load(in_ptr1 + (75 + (78*x2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp13, tmp14, tmp14)
    tmp16 = tl.where(tmp12, tmp15, 0.0)
    tmp18 = tl.where(tmp12, tmp16, tmp17)
    tmp19 = tl.where(tmp6, tmp7, tmp18)
    tmp21 = tl.load(in_ptr1 + (75 + x0 + (78*x2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp5, tmp14, tmp21)
    tmp23 = tl.where(tmp12, tmp22, 0.0)
    tmp25 = tl.where(tmp12, tmp23, tmp24)
    tmp26 = tl.where(tmp6, tmp20, tmp25)
    tmp27 = tl.where(tmp5, tmp19, tmp26)
    tmp28 = tmp0 >= tmp9
    tmp29 = tmp0 < tmp8
    tmp30 = tmp28 & tmp29
    tmp31 = tl.load(in_ptr1 + (3*x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr1 + (x5), tmp30 & xmask, other=0.0)
    tmp33 = tl.where(tmp5, tmp31, tmp32)
    tmp34 = tl.where(tmp30, tmp33, 0.0)
    tmp36 = tl.where(tmp30, tmp34, tmp35)
    tmp37 = tl.where(tmp2, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp27, tmp37)
    tl.store(in_out_ptr0 + (x5), tmp38, xmask)
''')
