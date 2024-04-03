

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ec/ceckoujwb7lkofjqylbepre4iok25au67bpa3sdzzwqet3sifmm6.py
# Source Nodes: [clone_1, iadd_52, iadd_53, iadd_54, mul_100, mul_145, mul_99, neg_54, setitem_104, setitem_105, setitem_106, setitem_107, sub_46, sub_47, truediv_73, truediv_74], Original ATen: [aten.clone, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# clone_1 => clone_1
# iadd_52 => slice_761, slice_765, slice_766, slice_scatter_73, slice_scatter_74
# iadd_53 => slice_791, slice_797, slice_798, slice_scatter_78, slice_scatter_79
# iadd_54 => slice_822, slice_826, slice_827, slice_scatter_83, slice_scatter_84
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# setitem_104 => copy_104, select_scatter_137, slice_scatter_70, slice_scatter_71, slice_scatter_72
# setitem_105 => copy_105, select_scatter_140, select_scatter_141, slice_scatter_75, slice_scatter_76
# setitem_106 => copy_106, select_scatter_143, slice_scatter_80, slice_scatter_81, slice_scatter_82
# setitem_107 => copy_107, select_scatter_146, select_scatter_147, slice_scatter_85, slice_scatter_86
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_86 = async_compile.triton('triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_86', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_86', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_86(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x2 = (xindex // 78)
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (75 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (75 + x0 + (78*x2)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 25, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')
