

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/lp/clpzsceu23kv3gqrxery5xdggapjhr3zhqpimuc6wuigzwntvchy.py
# Source Nodes: [iadd_55, mul_147, mul_148, mul_149, setitem_108, sub_50], Original ATen: [aten.add, aten.copy, aten.mul, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# iadd_55 => add_71, select_scatter_148, slice_857, slice_858, slice_859, slice_scatter_87, slice_scatter_88, slice_scatter_89
# mul_147 => mul_150
# mul_148 => mul_151
# mul_149 => mul_152
# setitem_108 => copy_108, select_scatter_149, slice_scatter_90, slice_scatter_91, slice_scatter_92
# sub_50 => sub_50
triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_87 = async_compile.triton('triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_87', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_87', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_87(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3)
    x2 = xindex
    tmp4 = tl.load(in_ptr0 + (1 + (3*x1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (3*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + (3*x1)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp6 = 1.6
    tmp7 = tmp5 * tmp6
    tmp9 = 0.6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 + tmp13
    tmp15 = tl.where(tmp3, tmp14, tmp4)
    tmp17 = tl.where(tmp2, tmp14, tmp16)
    tmp18 = tl.where(tmp2, tmp15, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''')
