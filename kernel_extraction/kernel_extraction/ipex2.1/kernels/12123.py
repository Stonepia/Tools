

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/7t/c7tbjltjduwahxwdmt4n25fxlddr4sbrld5mg4aqjiwutqo67inm.py
# Source Nodes: [setitem_105], Original ATen: [aten.copy, aten.select_scatter]
# setitem_105 => copy_105, select_scatter_140
triton_poi_fused_copy_select_scatter_83 = async_compile.triton('triton_poi_fused_copy_select_scatter_83', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_select_scatter_83', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_select_scatter_83(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 124848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x3 = (xindex // 3)
    x2 = (xindex // 612)
    x1 = (xindex // 3) % 204
    x4 = xindex
    tmp4 = tl.load(in_ptr0 + (3*x3), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (78*x3), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (x4), xmask)
    tmp33 = tl.load(in_ptr2 + (x0 + (78*x3)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp5 = x2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 202, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = x1
    tmp12 = tmp11 >= tmp6
    tmp13 = tmp11 < tmp8
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr1 + ((-10452) + (26*x1) + (5200*x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr2 + (78*x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp3, tmp16, tmp17)
    tmp19 = tl.where(tmp15, tmp18, 0.0)
    tmp20 = tl.load(in_ptr2 + (78*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp10, tmp21, 0.0)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tl.where(tmp3, tmp4, tmp24)
    tmp27 = tl.load(in_ptr2 + (x0 + (78*x3)), tmp15 & xmask, other=0.0)
    tmp28 = tl.where(tmp2, tmp16, tmp27)
    tmp29 = tl.where(tmp15, tmp28, 0.0)
    tmp30 = tl.load(in_ptr2 + (x0 + (78*x3)), tmp10 & xmask, other=0.0)
    tmp31 = tl.where(tmp14, tmp29, tmp30)
    tmp32 = tl.where(tmp10, tmp31, 0.0)
    tmp34 = tl.where(tmp10, tmp32, tmp33)
    tmp35 = tl.where(tmp3, tmp26, tmp34)
    tmp36 = tl.where(tmp2, tmp25, tmp35)
    tl.store(out_ptr0 + (x4), tmp36, xmask)
''')
