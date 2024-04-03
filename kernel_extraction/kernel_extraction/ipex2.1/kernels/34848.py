

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/bq/cbqnujtqhcxypowrbj2mbcil33fe5admymuaecwg2mm5d4sjvejr.py
# Source Nodes: [setitem_25], Original ATen: [aten.select_scatter, aten.slice_scatter]
# setitem_25 => select_scatter_15, slice_scatter_119, slice_scatter_120
triton_poi_fused_select_scatter_slice_scatter_25 = async_compile.triton('triton_poi_fused_select_scatter_slice_scatter_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_scatter_slice_scatter_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_select_scatter_slice_scatter_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4243200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 104) % 204
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x4 = (xindex // 21216)
    x5 = (xindex // 4) % 5304
    x7 = xindex
    tmp29 = tl.load(in_ptr3 + (42432 + x7), xmask)
    tmp0 = x3
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + ((-52) + x0 + (2*x5) + (10452*x4)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = 2 + x4
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tmp10 < tmp3
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp5
    tmp16 = tl.load(in_ptr1 + (x7), tmp15 & xmask, other=0.0)
    tmp17 = tl.where(tmp15, tmp16, 0.0)
    tmp18 = tl.load(in_ptr2 + (x7), tmp15 & xmask, other=0.0)
    tmp19 = tl.where(tmp15, tmp18, 0.0)
    tmp20 = tl.load(in_ptr3 + (42432 + x7), tmp5 & xmask, other=0.0)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp14, tmp17, tmp21)
    tmp23 = tl.where(tmp8, tmp9, tmp22)
    tmp24 = tl.where(tmp5, tmp23, 0.0)
    tmp25 = tl.load(in_ptr1 + (x7), tmp14 & xmask, other=0.0)
    tmp26 = tl.where(tmp14, tmp25, 0.0)
    tmp27 = tl.load(in_ptr2 + (x7), tmp14 & xmask, other=0.0)
    tmp28 = tl.where(tmp14, tmp27, 0.0)
    tmp30 = tl.where(tmp14, tmp28, tmp29)
    tmp31 = tl.where(tmp14, tmp26, tmp30)
    tmp32 = tl.where(tmp5, tmp24, tmp31)
    tl.store(out_ptr0 + (x7), tmp32, xmask)
''')
