

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/zj/czjqbtph2n74u4wdd4dfx37245efp4mnz5suorltttcq2v26p37l.py
# Source Nodes: [setitem_35], Original ATen: [aten.select_scatter, aten.slice_scatter]
# setitem_35 => select_scatter_31, slice_scatter_177
triton_poi_fused_select_scatter_slice_scatter_41 = async_compile.triton('triton_poi_fused_select_scatter_slice_scatter_41', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_scatter_slice_scatter_41', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_select_scatter_slice_scatter_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4160000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x5 = (xindex // 104)
    x4 = (xindex // 20800)
    x6 = xindex % 20800
    x7 = xindex
    tmp27 = tl.load(in_ptr3 + (42640 + x6 + (21216*x4)), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (2*x2) + (50*x5)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = 2 + x4
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 202, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp2
    tmp14 = tl.load(in_ptr1 + (208 + x6 + (21216*x4)), tmp13 & xmask, other=0.0)
    tmp15 = tl.where(tmp13, tmp14, 0.0)
    tmp16 = tl.load(in_ptr2 + (208 + x6 + (21216*x4)), tmp13 & xmask, other=0.0)
    tmp17 = tl.where(tmp13, tmp16, 0.0)
    tmp18 = tl.load(in_ptr3 + (42640 + x6 + (21216*x4)), tmp2 & xmask, other=0.0)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp15, tmp19)
    tmp21 = tl.where(tmp5, tmp6, tmp20)
    tmp22 = tl.where(tmp2, tmp21, 0.0)
    tmp23 = tl.load(in_ptr1 + (208 + x6 + (21216*x4)), tmp12 & xmask, other=0.0)
    tmp24 = tl.where(tmp12, tmp23, 0.0)
    tmp25 = tl.load(in_ptr2 + (208 + x6 + (21216*x4)), tmp12 & xmask, other=0.0)
    tmp26 = tl.where(tmp12, tmp25, 0.0)
    tmp28 = tl.where(tmp12, tmp26, tmp27)
    tmp29 = tl.where(tmp12, tmp24, tmp28)
    tmp30 = tl.where(tmp2, tmp22, tmp29)
    tl.store(out_ptr0 + (x7), tmp30, xmask)
''')
