

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/c5/cc5k3vqzvfmonvfd4cqrc4xdniuvwxmc7dstw232s4yfez6xuf2j.py
# Source Nodes: [setitem_32], Original ATen: [aten.select_scatter, aten.slice_scatter]
# setitem_32 => select_scatter_25, slice_scatter_159
triton_poi_fused_select_scatter_slice_scatter_38 = async_compile.triton('triton_poi_fused_select_scatter_slice_scatter_38', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_scatter_slice_scatter_38', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_select_scatter_slice_scatter_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4160000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4) % 26
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x5 = (xindex // 104)
    x4 = (xindex // 20800)
    x3 = (xindex // 104) % 200
    x8 = xindex
    x6 = xindex % 20800
    tmp34 = tl.load(in_ptr2 + (42640 + x6 + (21216*x4)), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (2*x2) + (50*x5)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = 2 + x4
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 202, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp2
    tmp14 = 2 + x3
    tmp15 = tmp14 >= tmp8
    tmp16 = tmp14 < tmp10
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17 & tmp13
    tmp19 = tl.load(in_ptr1 + (x8), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.load(in_ptr2 + (42640 + x6 + (21216*x4)), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp22, 0.0)
    tmp24 = tl.load(in_ptr2 + (42640 + x6 + (21216*x4)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.where(tmp12, tmp23, tmp24)
    tmp26 = tl.where(tmp5, tmp6, tmp25)
    tmp27 = tl.where(tmp2, tmp26, 0.0)
    tmp28 = tmp17 & tmp12
    tmp29 = tl.load(in_ptr1 + (x8), tmp28 & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.where(tmp28, tmp29, 0.0)
    tmp31 = tl.load(in_ptr2 + (42640 + x6 + (21216*x4)), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp32 = tl.where(tmp17, tmp30, tmp31)
    tmp33 = tl.where(tmp12, tmp32, 0.0)
    tmp35 = tl.where(tmp12, tmp33, tmp34)
    tmp36 = tl.where(tmp2, tmp27, tmp35)
    tl.store(out_ptr0 + (x8), tmp36, xmask)
''')
