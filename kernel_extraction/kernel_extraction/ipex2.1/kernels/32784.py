

# Original file: ./DALLE2_pytorch___60.0/DALLE2_pytorch___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/xn/cxnflyt3h26n56m4gsbdg2knb77w2edflsrksi42g2vdut5df2hl.py
# Source Nodes: [invert, masked_fill], Original ATen: [aten.bitwise_not, aten.masked_fill]
# invert => bitwise_not
# masked_fill => full_default, where
triton_poi_fused_bitwise_not_masked_fill_17 = async_compile.triton('triton_poi_fused_bitwise_not_masked_fill_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_masked_fill_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_not_masked_fill_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 77
    x3 = (xindex // 512)
    x0 = xindex % 512
    x2 = (xindex // 39424)
    x4 = xindex
    tmp9 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0 + (512*x2) + (1024*x1)), xmask)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 77, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-1) + x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 == tmp1
    tmp8 = tl.where(tmp5, tmp7, True)
    tmp10 = tmp9 != tmp1
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 == 0
    tmp14 = 0.0
    tmp15 = tl.where(tmp12, tmp14, tmp13)
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''')
